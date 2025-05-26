import os
import json
import unicodedata

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_pretrained_bert import BertModel, BertTokenizer
from flask import Flask, request, jsonify

# ——————————————— 全局配置 ———————————————
BERT_PATH   = "chinese_roberta_wwm_ext_pytorch/"
SCHEMA_PATH = "CMeIE/schema.json"
WEIGHT_PATH = "CMeIE/roberta_best.pth"  # 或者 roberta_best.pth
MAXLEN      = 256
DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ——————————————— 读取 Schema ———————————————
predicate2id = {}
id2predicate = {}
predicate2type = {}
with open(SCHEMA_PATH, encoding='utf-8') as f:
    for line in f:
        obj = json.loads(line)
        pid = obj['predicate']
        predicate2type[pid] = (obj['subject_type'], obj['object_type'])
        if pid not in predicate2id:
            idx = len(predicate2id)
            predicate2id[pid] = idx
            id2predicate[idx] = pid

# ——————————————— OurTokenizer 定义 ———————————————
class OurTokenizer(BertTokenizer):
    def tokenize(self, text):
        R = []
        for c in text:
            if c in self.vocab:
                R.append(c)
            elif self._is_whitespace(c):
                R.append('[unused1]')
            else:
                R.append('[UNK]')
        return R

    def _is_whitespace(self, char):
        if char in {" ", "\t", "\n", "\r"}:
            return True
        return unicodedata.category(char) == "Zs"

# ——————————————— 模型定义 ———————————————
class BertLayerNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-12):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.bias   = nn.Parameter(torch.zeros(hidden_size))
        self.eps    = eps
    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        var  = (x - mean).pow(2).mean(-1, keepdim=True)
        x_norm = (x - mean) / torch.sqrt(var + self.eps)
        return self.weight * x_norm + self.bias

class REModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.bert       = BertModel.from_pretrained(BERT_PATH)
        self.sub_output = nn.Linear(768, 2)
        self.sub_pos_emb= nn.Embedding(MAXLEN, 768)
        self.linear     = nn.Linear(768, 768)
        self.layernorm  = BertLayerNorm(768)
        self.relu       = nn.ReLU()
        self.obj_output = nn.Linear(768, len(predicate2id) * 2)

    def forward(self, token_ids, seg_ids, sub_ids=None):
        out, _ = self.bert(token_ids, token_type_ids=seg_ids,
                           output_all_encoded_layers=False)
        logits_sub = self.sub_output(out)
        if sub_ids is None:
            return logits_sub, None
        # 融入 subject 信息
        start_emb = self.sub_pos_emb(sub_ids[:, :1])
        end_emb   = self.sub_pos_emb(sub_ids[:, 1:])
        start_out = torch.gather(out, 1,
                        sub_ids[:, :1].unsqueeze(-1).expand(-1, -1, out.size(-1)))
        end_out   = torch.gather(out, 1,
                        sub_ids[:, 1:].unsqueeze(-1).expand(-1, -1, out.size(-1)))
        out1 = out + start_emb + start_out + end_emb + end_out
        out1 = self.layernorm(out1)
        out1 = F.dropout(out1, p=0.5, training=self.training)
        out1 = self.relu(self.linear(out1))
        out1 = F.dropout(out1, p=0.4, training=self.training)
        logits_obj = self.obj_output(out1)
        obj_preds  = logits_obj.view(-1, out1.size(1), len(predicate2id), 2)
        return logits_sub, obj_preds

# ——————————————— 初始化 ———————————————
tokenizer = OurTokenizer(vocab_file=os.path.join(BERT_PATH, "vocab.txt"))
model = REModel().to(DEVICE)
state = torch.load(WEIGHT_PATH, map_location=DEVICE)
model.load_state_dict(state)

# ——————————————— 工具函数 ———————————————
def pad_seq(seq):
    return seq[:MAXLEN] if len(seq) >= MAXLEN else seq + [0] * (MAXLEN - len(seq))

# ——————————————— 推理函数 extract_spoes ———————————————
def extract_spoes(text, model, threshold=0.01):
    model.eval()
    # 文本截断 + tokenize
    if len(text) > MAXLEN - 2:
        text = text[:MAXLEN - 2]
    tokens = ["[CLS]"] + tokenizer.tokenize(text) + ["[SEP]"]
    token_ids = tokenizer.convert_tokens_to_ids(tokens)
    seg_ids   = [0] * len(token_ids)
    pid = torch.LongTensor([pad_seq(token_ids)]).to(DEVICE)
    sid = torch.LongTensor([pad_seq(seg_ids)]).to(DEVICE)
    # 1. 预测 subject
    with torch.no_grad():
        logits_sub, _ = model(pid, sid, None)
        probs_sub     = torch.sigmoid(logits_sub)[0].cpu().numpy()
    starts = np.where(probs_sub[:, 0] > threshold)[0]
    ends   = np.where(probs_sub[:, 1] > threshold)[0]
    subjects = []
    for s in starts:
        e_cands = ends[ends >= s]
        if len(e_cands) > 0:
            subjects.append((s, e_cands[0]))
    if not subjects:
        return []
    # 2. 预测 object
    B = len(subjects)
    ids_b = np.repeat([pad_seq(token_ids)], B, 0)
    seg_b = np.repeat([pad_seq(seg_ids)], B, 0)
    sub_b = np.array(subjects, dtype=np.int64)
    with torch.no_grad():
        _, obj_preds = model(
            torch.LongTensor(ids_b).to(DEVICE),
            torch.LongTensor(seg_b).to(DEVICE),
            torch.LongTensor(sub_b).to(DEVICE)
        )
        obj_preds = obj_preds.cpu().numpy()
    spoes = []
    for (si, sj), pred in zip(subjects, obj_preds):
        starts_o, preds = np.where(pred[:, :, 0] > threshold)
        ends_o, ends_p  = np.where(pred[:, :, 1] > threshold)
        for o_s, p_id in zip(starts_o, preds):
            valid = ends_o[(ends_o >= o_s) & (ends_p == p_id)]
            if not len(valid): continue
            o_e = valid[0]
            spoes.append({
                'subject': text[si-1:sj],
                'predicate': id2predicate[p_id],
                'object': text[o_s-1:o_e]
            })
    return spoes

# ——————————————— Flask 服务 ———————————————
app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json(force=True)
    text = data.get('text', '').strip()
    if not text:
        return jsonify({'error': "请在 JSON body 中提供 'text' 字段"}), 400
    spoes = extract_spoes(text, model)
    return jsonify({'text': text, 'spo_list_pred': spoes})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
