import json
import os
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from pytorch_pretrained_bert import BertModel, BertTokenizer
import numpy as np

# ——————————————— Settings ———————————————
BERT_PATH = "chinese_roberta_wwm_ext_pytorch/"
MAXLEN    = 256
DEVICE    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
WEIGHT    = "CMeIE/bad.pth"   # or "CMeIE/roberta_best.pth"

# ——————————————— Read schema exactly as in training ———————————————
def load_schema(schema_path):
    predicate2id   = {}
    id2predicate   = {}
    predicate2type = {}
    with open(schema_path, encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            pid = obj["predicate"]
            predicate2type[pid] = (obj["subject_type"], obj["object_type"])
            if pid not in predicate2id:
                idx = len(predicate2id)
                predicate2id[pid] = idx
                id2predicate[idx] = pid
    return predicate2id, id2predicate, predicate2type

predicate2id, id2predicate, predicate2type = load_schema("CMeIE/schema.json")

# ——————————————— Model definition (same as training) ———————————————
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
    def __init__(self, predicate_num):
        super().__init__()
        self.bert       = BertModel.from_pretrained(BERT_PATH)
        self.sub_output = nn.Linear(768, 2)
        self.sub_pos_emb= nn.Embedding(MAXLEN, 768)
        self.linear     = nn.Linear(768, 768)
        self.layernorm  = BertLayerNorm(768)
        self.relu       = nn.ReLU()
        self.obj_output = nn.Linear(768, predicate_num * 2)

    def forward(self, token_ids, seg_ids, sub_ids=None):
        out, _     = self.bert(token_ids, token_type_ids=seg_ids,
                               output_all_encoded_layers=False)
        logits_sub = self.sub_output(out)  # [B, L, 2]
        if sub_ids is None:
            return logits_sub, None

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

# ——————————————— Helpers ———————————————
def pad_seq(seq):
    return seq[:MAXLEN] + [0] * max(0, MAXLEN - len(seq))

def extract_spoes(text, model, tokenizer, threshold=0.2):
    model.eval()
    # tokenize + pad
    if len(text) > MAXLEN - 2:
        text = text[:MAXLEN-2]
    tokens   = ["[CLS]"] + tokenizer.tokenize(text) + ["[SEP]"]
    token_ids= tokenizer.convert_tokens_to_ids(tokens)
    seg_ids  = [0] * len(token_ids)
    ids      = pad_seq(token_ids)
    seg      = pad_seq(seg_ids)

    tid = torch.LongTensor([ids]).to(DEVICE)
    sid = torch.LongTensor([seg]).to(DEVICE)

    # subject prediction
    with torch.no_grad():
        logits_sub, _ = model(tid, sid, None)
        probs_sub     = torch.sigmoid(logits_sub)[0].cpu().numpy()
    starts = np.where(probs_sub[:,0] > threshold)[0]
    ends   = np.where(probs_sub[:,1] > threshold)[0]
    subjects = []
    for s in starts:
        e_cands = ends[ends >= s]
        if len(e_cands):
            subjects.append((s, e_cands[0]))
    if not subjects:
        return []

    # object prediction
    B = len(subjects)
    ids_batch = np.repeat([ids], B, axis=0)
    seg_batch = np.repeat([seg], B, axis=0)
    sub_batch = np.array(subjects, dtype=np.int64)
    with torch.no_grad():
        _, obj_preds = model(
            torch.LongTensor(ids_batch).to(DEVICE),
            torch.LongTensor(seg_batch).to(DEVICE),
            torch.LongTensor(sub_batch).to(DEVICE)
        )
        obj_preds = obj_preds.cpu().numpy()

    spo = []
    for (si, sj), pred in zip(subjects, obj_preds):
        starts_o, ps = np.where(pred[:,:,0] > threshold)
        ends_o,  pe = np.where(pred[:,:,1] > threshold)
        for o_start, p_id in zip(starts_o, ps):
            valid_ends = ends_o[(ends_o >= o_start) & (pe == p_id)]
            if not len(valid_ends):
                continue
            o_end = valid_ends[0]
            subj_text = text[si-1: sj]
            obj_text  = text[o_start-1: o_end]
            pred_str  = id2predicate[p_id]
            spo.append((subj_text, pred_str, obj_text))
    return spo

# ——————————————— Evaluation ———————————————
def evaluate(data, model, tokenizer):
    X=Y=Z=1e-10
    for text, gold in tqdm(data, desc="Testing"):
        pred = extract_spoes(text, model, tokenizer)
        R = set(pred)
        T = set(gold)
        X += len(R & T)
        Y += len(R)
        Z += len(T)
    f1    = 2*X/(Y+Z)
    prec  = X/Y
    rec   = X/Z
    print(f"\nTest → F1: {f1:.4f}, Precision: {prec:.4f}, Recall: {rec:.4f}")

# ——————————————— Main ———————————————
if __name__ == "__main__":
    # 1. tokenizer
    tokenizer = BertTokenizer(vocab_file=os.path.join(BERT_PATH, "vocab.txt"))

    # 2. model & load weights
    model = REModel(len(predicate2id)).to(DEVICE)
    state = torch.load(WEIGHT, map_location=DEVICE)
    model.load_state_dict(state)

    # 3. load validation data
    valid = []
    with open("CMeIE/CMeIE_dev.json", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            spo = []
            for spo_o in obj["spo_list"]:
                for _, v in spo_o["object"].items():
                    spo.append((spo_o["subject"], spo_o["predicate"], v))
            valid.append((obj["text"], spo))

    # 4. evaluate
    evaluate(valid, model, tokenizer)
