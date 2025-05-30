import json
import os
import time
import unicodedata

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_pretrained_bert import BertModel, BertTokenizer
from torch.utils.data import DataLoader, Dataset
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm

# ——————————————— 全局设置 ———————————————
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
torch.backends.cudnn.benchmark = True  # cuDNN 会自动寻找最优的卷积算法，加速训练。

BERT_PATH = "chinese_roberta_wwm_ext_pytorch/"
MAXLEN = 256  # MAXLEN：每条文本最大截断（或填充）长度，防止过长。
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


train_l = []
val_f1_r = []

def record():
    os.makedirs("record", exist_ok=True)
    with open("record/val_f1_1e4.txt", "w", encoding="utf-8") as f1, \
         open("record/loss_1e4.txt", "w", encoding="utf-8") as f2:
        for (e, f1v), (_, loss) in zip(val_f1_r, train_l):
            f1.write(f"{e}\t{f1v:.6f}\n")
            f2.write(f"{e}\t{loss:.6f}\n")

# ——————————————— 数据加载 & 预处理 ———————————————
"""
文件格式：每一行都是一个 JSON 对象，包含 "text"（一段文字）和 "spo_list"（其中的关系三元组）。

输出格式：列表 D，每个元素是一个字典，有:
    text：原文
    spo_list：一个三元组列表，例：[("糖尿病","治疗方法","注射胰岛素"), ...]
"""
def load_data(filename):
    D = []
    with open(filename, encoding='utf-8') as f:
        for l in tqdm(f, desc=f"Loading {filename}"):
            obj = json.loads(l)
            # 将每个 SPO 三元组拆开，存成 (subject, predicate, object) 的形式
            d = {'text': obj['text'], 'spo_list': []}
            for spo in obj['spo_list']:
                for _, v in spo['object'].items():
                    d['spo_list'].append((spo['subject'], spo['predicate'], v))
            D.append(d)
    return D


train_data = load_data('CMeIE/CMeIE_train.json')
valid_data = load_data('CMeIE/CMeIE_dev.json')

# 过滤过长实例
# BERT 的输入长度有限，如果三元组的 subject 或 object 在 token 化后超出 MAXLEN-2
# 就丢掉这条样本，保证所有样本都能跑完整个模型。
train_data_new = []
for d in tqdm(train_data, desc="Filtering train samples"):
    ok = True
    for s, p, o in d['spo_list']:
        si = d['text'].find(s)
        oi = d['text'].find(o)
        if si < 0 or oi < 0 or si + len(s) > MAXLEN - 2 or oi + len(o) > MAXLEN - 2:
            ok = False
            break
    if ok:
        train_data_new.append(d)
print(f"Kept {len(train_data_new)} / {len(train_data)} train samples")

# 读取 schema
# 把 schema 文件里的“谓词”信息都读进来，建立几个映射字典，方便后面把每个谓词都转成数字 ID，以及查它的类型
predicate2id = {}
id2predicate = {}
predicate2type = {}
with open('CMeIE/schema.json', encoding='utf-8') as f:
    for line in f:
        obj = json.loads(line)
        pid = obj['predicate']
        predicate2type[pid] = (obj['subject_type'], obj['object_type'])
        if pid not in predicate2id:
            idx = len(predicate2id)
            predicate2id[pid] = idx
            id2predicate[idx] = pid

# 自定义 tokenizer：保证中文字符级别切分
class OurTokenizer(BertTokenizer):
    def tokenize(self, text):
        # 按字（Character）级别切分，保证每个中文字符都能对上 BERT 的词表
        # 把一句话切成一个个“字”，然后再去查词表，这样就不会把生僻词给忽略掉。
        R = []
        for c in text:
            if c in self.vocab:
                R.append(c)  # 如果这个字在词表里，就当一个 token
            elif self._is_whitespace(c):
                R.append('[unused1]')  # 空白字符统一用 [unused1] 占位
            else:
                R.append('[UNK]')  # 词表里没有的字就打 [UNK]
        return R

    def _is_whitespace(self, char):
        if char in {" ", "\t", "\n", "\r"}:
            return True
        return unicodedata.category(char) == "Zs"

tokenizer = OurTokenizer(vocab_file=os.path.join(BERT_PATH, "vocab.txt"))

def preprocess_dataset(data):
    """预计算 token_ids, seg_ids, sub_ids, sub_labels, obj_labels"""
    records = []
    for d in tqdm(data, desc="Preprocessing"):

        text = d['text']
        tokens = ["[CLS]"] + tokenizer.tokenize(text) + ["[SEP]"]
        token_ids = tokenizer.convert_tokens_to_ids(tokens)
        seg_ids = [0] * len(token_ids)

        spoes = {}
        for s, p, o in d['spo_list']:
            s_tokens = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(s))
            o_tokens = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(o))
            si = search(s_tokens, token_ids)
            oi = search(o_tokens, token_ids)
            if si >= 0 and oi >= 0:
                sub_span = (si, si + len(s_tokens) - 1)
                obj_span = (oi, oi + len(o_tokens) - 1, predicate2id[p])
                spoes.setdefault(sub_span, []).append(obj_span)

        if not spoes:
            continue

        # 构造 label 张量
        sub_labels = np.zeros((len(token_ids), 2), dtype=np.float32)
        for (si, sj) in spoes:
            sub_labels[si, 0] = 1
            sub_labels[sj, 1] = 1

        # 随机选一个 subject
        starts, ends = zip(*spoes.keys())
        rand_idx = np.random.choice(len(starts))
        sub_ids = (starts[rand_idx], ends[rand_idx])

        obj_labels = np.zeros((len(token_ids), len(predicate2id), 2), dtype=np.float32)
        for oi, oj, pid in spoes.get(sub_ids, []):
            obj_labels[oi, pid, 0] = 1
            obj_labels[oj, pid, 1] = 1

        # Padding / Truncate
        arr1d = lambda x, pad: (
            x[:MAXLEN] if len(x) >= MAXLEN
            else np.concatenate([x, np.full((MAXLEN - len(x),), pad)], axis=0)
        )
        arr2d = lambda x, pad: (
            x[:MAXLEN] if x.shape[0] >= MAXLEN
            else np.concatenate([x, np.full((MAXLEN - x.shape[0],) + x.shape[1:], pad)], axis=0)
        )

        token_ids = arr1d(np.array(token_ids, dtype=np.int64), 0)
        seg_ids   = arr1d(np.array(seg_ids, dtype=np.int64), 0)
        sub_labels= arr2d(sub_labels, 0.0)
        obj_labels= arr2d(obj_labels, 0.0)

        records.append((token_ids, seg_ids, np.array(sub_ids, dtype=np.int64),
                        sub_labels, obj_labels))
    return records


"""
token_ids：一句话对应的 ID 序列

seg_ids：全 0 的句子分割标志

sub_ids：当前训练要定位的那个 subject 的起始和结束索引

sub_labels：所有主语起止的监督信号

obj_labels：在给定主语条件下，各关系下宾语起止的监督信号
"""

# 辅助：在 list 中搜索子序列
def search(pattern, sequence):
    n = len(pattern)
    for i in range(len(sequence) - n + 1):
        if sequence[i:i+n] == pattern:
            return i
    return -1

train_records = preprocess_dataset(train_data_new)
valid_records = preprocess_dataset(valid_data)

class TorchDataset(Dataset):
    def __init__(self, records):
        self.records = records
    def __len__(self):
        return len(self.records)
    def __getitem__(self, idx):
        t_ids, s_ids, sb_ids, sb_lab, ob_lab = self.records[idx]
        return (torch.LongTensor(t_ids),
                torch.LongTensor(s_ids),
                torch.LongTensor(sb_ids),
                torch.FloatTensor(sb_lab),
                torch.FloatTensor(ob_lab))

train_loader = DataLoader(
    TorchDataset(train_records),
    batch_size=8,
    shuffle=True,
    num_workers=2,
    pin_memory=True,
    drop_last=True
)
"""
Dataset：封装上一步的 records，实现 __len__ 和 __getitem__。
DataLoader：每次取出一个 batch（这里是 8 条样本），并行加载、随机打乱、自动拼成一个大张量。
"""

valid_loader = DataLoader(
    TorchDataset(valid_records),
    batch_size=25,
    shuffle=False,
    num_workers=4,
    pin_memory=True
)

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
        # 1) 加载预训练 BERT
        # 2) 定义一个 Linear+Embedding+LayerNorm+ReLU 等层，用来把 subject 信息融入后再去预测 object
        super().__init__()
        self.bert = BertModel.from_pretrained(BERT_PATH)
        for p in self.bert.parameters():
            p.requires_grad = True
        self.sub_output   = nn.Linear(768, 2)
        self.sub_pos_emb  = nn.Embedding(MAXLEN, 768)
        self.linear       = nn.Linear(768, 768)
        self.layernorm    = BertLayerNorm(768)
        self.relu         = nn.ReLU()
        self.obj_output   = nn.Linear(768, len(predicate2id)*2)

    def forward(self, token_ids, seg_ids, sub_ids=None):
        # 1) 先跑 BERT，得到每个位置的上下文表示 out
        # 2) sub_output：直接对 out 做线性变换，得到每个位置是不是 subject 开始/结束的 logits_sub
        # 3) 如果没有给 sub_ids（预测时就不传），就提前返回只算 subject
        # 4) 给了 sub_ids（训练/抽取 object 阶段），再把 subject 的位置信息拼进去，做 layernorm、dropout、ReLU
        # 5) obj_output：再做一次线性变换，输出每个位置、每个 predicate、是否为 object 起止的 logits_obj

        # 1) 先跑 BERT
        out, _ = self.bert(token_ids, token_type_ids=seg_ids,
                           output_all_encoded_layers=False)
        # 2) Subject 部分直接输出 logits
        logits_sub = self.sub_output(out)  # [B, L, 2]

        # ←—— 当 sub_ids=None 时，也返回两项，第二项用 None 占位
        if sub_ids is None:
            return logits_sub, None

        # 3) 融入 subject 信息
        start_emb = self.sub_pos_emb(sub_ids[:, :1])  # [B,1,768]
        end_emb   = self.sub_pos_emb(sub_ids[:, 1:])  # [B,1,768]
        start_out = torch.gather(
            out, 1,
            sub_ids[:, :1].unsqueeze(-1).expand(-1, -1, out.size(-1))
        )  # [B,1,768]
        end_out   = torch.gather(
            out, 1,
            sub_ids[:, 1:].unsqueeze(-1).expand(-1, -1, out.size(-1))
        )  # [B,1,768]

        out1 = out + start_emb + start_out + end_emb + end_out
        out1 = self.layernorm(out1)
        out1 = F.dropout(out1, p=0.5, training=self.training)
        out1 = self.relu(self.linear(out1))
        out1 = F.dropout(out1, p=0.4, training=self.training)

        # 4) Object 部分同样只输出 logits
        logits_obj = self.obj_output(out1)           # [B, L, P*2]
        obj_preds  = logits_obj.view(
            -1, out1.size(1), len(predicate2id), 2
        )  # [B, L, P, 2]

        return logits_sub, obj_preds

net = REModel().to(DEVICE)
"""
for name, param in net.bert.named_parameters():
    # 冻结除了最后两层 encoder 的所有层
    if not name.startswith("encoder.layer.10") and not name.startswith("encoder.layer.11"):
        param.requires_grad = False
"""

try:
    net = torch.compile(net)  # PyTorch 2.0+ 编译加速
except:
    pass

optimizer = torch.optim.Adam(net.parameters(), lr=1e-5)
scaler = GradScaler()

# ——————————————— 损失计算 ———————————————
# 损失函数与优化器

def compute_loss(logits_sub, logits_obj, sub_labels, obj_labels, token_ids):
    # 1) 用 BCEWithLogitsLoss（对二分类做交叉熵）算 subject 的损失
    # 2) 同样算 object 的损失
    # 3) 对 token_ids>0（真正的词，不是填充）位置做掩码，最后加起来归一化
    mask = (token_ids > 0).float()  # [B, L]

    # subject loss
    # BCEWithLogitsLoss 会在内部做 sigmoid，然后计算交叉熵
    loss_sub = F.binary_cross_entropy_with_logits(
        logits_sub, sub_labels, reduction='none'
    )  # [B, L, 2]
    loss_sub = loss_sub.mean(-1) * mask  # [B, L]
    loss_sub = loss_sub.sum() / mask.sum()

    # object loss
    loss_obj = F.binary_cross_entropy_with_logits(
        logits_obj, obj_labels, reduction='none'
    )  # [B, L, P, 2]
    loss_obj = loss_obj.mean((-1, -2)) * mask  # [B, L]
    loss_obj = loss_obj.sum() / mask.sum()

    return loss_sub + loss_obj
# ——————————————— 评估 & 推理 ———————————————

def pad_seq(seq):
    if len(seq) >= MAXLEN:
        return seq[:MAXLEN]
    return seq + [0]*(MAXLEN-len(seq))

def extract_spoes(text, model, device, threshold=0.2):
    """
    抽取三元组。假设 forward 总是返回 (logits_sub, obj_preds)。
    """
    model.eval()
    with torch.no_grad():
        # 1. 文本截断 + tokenize
        if len(text) > MAXLEN - 2:
            text = text[:MAXLEN - 2]
        tokens = ["[CLS]"] + tokenizer.tokenize(text) + ["[SEP]"]
        token_ids = tokenizer.convert_tokens_to_ids(tokens)
        seg_ids   = [0] * len(token_ids)
        # padding
        pid_tensor = torch.LongTensor([pad_seq(token_ids)]).to(device)
        sid_tensor = torch.LongTensor([pad_seq(seg_ids)]).to(device)

        # 2. forward 得到 logits_sub, obj_preds
        logits_sub, _ = model(pid_tensor, sid_tensor, None)  # logits_sub: [1, L, 2]

        # 3. 把 logits 转成概率，然后去掉 batch 维度
        probs_sub = torch.sigmoid(logits_sub)                # [1, L, 2]
        sub_preds = probs_sub[0].cpu().numpy()               # [L, 2]

        # 4. 找 subject start/end
        starts = np.where(sub_preds[:, 0] > threshold)[0]
        ends   = np.where(sub_preds[:, 1] > threshold)[0]

        subjects = []
        for s in starts:
            e_cands = ends[ends >= s]
            if len(e_cands) > 0:
                subjects.append((s, e_cands[0]))
        if not subjects:
            return []

        # 5. 对每个 subject 批量抽 obj
        # 构造批量 input
        B = len(subjects)
        ids_batch = np.repeat([pad_seq(token_ids)], B, 0)
        seg_batch = np.repeat([pad_seq(seg_ids)], B, 0)
        sub_batch = np.array(subjects, dtype=np.int64)

        _, obj_preds = model(
            torch.LongTensor(ids_batch).to(device),
            torch.LongTensor(seg_batch).to(device),
            torch.LongTensor(sub_batch).to(device)
        )
        obj_preds = obj_preds.cpu().numpy()  # [B, L, P, 2]

        # 6. 根据 obj_preds threshold，组装 (sub,pred,obj) 列表
        spoes = []
        for (si, sj), pred in zip(subjects, obj_preds):
            # pred: [L, P, 2]
            starts_o, preds = np.where(pred[:, :, 0] > threshold)
            ends_o, ends_pred = np.where(pred[:, :, 1] > threshold)
            for o_start, p_id in zip(starts_o, preds):
                # 找第一个合法 end
                valid_ends = ends_o[(ends_o >= o_start) & (ends_pred == p_id)]
                if len(valid_ends) == 0:
                    continue
                o_end = valid_ends[0]
                spoes.append((
                    text[si-1:sj],          # subject text
                    id2predicate[p_id],     # predicate str
                    text[o_start-1:o_end]   # object text
                ))
        return spoes


def evaluate(data, model, device):
    model.eval()
    X=Y=Z=1e-10
    with open('CMeIE/dev_pred.json','w',encoding='utf-8') as fout:
        for d in tqdm(data, desc="Evaluating"):
            R = set(extract_spoes(d['text'], model, device))
            T = set(d['spo_list'])
            X += len(R & T)
            Y += len(R)
            Z += len(T)
            s = json.dumps({
                'text': d['text'],
                'spo_list': list(T),
                'spo_list_pred': list(R),
                'new': list(R-T),
                'lack': list(T-R)
            }, ensure_ascii=False)
            fout.write(s+"\n")
    f1 = 2*X/(Y+Z)
    return f1, X/Y, X/Z

# ——————————————— 训练循环 ———————————————

def train(model, train_loader, valid_data, epochs, device):
    best_f1 = 0.2
    for epoch in range(1, epochs+1):
        model.train()
        epoch_loss = 0.0
        t0 = time.time()
        # 用 tqdm 包装 DataLoader
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}", dynamic_ncols=True)
        for batch_idx, batch in enumerate(pbar, start=1):
            # 1) 清零梯度
            # 2) 前向、算损失、反向、梯度更新
            # 3) 记录平均损失
            # 4) 切到 eval 模式，用 evaluate() 在验证集上算 F1
            # 5) 如果 F1 更好，就保存为最佳模型，否则保存为“坏”模型
            token_ids, seg_ids, sub_ids, sub_lab, obj_lab = [x.to(device) for x in batch]
            optimizer.zero_grad()
            with autocast():
                sub_pred, obj_pred = model(token_ids, seg_ids, sub_ids)
                loss = compute_loss(sub_pred, obj_pred, sub_lab, obj_lab, token_ids)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            epoch_loss += loss.item()
            avg_loss = epoch_loss / batch_idx
            elapsed = time.time() - t0
            # 每个 batch 结束后更新进度条上的信息
            # 每次用 8 条样本同时计算，算出平均梯度。
            pbar.set_postfix({
                'avg_loss': f"{avg_loss:.4f}",
                'elapsed': f"{elapsed:.1f}s"
            })

        train_l.append((epoch, avg_loss))

        # 验证阶段
        # 每个 epoch 结束后，在开发集上做一次完整的抽取，计算精确率、召回率、F1，用来判断模型是否进步。
        f1, prec, rec = evaluate(valid_data, model, device)
        val_f1_r.append((epoch, f1))
        print(f"Epoch {epoch} done | f1: {f1:.4f}, precision: {prec:.4f}, recall: {rec:.4f}")
        if f1 > best_f1:
            best_f1 = f1
            torch.save(model.state_dict(), "CMeIE/roberta_best.pth")
            print("  Saved new best model.")
        else:
            torch.save(model.state_dict(), "CMeIE/bad.pth")
            print("  Saved new bad model.")

if __name__ == "__main__":
    train(net, train_loader, valid_data, epochs=10, device=DEVICE)

    # 预测测试集并写入文件
    def combine_spoes(spoes):
        d = {}
        for s,p,o in spoes:
            d.setdefault((s,p), {})['@value'] = o
        return [(s,p,v) for (s,p),v in d.items()]

    with open('CMeIE/CMeIE_test.json','r',encoding='utf-8') as fin, \
         open('CMeIE/RE_pred.json','w',encoding='utf-8') as fout:
        for line in tqdm(fin, desc="Predicting test"):
            obj = json.loads(line)
            spoes = combine_spoes(extract_spoes(obj['text'], net, DEVICE))
            obj['spo_list'] = [
                {
                    'subject': s,
                    'subject_type': predicate2type[p][0],
                    'predicate': p,
                    'object': spo_o,
                    'object_type': {k: predicate2type[p][1] for k in spo_o}
                } for s,p,spo_o in spoes
            ]
            fout.write(json.dumps(obj, ensure_ascii=False)+"\n")

    record()
