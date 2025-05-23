# visualize_kg.py

import os
import json
import torch
import networkx as nx
from pyvis.network import Network
from code import REModel, tokenizer, predicate2id, id2predicate, predicate2type, DEVICE  # 按需调整 import
from pyvis.network import Network
import webbrowser

def load_model(model_path: str):
    model = REModel().to(DEVICE)
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.eval()
    return model

def extract_all_spoes(model, texts, threshold=0.2):
    from code import extract_spoes  # reuse你训练脚本里的抽取函数
    all_spoes = []
    for text in texts:
        spoes = extract_spoes(text, model, DEVICE, threshold)
        all_spoes.extend(spoes)
    return all_spoes

def build_graph(triples):
    G = nx.DiGraph()
    for subj, pred, obj in triples:
        # 添加节点（如果已存在则忽略）
        if not G.has_node(subj):
            G.add_node(subj, title=subj,
                       url=f"https://zh.wikipedia.org/wiki/{subj}")
        if not G.has_node(obj):
            G.add_node(obj, title=obj,
                       url=f"https://zh.wikipedia.org/wiki/{obj}")
        # 添加有向边，label 为关系
        G.add_edge(subj, obj, label=pred, title=pred)
    return G

def visualize_graph(G, output_html="kg.html"):
    net = Network(height="800px", width="100%", directed=True)
    net.from_nx(G)
    # 给每个节点加上点击链接
    for node in net.nodes:
        node["shape"] = "ellipse"
        node["font"] = {"size": 16}
        node["title"] = f"<a href='{node['url']}' target='_blank'>{node['id']}</a>"
    net.show_buttons(filter_=['physics'])
    # 直接写 HTML 文件，避免调用 .show() 导致模板问题
    net.write_html(output_html)
    print(f"已生成交互式知识图谱：{output_html}")
    # 可选：自动在默认浏览器中打开
    webbrowser.open(output_html, new=2)
def main():
    # 1. 加载模型
    model = load_model("CMeIE/bad.pth")

    # 2. 准备待抽取文本，可替换为任意文本列表
    #    这里我们加载验证集中的文本
    texts = []
    with open("CMeIE/CMeIE_dev.json", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            texts.append(obj["text"])

    # 3. 抽取所有三元组
    triples = extract_all_spoes(model, texts, threshold=0.2)
    print(f"共抽取到 {len(triples)} 条三元组")

    # 4. 构建图并可视化
    G = build_graph(triples)
    visualize_graph(G, output_html="kg.html")

if __name__ == "__main__":
    main()
