import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

fig, ax = plt.subplots(figsize=(6, 6))
ax.axis('off')

# 四个节点标签与说明
nodes = ['查询处理', '文档检索', '结果融合', '文本生成']
descriptions = [
    '将自然语言问题编码为查询向量',
    '基于查询向量检索相关文档',
    '拼接问题与文档或使用注意力融合',
    '语言模型根据上下文生成回答'
]

# 回环四象限的坐标
positions = {
    '查询处理': (0.2, 0.75),
    '文档检索': (0.75, 0.75),
    '结果融合': (0.75, 0.25),
    '文本生成': (0.2, 0.25),
}

# 绘制节点
box_w, box_h = 0.2, 0.12
for node, desc in zip(nodes, descriptions):
    x, y = positions[node]
    ax.add_patch(FancyBboxPatch((x, y),
                                box_w, box_h,
                                boxstyle="round,pad=0.02",
                                edgecolor='black',
                                facecolor='#CCE5FF'))
    ax.text(x + box_w/2, y + box_h*0.6,
            node, ha='center', va='center',
            fontsize=14, weight='bold')
    ax.text(x + box_w/2, y + box_h*0.35,
            desc, ha='center', va='center',
            fontsize=12, wrap=True)

# 缩短箭头的偏移量
pad_x = 0.03 * box_w  # 横向内缩
pad_y = 0.02 * box_h  # 纵向内缩

# 绘制闭环箭头（用 pad 内缩）
arrow_props = dict(arrowstyle='->', lw=2)
# 查询处理 -> 文档检索
start = (positions['查询处理'][0] + box_w - pad_x,
         positions['查询处理'][1] + box_h/2)
end   = (positions['文档检索'][0] + pad_x,
         positions['文档检索'][1] + box_h/2)
ax.annotate('', xy=end, xytext=start, arrowprops=arrow_props)

# 文档检索 -> 结果融合
start = (positions['文档检索'][0] + box_w/2,
         positions['文档检索'][1] - pad_y)
end   = (positions['结果融合'][0] + box_w/2,
         positions['结果融合'][1] + box_h + pad_y)
ax.annotate('', xy=end, xytext=start, arrowprops=arrow_props)

# 结果融合 -> 文本生成
start = (positions['结果融合'][0] - pad_x,
         positions['结果融合'][1] + box_h/2)
end   = (positions['文本生成'][0] + box_w - pad_x,
         positions['文本生成'][1] + box_h/2)
ax.annotate('', xy=end, xytext=start, arrowprops=arrow_props)

# 标题
plt.text(0.55, 0.95, '图 2-1  RAG 的核心流程图', fontsize=16, ha='center')

plt.tight_layout()
plt.show()