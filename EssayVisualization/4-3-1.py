import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyBboxPatch

# 设置中文字体为 SimHei
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 创建画布
fig, ax = plt.subplots(figsize=(12, 4))
ax.axis('off')

# 1. Input 虚线框
input_box = FancyBboxPatch((0, 0.8), 1.2, 2.4,
                           boxstyle="round,pad=0.3",
                           linestyle='--', linewidth=1)
ax.add_patch(input_box)
ax.text(0.6, 3.6, 'Input', ha='center', va='bottom', fontsize=16)

# 在 Input 内部绘制实体框（删除 '…'，三个实体框纵向均匀分布）
items = ['文本语料库', '图像', '音频']
for i, txt in enumerate(items):
    # 对三个框，中心 y 坐标分别设置为 3.0、2.0、1.0，刚好填满 0.8–3.2 区间
    y_center = 3.0 - i * 1.0
    ent_box = FancyBboxPatch(
        (0.1, y_center - 0.15),  # 框左下角 = (x, y_center - 高度/2)
        1.0,                      # 宽度
        0.3,                      # 高度
        boxstyle="round,pad=0.2",
        edgecolor='black',
        facecolor='white'
    )
    ax.add_patch(ent_box)
    ax.text(
        0.6,                      # 文本水平居中
        y_center,                 # 文本垂直居中
        txt,
        ha='center',
        va='center',
        fontsize=15
    )

# 箭头 Input -> RAG
ax.annotate('', xy=(2, 2.0), xytext=(1.2, 2.0),
            arrowprops=dict(arrowstyle="->", lw=1.2, color='black'))

# 2. RAG 正方形（蓝色）
rag_box = Rectangle((2, 2.5), 1, 1, edgecolor='black', facecolor='#AED6F1')
ax.add_patch(rag_box)
ax.text(2.5, 3, 'RAG', ha='center', va='center', fontsize=18)

# 3. KGs 正方形（浅黄色）
kg_box = Rectangle((2, 0.5), 1, 1, edgecolor='black', facecolor='#F9E79F')
ax.add_patch(kg_box)
ax.text(2.5, 1, 'KGs', ha='center', va='center', fontsize=18)

# 用一条 annotate 画出从 KGs 中心 (2.5, 1.0) 到 RAG 中心 (2.5, 3.0) 的双向箭头
ax.annotate(
    '',
    xy=(2.5, 2.5),     # RAG 正方形中心
    xytext=(2.5, 1.5), # KGs 正方形中心
    arrowprops=dict(
        arrowstyle='<->',   # 双向箭头样式
        lw=1.2,             # 线宽
        color='black',      # 颜色
        shrinkA=0,          # 不在起点收缩
        shrinkB=0,          # 不在终点收缩
        mutation_scale=20   # 箭头头部大小比例，可微调
    )
)
# 箭头 RAG -> Question+Info
ax.annotate('', xy=(4, 2.0), xytext=(3, 2.0),
            arrowprops=dict(arrowstyle="->", lw=1.2, color='black'))

# 4. Question + Relevant information 虚线框 → 白底 + 黑边（保留虚线）
qr_box = FancyBboxPatch(
    (4, 1.2),       # 位置
    2, 1.6,         # 宽、高
    boxstyle="round,pad=0.3",
    linestyle='-', # 虚线边框
    linewidth=1,
    edgecolor='black',
    facecolor='white'  # 填充白色
)
ax.add_patch(qr_box)
ax.text(5, 2.4, '问题', ha='center', va='bottom', fontsize=18)
ax.text(5, 1.6, '相关信息', ha='center', va='center', fontsize=18)
ax.text(5, 2.0, '+', ha='center', va='center', fontsize=20)

# 箭头 Question+Info -> LLM
ax.annotate('', xy=(7, 2.0), xytext=(6, 2.0),
            arrowprops=dict(arrowstyle="->", lw=1.2, color='black'))

# 5. LLM 正方形（浅绿色底，无黄色模块）
llm_box = Rectangle(
    (7, 1.5),        # 矩形左下角坐标
    1.5,             # 宽度
    1,               # 高度
    edgecolor='black',
    facecolor='#ABEBC6'  # 浅绿色底色
)
ax.add_patch(llm_box)

# 模型标签
ax.text(
    7.75, 2,
    '大语言\n模型',
    ha='center',
    va='center',
    fontsize=15
)

# 箭头 LLM -> Output
ax.annotate('', xy=(9.5, 2.0), xytext=(8.5, 2.0),
            arrowprops=dict(arrowstyle="->", lw=1.2, color='black'))

# 6. Output 框
output_box = Rectangle((9.5, 1.5), 1, 1, edgecolor='black', facecolor='white')
ax.add_patch(output_box)
ax.text(10, 2, 'Output', ha='center', va='center', fontsize=12)

# 设置视图范围
ax.set_xlim(-0.5, 11)
ax.set_ylim(0, 4)

plt.tight_layout()
plt.show()
