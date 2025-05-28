import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyArrowPatch

# 设置中文字体为 SimHei，确保能显示中文
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 创建画布和坐标轴
fig, ax = plt.subplots(figsize=(10, 2))
ax.axis('off')  # 不显示坐标轴

# 流程节点与位置
steps = ['原始文本', '实体识别', '关系抽取', '本体映射', '图谱生成']
for i, label in enumerate(steps):
    x = i * 2  # 每个节点横向间隔 2 单位
    # 绘制矩形框
    rect = Rectangle((x, 0.4), 1.8, 0.6,
                     linewidth=1, edgecolor='black', facecolor='lightgray')
    ax.add_patch(rect)
    # 添加文字
    ax.text(x + 0.9, 0.7, label, ha='center', va='center', fontsize=14)

    # 绘制箭头（除了最后一个节点之外）
    if i < len(steps) - 1:
        arrow = FancyArrowPatch((x + 1.8, 0.7), (x + 2.0, 0.7),
                                arrowstyle='->', mutation_scale=20)
        ax.add_patch(arrow)

# 设置显示范围
ax.set_xlim(-0.2, 2 * len(steps))
ax.set_ylim(0, 1.5)
plt.tight_layout()

# 显示
plt.show()
