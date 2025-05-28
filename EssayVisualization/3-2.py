import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Polygon

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

fig, ax = plt.subplots(figsize=(12, 6))
ax.axis('off')

# 工具函数：画框
def draw_box(x, y, width, height, text, color="#D9D9D9"):
    box = FancyBboxPatch((x, y), width, height,
                         boxstyle="round,pad=0.3",
                         edgecolor="black",
                         facecolor=color,
                         linewidth=1.5)
    ax.add_patch(box)
    ax.text(x + width / 2, y + height / 2, text, ha='center', va='center', fontsize=12)

# 工具函数：画菱形判断框
def draw_diamond(cx, cy, width, height, text):
    points = [
        (cx, cy + height / 2),
        (cx + width / 2, cy),
        (cx, cy - height / 2),
        (cx - width / 2, cy)
    ]
    diamond = Polygon(points, closed=True, edgecolor="black", facecolor="white", linewidth=1.5)
    ax.add_patch(diamond)
    ax.text(cx, cy, text, ha='center', va='center', fontsize=12)

# 步骤框及判断框坐标与大小
box_w, box_h = 1.8, 0.8
y_top = 4.5

# Redis 缓存查询框（黄色）
draw_box(1, y_top, box_w, box_h, "查询 Redis 缓存", "#FFE699")

# 判断框（是否命中缓存）
draw_diamond(4.5, y_top + box_h / 2, 1.5, 1.0, "是否命中？")

# MySQL 查询框（蓝色）
draw_box(7, y_top, box_w, box_h, "查询 MySQL 数据库", "#9FC5E8")

# 回填缓存（黄色）
draw_box(1, 2.5, box_w, box_h, "回填 Redis 缓存", "#FFE699")

# 返回结果（灰色）
draw_box(7, 2.5, box_w, box_h, "返回结果", "#D9D9D9")

# 箭头样式
arrowprops = dict(arrowstyle="->", linewidth=1.5, color="black")

# 连线：Redis → 判断框
ax.annotate("", xy=(2.8, y_top + box_h / 2), xytext=(2.0, y_top + box_h / 2), arrowprops=arrowprops)

# 判断框 → 返回结果（命中）
ax.annotate("", xy=(4.5, y_top), xytext=(4.5, 3.3), arrowprops=arrowprops)
ax.text(4.6, 3.8, "是", fontsize=12)

# 判断框 → MySQL（未命中）
ax.annotate("", xy=(5.25, y_top + 0.4), xytext=(6.8, y_top + 0.4), arrowprops=arrowprops)
ax.text(5.7, y_top + 0.5, "否", fontsize=12)

# MySQL → 回填缓存
ax.annotate("", xy=(7.9, y_top), xytext=(2.0, 3.3), arrowprops=arrowprops)

# 回填缓存 → 返回结果
ax.annotate("", xy=(2.8, 2.9), xytext=(7.0, 2.9), arrowprops=arrowprops)

plt.tight_layout()
plt.show()
