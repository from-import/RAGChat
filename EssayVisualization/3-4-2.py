import matplotlib.pyplot as plt

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

fig, ax = plt.subplots(figsize=(12, 6))
ax.axis('off')

# 定义流程节点（x, y, 文本）
nodes = {
    "req": (0.05, 0.5, "业务层\n请求对话列表"),
    "redis": (0.35, 0.5, "Redis 查询"),
    "hit": (0.65, 0.72, "命中缓存\nJSON→List\n≈6 ms"),
    "miss": (0.65, 0.28, "未命中缓存\n查询MySQL \n并写回Redis\n≈14 ms"),
    "return": (0.92, 0.5, "返回\n会话列表")
}

# 绘制节点，浅蓝色背景，字体大小18
for x, y, text in nodes.values():
    ax.text(
        x,
        y,
        text,
        ha="center",
        va="center",
        fontsize=18,
        bbox=dict(boxstyle="round,pad=0.5", facecolor="lightblue", edgecolor="black")
    )

# 绘制箭头，从节点中心沿向量方向缩短10%后开始，并在目标中心前10%位置结束
arrows = [
    ("req", "redis"),
    ("redis", "hit"),
    ("redis", "miss"),
    ("hit", "return"),
    ("miss", "return"),
]

for src, dst in arrows:
    x1, y1, _ = nodes[src]
    x2, y2, _ = nodes[dst]
    dx = x2 - x1
    dy = y2 - y1
    # 缩短10%：起点移动10%，终点移动10%
    value = 0.3
    sx = x1 + dx * value
    sy = y1 + dy * value
    ex = x1 + dx * (1-value)
    ey = y1 + dy * (1-value)
    ax.annotate(
        "",
        xy=(ex, ey),
        xytext=(sx, sy),
        arrowprops=dict(arrowstyle="->", lw=2, connectionstyle="arc3")
    )


plt.tight_layout()
plt.show()
