import matplotlib.pyplot as plt

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

fig, ax = plt.subplots(figsize=(10, 4))
ax.axis('off')

# 定义流程节点：位置、文本
nodes = {
    "input": (0.1, 0.5, "用户提交明文密码"),
    "bcrypt": (0.5, 0.5, "BCrypt 单向哈希\n盐值嵌入密文"),
    "store": (0.9, 0.5, "存储密文密码\n（数据库）")
}

# 绘制节点，浅绿色背景，字体大小16
for x, y, text in nodes.values():
    ax.text(
        x, y, text,
        ha="center", va="center",
        fontsize=16,
        bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgreen", edgecolor="darkgreen")
    )

# 箭头列表
arrows = [("input", "bcrypt"), ("bcrypt", "store")]

for src, dst in arrows:
    x1, y1, _ = nodes[src]
    x2, y2, _ = nodes[dst]
    dx, dy = x2 - x1, y2 - y1
    start_frac, end_frac = 0.3, 0.7
    sx, sy = x1 + dx * start_frac, y1 + dy * start_frac
    ex, ey = x1 + dx * end_frac, y1 + dy * end_frac
    ax.annotate(
        "",
        xy=(ex, ey),
        xytext=(sx, sy),
        arrowprops=dict(arrowstyle="->", lw=1.5, connectionstyle="arc3")
    )

plt.tight_layout()
plt.show()
