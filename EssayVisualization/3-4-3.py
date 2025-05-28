import matplotlib.pyplot as plt

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

fig, ax = plt.subplots(figsize=(12, 6))
ax.axis('off')

# 定义流程节点
nodes = {
    "invoke": (0.05, 0.5, "saveChatMessage"),
     "lock": (0.3, 0.5, "tryLock(wait=5s)"),
    "no_lock": (0.6, 0.7, "未获取锁\n写库，不清理缓存"),
    "locked": (0.55, 0.3, "获取锁成功\n写库"),
    "del_hist": (0.8, 0.3, "删除历史对话缓存"),
    "del_resp": (0.8, 0.1, "删除响应结果缓存"),
    "unlock": (0.95, 0.3, "unlock"),
    "return": (0.95, 0.5, "返回 Mono<Void>")
}

# 绘制节点
for x, y, text in nodes.values():
    ax.text(
        x, y, text,
        ha="center", va="center",
        fontsize=16,
        bbox=dict(boxstyle="round,pad=0.5", edgecolor="black")
    )

# 绘制箭头
arrows = [
    ("invoke", "lock"),
    ("lock", "no_lock"),
    ("lock", "locked"),
    ("no_lock", "return"),
    ("locked", "del_hist"),
    ("del_hist", "del_resp"),
    ("del_resp", "unlock"),
    ("unlock", "return"),
]

for src, dst in arrows:
    x1, y1, _ = nodes[src]
    x2, y2, _ = nodes[dst]
    dx, dy = x2 - x1, y2 - y1
    value = 0.4
    sx = x1 + dx * value
    sy = y1 + dy * value
    ex = x1 + dx * (1-value)
    ey = y1 + dy * (1-value)
    ax.annotate(
        "", xy=(ex, ey), xytext=(sx, sy),
        arrowprops=dict(arrowstyle="->", lw=1.5, connectionstyle="arc3")
    )

plt.tight_layout()
plt.show()
