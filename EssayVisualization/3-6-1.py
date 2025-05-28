import matplotlib.pyplot as plt

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

fig, ax = plt.subplots(figsize=(14, 8))
ax.axis('off')

# 定义流程节点：位置和文本
nodes = {
    "frontend": (0.05, 0.5, "前端用户操作"),
    "controller": (0.25, 0.5, "ChatController\n/chat 接口"),
    "convService": (0.45, 0.7, "ConversationService\n获取或创建会话"),
    "msgService": (0.45, 0.3, "ChatMessageService\n写库并删除缓存"),
    "producer": (0.65, 0.5, "RabbitTemplate\nconvertAndSend"),
    "exchange": (0.85, 0.65, "TopicExchange\nchatExchange"),
    "queue": (0.85, 0.5, "队列\nchatQueue"),
    "consumer": (0.85, 0.35, "ChatMessageConsumer\n@RabbitListener"),
    "threadpool": (0.65, 0.15, "线程池任务"),
    "ws": (0.9, 0.15, "WebSocket推送")
}

# 绘制节点
for x, y, text in nodes.values():
    ax.text(
        x, y, text,
        ha="center", va="center",
        fontsize=16,
        bbox=dict(boxstyle="round,pad=0.5", facecolor="lightblue", edgecolor="black")
    )

# 定义箭头连接
arrows = [
    ("frontend", "controller"),
    ("controller", "convService"),
    ("controller", "msgService"),
    ("convService", "producer"),
    ("msgService", "producer"),
    ("producer", "exchange"),
    ("exchange", "queue"),
    ("queue", "consumer"),
    ("consumer", "threadpool"),
    ("threadpool", "ws")
]

# 绘制缩短箭头：起点尾部各缩短30%，头部尾部各缩短30%
for src, dst in arrows:
    x1, y1, _ = nodes[src]
    x2, y2, _ = nodes[dst]
    dx, dy = x2 - x1, y2 - y1
    start_frac, end_frac = 0.3, 0.7  # 缩短比例
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
