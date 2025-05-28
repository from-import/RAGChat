import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei']  # 中文字体
plt.rcParams['axes.unicode_minus'] = False

fig, ax = plt.subplots(figsize=(12, 6))

actors = ['用户前端', 'Controller层', 'RabbitMQ', '消费者线程池', 'OpenAI服务', '数据库与Redis', 'WebSocket']
positions = {actor: i for i, actor in enumerate(actors)}

# 更新最后一个步骤的起点为上一步终点（数据库与Redis），保持流程连续
steps = [
    ("用户前端", "Controller层", "用户发送问题"),
    ("Controller层", "RabbitMQ", "消息异步投递"),
    ("RabbitMQ", "消费者线程池", "监听到消息"),
    ("消费者线程池", "OpenAI服务", "关键词抽取与问答请求"),
    ("OpenAI服务", "消费者线程池", "返回回答"),
    ("消费者线程池", "数据库与Redis", "保存回答与缓存更新"),
    ("数据库与Redis", "WebSocket", "推送回答至前端")  # ✅ 修复这里
]

for t, (src, dst, msg) in enumerate(steps):
    x_start = t
    x_end = t + 1
    y_start = positions[src]
    y_end = positions[dst]

    ax.annotate("",
                xy=(x_end, y_end),
                xytext=(x_start, y_start),
                arrowprops=dict(arrowstyle="->", lw=1.5))
    ax.text((x_start + x_end) / 2,
            (y_start + y_end) / 2 + 0.2,
            msg,
            ha='center',
            fontsize=10,
            bbox=dict(boxstyle="round", fc="w", ec="gray", alpha=0.9))

ax.set_yticks(list(positions.values()))
ax.set_yticklabels(actors)
ax.set_xticks([])
ax.set_xlim(0, len(steps))
ax.set_title("RAG 问答系统核心业务流程时序图", fontsize=14)
ax.grid(axis='y', linestyle='--', alpha=0.3)

plt.tight_layout()
plt.show()
