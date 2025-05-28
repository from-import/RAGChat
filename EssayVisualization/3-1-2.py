import matplotlib.pyplot as plt

# 设置中文字体为 SimHei，确保中文字符正确显示
plt.rcParams['font.sans-serif'] = ['SimHei']
# 关闭坐标轴上的负号显示问题
plt.rcParams['axes.unicode_minus'] = False

# 创建图表和坐标轴，figsize=(宽度, 高度) 单位为英寸
fig, ax = plt.subplots(figsize=(14, 8))

# 调整子图左右边距，left/right 值为相对于画布宽度的比例
plt.subplots_adjust(left=0.15, right=0.95)

# 定义流程图中的节点顺序，从上到下排列更符合阅读逻辑
actors = [
    '用户前端',      # 0
    'Controller层',  # 1
    'RabbitMQ',      # 2
    '消费者线程池',  # 3
    'OpenAI服务',    # 4
    '数据库与Redis', # 5
    'WebSocket'      # 6
]
# 将 actors 列表反转并枚举，用于计算每个节点在纵轴上的位置索引
positions = {actor: i for i, actor in enumerate(actors[::-1])}

# 定义流程中的每一步：来源节点、目标节点、箭头标签
steps = [
    ("用户前端", "Controller层", "用户发送问题"),
    ("Controller层", "RabbitMQ", "消息异步投递"),
    ("RabbitMQ", "消费者线程池", "监听到消息"),
    ("消费者线程池", "OpenAI服务", "关键词抽取"),
    ("OpenAI服务", "消费者线程池", "返回回答"),
    ("消费者线程池", "数据库与Redis", "保存回答内容"),
    ("数据库与Redis", "WebSocket", "推送回答至前端")
]

# 绘制每条泳道的横向虚线，y 为节点的位置，linewidth 为线宽
for y in positions.values():
    ax.axhline(y=y, color='gray', linestyle='--', linewidth=0.5)

# 遍历每个步骤，绘制箭头和标签
for t, (src, dst, msg) in enumerate(steps):
    # x_start 和 x_end 定义箭头的起止横坐标
    x_start = t
    x_end = t + 1
    # y_start 和 y_end 定义箭头的起止纵坐标
    y_start = positions[src]
    y_end = positions[dst]

    # 计算文字放置位置的偏移量
    dx = x_end - x_start
    dy = y_end - y_start
    text_x = x_start + 0.5  # 文本在箭头起点之后 0.4 单位
    # 如果目标在下方，就把标签放在起点下方，否则放在起点上方
    text_y = y_start - 0.2 if dy > 0 else y_start + 0.2

    ax.annotate(
        msg,                             # 要显示的文字
        xy=(x_end, y_end),               # 箭头指向的坐标
        xytext=(text_x, text_y),         # 文字所在坐标
        fontsize=20,                     # 字体大小
        ha='center',                     # 水平对齐方式：居中
        va='center',                     # 垂直对齐方式：居中
        bbox=dict(                       # 文本框样式
            boxstyle="round,pad=0.4",    # 圆角+内边距
            fc="white",                  # 填充色
            ec="black",                  # 边框颜色
            lw=1                       # 边框宽度
        ),
        arrowprops=dict(                 # 箭头样式
            arrowstyle="->",             # 箭头形状
            lw=1.5,                      # 箭头线宽
            color='black'                # 箭头颜色
        )
    )

# 设置 y 轴刻度位置为各节点的纵坐标
ax.set_yticks(list(positions.values()))
# 设置 y 轴刻度标签为节点名称，fontsize 控制标签字体大小
ax.set_yticklabels(list(positions.keys()), fontsize=18)

# 不显示 x 轴刻度
ax.set_xticks([])

# x 轴显示范围从 0 到 步骤数
ax.set_xlim(0, len(steps))
# y 轴显示范围从 -1 到 节点总数，以留出上下边距
ax.set_ylim(-1, len(actors))

# 设置图表标题及其字体大小
ax.set_title("RAG 问答系统核心业务流程时序图", fontsize=25)
# 关闭默认网格线
ax.grid(False)

# 自动调整子图参数，使内容填充画布
plt.tight_layout()
# 渲染并显示图表
plt.show()
