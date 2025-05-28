import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyArrowPatch

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

fig, ax = plt.subplots(figsize=(12, 8))
ax.set_xlim(0, 10)
ax.set_ylim(0, 10)
ax.axis('off')  # 不显示坐标轴

# 定义模块位置
modules = {
    "前端页面": (1, 5),
    "后端服务（Spring Boot）": (4, 5),
    "Redis 缓存": (7.2, 7.5),
    "MySQL 数据库": (7.2, 5),
    "RabbitMQ 队列": (7.2, 2.5),
    "WebSocket 通道": (4, 2),
    "Ollama 模型服务": (9, 0.5)
}

# 绘制模块
for name, (x, y) in modules.items():
    ax.add_patch(Rectangle((x-0.8, y-0.5), 2, 1, edgecolor='black', facecolor='#d0e9f2'))
    ax.text(x + 0.2, y, name, fontsize=10, verticalalignment='center')

# 绘制箭头函数
def draw_arrow(x1, y1, x2, y2):
    arrow = FancyArrowPatch((x1, y1), (x2, y2),
                            arrowstyle='->',
                            mutation_scale=15,
                            color='gray')
    ax.add_patch(arrow)

# 数据流连接关系
draw_arrow(2.2, 5, 3.2, 5)  # 前端 -> 后端
draw_arrow(5.2, 5, 6.4, 7.5)  # 后端 -> Redis
draw_arrow(5.2, 5, 6.4, 5)  # 后端 -> MySQL
draw_arrow(5.2, 5, 6.4, 2.5)  # 后端 -> RabbitMQ
draw_arrow(6.6, 2.5, 8.8, 0.5)  # RabbitMQ -> 模型服务
draw_arrow(4, 5, 4, 3)  # 后端 -> WebSocket
draw_arrow(4, 2.5, 2.2, 5)  # WebSocket -> 前端（响应）

plt.title("系统整体架构图", fontsize=14)
plt.tight_layout()
plt.show()
