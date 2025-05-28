import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

fig, ax = plt.subplots(figsize=(10, 8))
ax.axis('off')

# 前端模块
ax.add_patch(Rectangle((0.05, 0.55), 0.2, 0.2, edgecolor='black', facecolor='#CCE5FF'))
ax.text(0.15, 0.65, '前端\n(Browser)', ha='center', va='center', fontsize=12)

# 后端模块框
ax.add_patch(Rectangle((0.3, 0.1), 0.4, 0.7, edgecolor='black', facecolor='#E6F2FF'))
ax.text(0.5, 0.76, '后端 (Spring Boot)', ha='center', va='center', fontsize=14)

# 后端子模块
submodules = [
    ('用户认证与会话管理', 0.65),
    ('对话记录管理', 0.55),
    ('消息处理与生成', 0.45),
    ('缓存优化模块', 0.35),
    ('异步消息队列模块', 0.25)
]
for name, y in submodules:
    ax.add_patch(Rectangle((0.33, y), 0.34, 0.08, edgecolor='gray', facecolor='white'))
    ax.text(0.5, y + 0.04, name, ha='center', va='center', fontsize=11)

# 外部服务模块
externals = [
    ('WebSocket', 0.65),
    ('RabbitMQ', 0.55),
    ('Redis', 0.45),
    ('MySQL', 0.35)
]
for idx, (name, y) in enumerate(externals):
    ax.add_patch(Rectangle((0.75, y), 0.18, 0.08, edgecolor='black', facecolor='#FFE5CC'))
    ax.text(0.75 + 0.09, y + 0.04, name, ha='center', va='center', fontsize=11)
    ax.annotate('', xy=(0.75, y + 0.04), xytext=(0.67, y + 0.04),
                arrowprops=dict(arrowstyle='->', lw=1.8))

# 前端 -> 后端
ax.annotate('', xy=(0.3, 0.65), xytext=(0.25, 0.65),
            arrowprops=dict(arrowstyle='->', lw=1.8))

# 标题
plt.text(0.5, 0.85, '图 3-1 系统整体架构概述', fontsize=16, ha='center')
plt.tight_layout()
plt.show()
