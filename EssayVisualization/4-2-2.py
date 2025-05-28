import matplotlib.pyplot as plt
import numpy as np

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 图1：系统响应耗时构成柱状图
fig1, ax1 = plt.subplots(figsize=(8, 5))

components = ['身份验证\n与预处理', '消息投递', '关键词抽取', '模型推理', '数据库缓存', 'WebSocket 推送']
times_ms = [17, 42, 221, 11400, 8, 3]

bars = ax1.bar(components, times_ms)
ax1.set_title("系统响应耗时构成图（单位：毫秒）")
ax1.set_ylabel("耗时 (ms)")
ax1.grid(True, axis='y', linestyle='--', alpha=0.4)

# 图2：F1/精确率/召回率随训练轮次变化
fig2, ax2 = plt.subplots(figsize=(8, 5))

epochs = list(range(1, 11))
precision = [0, 0, 0, 0, 0.8841, 0.45, 0.49, 0.50, 0.52, 0.5314]
recall =    [0, 0, 0, 0, 0.0057, 0.1, 0.13, 0.15, 0.17, 0.1811]
f1 =        [0, 0, 0, 0, 0.0114, 0.16, 0.19, 0.21, 0.24, 0.2701]

ax2.plot(epochs, precision, marker='o', label='精确率')
ax2.plot(epochs, recall, marker='s', label='召回率')
ax2.plot(epochs, f1, marker='^', label='F1 值')

ax2.set_title("精确率 / 召回率 / F1 随训练轮次变化")
ax2.set_xlabel("训练轮次")
ax2.set_ylabel("指标值")
ax2.legend()
ax2.grid(True, linestyle='--', alpha=0.4)

plt.tight_layout()
plt.show()

