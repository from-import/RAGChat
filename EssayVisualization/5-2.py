import matplotlib.pyplot as plt
import numpy as np

# 设置中文字体
plt.rcParams['font.family'] = 'SimHei'

# 数据
labels = ['GraphRAG + Ollama', 'AnythingLLM']
times = [57210, 11400]  # 单位：ms

# 设置图形大小
fig, ax = plt.subplots(figsize=(8, 6))

# 设置x轴位置，使得柱子间隔更宽
x = np.arange(len(labels)) * 1.5  # 拉大间隔
bars = ax.bar(x, times, color=['skyblue', 'orange'], width=0.3)

# 添加数据标签
for bar, time in zip(bars, times):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1000,
            f'{time} ms', ha='center', va='bottom', fontsize=10)

# 设置标题和标签
ax.set_title('模型生成阶段响应时长对比', fontsize=14)
ax.set_ylabel('耗时（毫秒）', fontsize=12)
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.grid(axis='y', linestyle='--', alpha=0.6)

plt.tight_layout()
plt.show()
