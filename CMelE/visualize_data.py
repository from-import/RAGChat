import matplotlib.pyplot as plt
import pandas as pd

# 设置中文字体为 SimHei（黑体），避免乱码
plt.rcParams['font.family'] = 'SimHei'
plt.rcParams['axes.unicode_minus'] = False

# 手动整理的训练数据（基于用户提供的训练日志）
data = {
    "Epoch": list(range(1, 11)),
    "Avg Loss": [0.0896, 0.0229, 0.0190, 0.0155, 0.0125, 0.0105, 0.0088, 0.0076, 0.0068, 0.0061],
    "F1": [0.0000, 0.0000, 0.0000, 0.0000, 0.0114, 0.1590, 0.1972, 0.2225, 0.2620, 0.2701],
    "Precision": [1.0000, 1.0000, 1.0000, 1.0000, 0.8841, 0.5980, 0.5270, 0.5362, 0.4688, 0.5314],
    "Recall": [0.0000, 0.0000, 0.0000, 0.0000, 0.0057, 0.0917, 0.1213, 0.1404, 0.1818, 0.1811]
}
df = pd.DataFrame(data)

# 绘制 F1 值变化曲线
plt.figure(figsize=(8, 5))
plt.plot(df["Epoch"], df["F1"], marker='o', color='blue', label="F1 分数")
plt.title("模型训练过程中 F1 值变化曲线")
plt.xlabel("训练轮次 Epoch")
plt.ylabel("F1 分数")
plt.xticks(df["Epoch"])
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

# 绘制指标表格
df = pd.DataFrame(data)

# 正确设置表格绘制区域
fig, ax = plt.subplots(figsize=(10, 4))
ax.axis('off')

# 创建表格
table = ax.table(
    cellText=df.values,
    colLabels=df.columns,
    cellLoc='center',
    loc='center'
)
table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1.2, 1.5)

plt.title("训练各轮次评估指标", pad=12)
plt.tight_layout()
plt.show()
