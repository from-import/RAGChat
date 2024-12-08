import pandas as pd

# 指定 Parquet 文件的路径
file_path = "create_final_documents.parquet"

# 使用 Pandas 加载文件
df = pd.read_parquet(file_path)

# 查看文件内容
print(df.head())  # 显示前5行
print(df.info())  # 显示列信息和数据类型

# 导出为 CSV 文件
df.to_csv("final_documents.csv", index=False)

# 导出为 Excel 文件
df.to_excel("final_documents.xlsx", index=False)

