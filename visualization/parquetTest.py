import pandas as pd

nodes_file = "C:/Users/Red/Desktop/GraphRag/autogen_graphRAG/ragtest/output/20250103-004457/artifacts/create_final_nodes.parquet"
nodes = pd.read_parquet(nodes_file)
print(nodes.columns)
print(nodes.head())