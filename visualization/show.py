import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt

# 文件路径配置
BASE_DIR = r"output\20250103-004457\artifacts"
NODES_FILE = f"{BASE_DIR}\create_final_nodes.parquet"
RELATIONSHIPS_FILE = f"{BASE_DIR}\create_final_relationships.parquet"


def load_data():
    """
    加载节点和关系数据，并进行必要的列重命名或检查
    """
    try:
        # 读取数据
        nodes = pd.read_parquet(NODES_FILE)
        relationships = pd.read_parquet(RELATIONSHIPS_FILE)
        print("Data loaded successfully!")

        # 调试：打印列名
        print("Nodes columns:", nodes.columns.tolist())
        print("Relationships columns:", relationships.columns.tolist())

        # 如果 relationships 中没有 'source_id'，但有可能是 'start' 或 'source'
        if 'source_id' not in relationships.columns:
            if 'start' in relationships.columns:
                relationships.rename(columns={'start': 'source_id'}, inplace=True)
            elif 'source' in relationships.columns:
                relationships.rename(columns={'source': 'source_id'}, inplace=True)

        # 如果 relationships 中没有 'target_id'，但有可能是 'end' 或 'target'
        if 'target_id' not in relationships.columns:
            if 'end' in relationships.columns:
                relationships.rename(columns={'end': 'target_id'}, inplace=True)
            elif 'target' in relationships.columns:
                relationships.rename(columns={'target': 'target_id'}, inplace=True)

        # 再次检查 source_id/target_id
        missing_cols = []
        if 'source_id' not in relationships.columns:
            missing_cols.append('source_id')
        if 'target_id' not in relationships.columns:
            missing_cols.append('target_id')
        if missing_cols:
            raise ValueError(
                f"Relationships DataFrame 缺少必要的列：{missing_cols}，"
                f"请检查实际文件中的列名：{relationships.columns.tolist()}"
            )

        return nodes, relationships

    except Exception as e:
        print(f"Error loading data: {e}")
        return None, None


def build_graph(nodes, relationships):
    """
    构建一个有向图 DiGraph
    """
    G = nx.DiGraph()  # 有向图

    # 添加节点
    for _, row in nodes.iterrows():
        node_id = row['id']  # 必须保证 DataFrame 中有 'id'
        node_label = row.get('human_readable_id', 'Unknown')  # 使用 'type' 作为节点的 label
        node_title = row.get('description', 'Unnamed')  # 使用 'title' 作为节点显示名称
        G.add_node(node_id, label=node_label, title=node_title)

    # 添加关系
    for _, row in relationships.iterrows():
        source = row['source_id']
        target = row['target_id']
        rel_type = row.get('weight', 'description')
        # 可选参数列表： Relationships columns: ['source', 'target', 'weight', 'description', 'text_unit_ids', 'id', 'human_readable_id', 'source_degree', 'target_degree', 'rank']
        G.add_edge(source, target, label=rel_type)

    return G


def visualize_graph(G, filter_node_type=None, filter_edge_type=None):
    """
    使用 NetworkX + matplotlib 对图进行可视化
    """
    plt.figure(figsize=(30, 20))  # 设置更大的图形尺寸
    pos = nx.spring_layout(G, k=2, iterations=100)  # 调整布局参数

    # 根据需求筛选节点和边
    if filter_node_type:
        nodes_to_draw = [n for n, d in G.nodes(data=True) if d.get('label') == filter_node_type]
    else:
        nodes_to_draw = G.nodes()

    if filter_edge_type:
        edges_to_draw = [(u, v) for u, v, d in G.edges(data=True) if d.get('label') == filter_edge_type]
    else:
        edges_to_draw = G.edges()

    # 绘制节点
    nx.draw_networkx_nodes(G, pos, nodelist=nodes_to_draw, node_size=800, node_color='lightblue')

    # 绘制边
    nx.draw_networkx_edges(G, pos, edgelist=edges_to_draw, edge_color='gray')

    # 添加节点和边标签
    node_labels = {n: d.get('title', 'Unnamed') for n, d in G.nodes(data=True) if n in nodes_to_draw}
    edge_labels = {(u, v): d['label'] for u, v, d in G.edges(data=True) if (u, v) in edges_to_draw}

    nx.draw_networkx_labels(G, pos, labels=node_labels, font_size=10, font_color='black')
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=8)

    plt.title("GraphRAG Relationship Visualization")
    plt.axis("off")

    # 保存为 SVG 矢量图（放在 plt.show 之前）
    plt.savefig("graph_visualization.svg", format="svg", dpi=300)

    # 显示图形
    plt.show()

if __name__ == "__main__":
    # 加载数据
    nodes, relationships = load_data()
    if nodes is not None and relationships is not None:
        # 构建图
        graph = build_graph(nodes, relationships)

        # 可视化图（不加过滤）
        visualize_graph(graph)

        # 如果需要只可视化某一种节点：
        # visualize_graph(graph, filter_node_type='Person')

        # 如果需要只可视化某一种关系：
        # visualize_graph(graph, filter_edge_type='colleague')
