# 分别计算 BFS 和 DFS 遍历 共同被投资网络 的平均时间，并绘制柱状图
import pandas as pd
import time
import networkx as nx
import random
import matplotlib.pyplot as plt

# 读取 CSV 文件并使用 networkx 构建图
def read_graph(file_path):
    # 读取 CSV 文件
    df = pd.read_csv(file_path)

    # 创建一个有向图
    graph = nx.DiGraph()
    for _, row in df.iterrows():
        source, target = row['Source'], row['Target']
        graph.add_edge(source, target)

    return graph

# BFS 遍历
def bfs(graph, start_node):
    visited = set()
    queue = [start_node]
    traversal_order = []

    while queue:
        node = queue.pop(0)
        if node not in visited:
            visited.add(node)
            traversal_order.append(node)
            queue.extend(graph.successors(node))  # 遍历出邻居

    return traversal_order

# DFS 遍历
def dfs(graph, start_node):
    visited = set()
    stack = [start_node]
    traversal_order = []

    while stack:
        node = stack.pop()
        if node not in visited:
            visited.add(node)
            traversal_order.append(node)
            stack.extend(reversed(list(graph.successors(node))))  # 遍历出邻居

    return traversal_order

# 遍历时间
def measure_time(graph, start_node, algorithm, repetitions=10):
    total_time = 0
    for _ in range(repetitions):
        start_time = time.time()
        algorithm(graph, start_node)
        total_time += time.time() - start_time
    return total_time / repetitions


if __name__ == "__main__":
    # 文件路径
    # file_path = "/bfs-dfs/初创企业关联的边文件（2015-2020）.csv"  # 替换为实际文件路径
    #
    # # 构建图
    # graph = read_graph(file_path)

    # 生成一个随机图
    graph = nx.gnm_random_graph(50000, 100000, directed=True)

    # 获取起始节点
    random_start_node = random.choice(list(graph.nodes()))
    max_outdegree_node = max(graph.out_degree, key=lambda x: x[1])[0]

    # 定义节点来源
    start_nodes = [random_start_node, max_outdegree_node]
    labels = ['Random Start', 'Max Outdegree']

    # 记录结果
    bfs_times = []
    dfs_times = []

    for start_node in start_nodes:
        bfs_time = measure_time(graph, start_node, bfs)
        dfs_time = measure_time(graph, start_node, dfs)


        bfs_times.append(bfs_time)
        dfs_times.append(dfs_time)

    # 绘制分组柱状图
    x = range(len(labels))
    width = 0.2

    plt.bar(x, bfs_times, width, label='BFS')
    plt.bar([i + width for i in x], dfs_times, width, label='DFS')

    plt.xlabel('Start Node Type')
    plt.ylabel('Average Time (seconds)')
    plt.title('BFS vs DFS Traversal Time')
    plt.xticks([i + width / 2 for i in x], labels)
    plt.legend()

    plt.tight_layout()
    plt.show()



