import networkx as nx
import pandas as pd
import random
import numpy as np
from collections import defaultdict
import random_walk


def generate_random_walk(G, start_node, walk_length=10, method='deepwalk'):
    """
    生成随机游走链，支持三种算法：DeepWalk、Node2Vec、改进的Node2Vec
    """
    if method == 'deepwalk':
        walks, _ = random_walk.deepwalk(G, num_walks=1, walk_length=walk_length)
    elif method == 'node2vec':
        walks, _ = random_walk.node2vec_random_walks(G, num_walks=1, walk_length=walk_length)
    elif method == 'node2vec_new':
        walks, _ = random_walk.node2vec_new_random_walks(G, num_walks=1, walk_length=walk_length)
    else:
        raise ValueError("Invalid method, choose from ['deepwalk', 'node2vec', 'node2vec_new']")

    return walks


def jaccard_similarity(walks, node1, node2):
    """
    计算Jaccard相似性，基于随机游走
    """
    walks_node1 = set([i for walk in walks if node1 in walk for i in walk])
    walks_node2 = set([i for walk in walks if node2 in walk for i in walk])
    intersection = len(walks_node1 & walks_node2)
    union = len(walks_node1 | walks_node2)
    return intersection / union if union != 0 else 0


def co_occurrence_similarity(walks, node1, node2):
    """
    计算节点1和节点2在随机游走中共同出现的次数
    """
    co_occurrence = 0
    for walk in walks:
        if node1 in walk and node2 in walk:
            co_occurrence += 1
    return co_occurrence

def cosine_similarity(walks, node1, node2):
    """
    计算Cosine相似性，基于随机游走
    """
    # 获取游走中所有与node1和node2共同出现的节点
    walks_node1 = [walk for walk in walks if node1 in walk]
    walks_node2 = [walk for walk in walks if node2 in walk]

    # 创建对应节点的共现向量
    common_node1 = set([node for walk in walks_node1 for node in walk])
    common_node2 = set([node for walk in walks_node2 for node in walk])

    # 如果两个节点没有共同邻居，返回0相似度
    if len(common_node1) == 0 or len(common_node2) == 0:
        return 0.0

    common_neighbors = common_node1 & common_node2

    # 如果两个节点没有共同的邻居，返回0相似度
    if len(common_neighbors) == 0:
        return 0.0

    return len(common_neighbors) / (np.sqrt(len(common_node1)) * np.sqrt(len(common_node2)))
def calculate_similarity_for_selected_nodes(walks, nodes, method='jaccard', threshold=0.1):
    """
    计算指定节点对之间的相似性并筛选出相似性高于阈值的节点对
    """
    similarity_matrix = defaultdict(dict)
    for i, node1 in enumerate(nodes):
        for node2 in nodes[i + 1:]:
            if method == 'jaccard':
                similarity = jaccard_similarity(walks, node1, node2)
            elif method == 'cosine':
                similarity = cosine_similarity(walks, node1, node2)
            elif method == 'co_occurrence':
                similarity = co_occurrence_similarity(walks, node1, node2)
            if similarity > threshold:  # 只保留高于阈值的节点对
                similarity_matrix[node1][node2] = similarity
                similarity_matrix[node2][node1] = similarity  # 相似性是对称的
    return similarity_matrix


# 主函数
def main(selected_nodes=None):
    # 加载网络数据
    edges = []
    with open('初创企业关联的边文件（2015-2020）.csv', 'r') as file:
        for line in file:
            source, target = line.strip().split(',')
            edges.append((source, target))

    G = nx.Graph()
    G.add_edges_from(edges)

    # 设定随机游走的参数
    num_walks = 10
    walk_length = 5

    # 随机选取几个度较高的节点，如果没有传入指定节点
    if selected_nodes is None:
        degree_nodes = sorted(G.degree, key=lambda x: x[1], reverse=True)[:10]  # 选取度前10的节点
        selected_nodes = [node for node, _ in degree_nodes]

    print(f"选取的节点: {selected_nodes}")

    # 分别使用三种方法生成随机游走链
    all_walks_deepwalk = []
    all_walks_node2vec = []
    all_walks_node2vec_new = []
    for node in selected_nodes:
        all_walks_deepwalk.extend(generate_random_walk(G, node, walk_length, 'deepwalk'))
        all_walks_node2vec.extend(generate_random_walk(G, node, walk_length, 'node2vec'))
        all_walks_node2vec_new.extend(generate_random_walk(G, node, walk_length, 'node2vec_new'))

    # 执行相似性分析
    similarity_results = {}
    for method in ['jaccard', 'cosine', 'co_occurrence']:
        print(f"正在计算{method}相似性...")
        # 对每种游走方法计算相似性
        similarity_matrix_deepwalk = calculate_similarity_for_selected_nodes(all_walks_deepwalk, selected_nodes, method,
                                                                             threshold=0.1)
        similarity_matrix_node2vec = calculate_similarity_for_selected_nodes(all_walks_node2vec, selected_nodes, method,
                                                                             threshold=0.1)
        similarity_matrix_node2vec_new = calculate_similarity_for_selected_nodes(all_walks_node2vec_new, selected_nodes,
                                                                                 method, threshold=0.1)

        similarity_results[f'deepwalk_{method}'] = similarity_matrix_deepwalk
        similarity_results[f'node2vec_{method}'] = similarity_matrix_node2vec
        similarity_results[f'node2vec_new_{method}'] = similarity_matrix_node2vec_new

    # 输出相似性结果
    for method, result in similarity_results.items():
        print(f"\n{method.capitalize()} 相似性结果:")
        for node1, neighbors in result.items():
            for node2, similarity in neighbors.items():
                print(f"节点 {node1} 和 节点 {node2} 的相似性: {similarity:.4f}")

    # 保存相似性结果为Excel文件
    with pd.ExcelWriter('Selected_Nodes_Similarity_Results.xlsx') as writer:
        for method, result in similarity_results.items():
            df_similarity = pd.DataFrame(result)
            df_similarity.to_excel(writer, sheet_name=method)


if __name__ == "__main__":
    # 指定选定的节点，例如：['87368', '90415'] 106103 22510
    # 如果不指定节点，随机选择度较高的节点
    main(selected_nodes=None)
