# 使用dfs检测共同被投资网络中的环
# 只检测出度排名前100个节点的环，输出环的长度、起始节点的名称、出度和完整环信息
# 输出环的长度、起始节点的名称、出度和完整环信息
import pandas as pd
import networkx as nx


# # 读取网络边表的 CSV 文件
# file_path = "/bfs-dfs/初创企业关联的边文件（2015-2020）.csv"
# df = pd.read_csv(file_path)
#
# # 读取节点名称和编号对应表
# mapping_file = "/bfs-dfs/初创公司索引文件.xlsx"  # 替换为实际路径
# mapping_df = pd.read_excel(mapping_file)
#
# # 创建一个字典用于编号到名称的映射
# id_to_name = dict(zip(mapping_df['Organization Name No'], mapping_df['Organization Name(去重)']))
#
# # 创建有向图
# graph = nx.DiGraph()
#
# # 根据网络边表中的 Source 和 Target 列添加边
# for _, row in df.iterrows():
#     source, target = row['Source'], row['Target']
#     graph.add_edge(source, target)


# 生成一个随机网络
graph = nx.gnm_random_graph(50000, 100000, directed=True)

# 按出度大小降序排序节点
nodes_sorted_by_out_degree = sorted(graph.out_degree, key=lambda x: x[1], reverse=True)

# 取出出度最大的前 100 个节点
top_100_nodes = [node for node, out_degree in nodes_sorted_by_out_degree[:100]]

# 初始化存储结果的 DataFrame
results = []

def dfs_find_cycle(G, source=None):
    def tailhead(edge):
        return edge[:2]

    explored = set()
    cycle = []
    final_node = None
    for start_node in G.nbunch_iter(source):
        if start_node in explored:
            continue

        edges = []
        seen = {start_node}
        active_nodes = {start_node}
        previous_head = None

        for edge in nx.edge_dfs(G, start_node):
            tail, head = tailhead(edge)
            if head in explored:
                continue
            if previous_head is not None and tail != previous_head:
                while True:
                    try:
                        popped_edge = edges.pop()
                    except IndexError:
                        edges = []
                        active_nodes = {tail}
                        break
                    else:
                        popped_head = tailhead(popped_edge)[1]
                        active_nodes.remove(popped_head)

                    if edges:
                        last_head = tailhead(edges[-1])[1]
                        if tail == last_head:
                            break
            edges.append(edge)

            if head in active_nodes:
                cycle.extend(edges)
                final_node = head
                break
            else:
                seen.add(head)
                active_nodes.add(head)
                previous_head = head
        if cycle:
            break
        else:
            explored.update(seen)
    else:
        assert len(cycle) == 0
        raise nx.exception.NetworkXNoCycle("No cycle found.")

    for i, edge in enumerate(cycle):
        tail, head = tailhead(edge)
        if tail == final_node:
            break

    return cycle[i:]

# 遍历前 100 个节点，检测是否存在以该节点为起点的环
for start_node in top_100_nodes:
    try:
        # 使用 dfs_find_cycle 检测以 start_node 为起点的环
        cycle = dfs_find_cycle(graph, source=start_node)

        # 统计环的长度
        cycle_length = len(cycle)

        # 获取节点名称（如果存在映射关系）
        # start_node_name = id_to_name.get(start_node, "Unknown")

        # 获取节点的出度
        out_degree = graph.out_degree[start_node]

        # 将环的开始节点、长度、出度和完整环信息存储到结果列表
        results.append({
            "Start Node": start_node,
            # "Name": start_node_name,
            "Cycle Length": cycle_length,
            "Out Degree": out_degree,
            "Cycle": cycle
        })
    except nx.NetworkXNoCycle:
        # 如果没有找到环，跳过
        continue

# 将结果转为 DataFrame
results_df = pd.DataFrame(results)

# 打印结果 DataFrame
print(results_df)

# 保存到 CSV 文件
# results_df.to_csv(
#     "E:\\data\\code\\python project\\graph_embedding\\bfs-dfs\\cycle_results_with_names_and_outdegree.csv", index=False)
