import random
import time
def random_walk(graph, start_node, walk_length):
    path = [start_node]
    current_node = start_node
    visited_nodes = set([start_node])  # 记录已访问的节点

    for _ in range(walk_length - 1):
        neighbors = list(graph.neighbors(current_node))
        # 从邻居中排除已经访问过的节点
        unvisited_neighbors = [n for n in neighbors if n not in visited_nodes]
        if not unvisited_neighbors:
            break
        current_node = random.choice(unvisited_neighbors)
        path.append(current_node)
        visited_nodes.add(current_node)  # 将当前节点标记为已访问

    return path

def deepwalk(graph, num_walks, walk_length):
    walks = []  # 存储所有随机游走路径
    start_time = time.time()
    for _ in range(num_walks):
        for node in list(graph.nodes()):
            walk = [node]
            visited = set()
            while len(walk) < walk_length:
                current_node = walk[-1]
                neighbors = list(graph.neighbors(current_node))
                unvisited_neighbors = [n for n in neighbors if n not in visited]
                if len(unvisited_neighbors) == 0:
                    break
                next_node = random.choice(unvisited_neighbors)
                walk.append(next_node)
                visited.add(next_node)
            walks.append(walk)
    end_time = time.time()
    node2vec_times = end_time - start_time
    return walks, node2vec_times



#====================================================================================================================
#第二次优化

def node2vec_new_random_walks(graph, num_walks, walk_length, p=1, q=1, sampling_ratio=1.0, sampling_threshold=10):
    """
    优化后的随机游走路径生成函数，提升性能，减少不必要开销。
    """
    print("调用node2vec_new_random_walks")

    walks = []
    nodes = list(graph.nodes())
    start_time = time.time()
    for _ in range(num_walks):
        for node in nodes:
            walk = [node]
            visited = {node: 0}  # 使用字典记录访问过的节点及其步长位置
            while len(walk) < walk_length:
                current_node = walk[-1]
                neighbors = list(graph.neighbors(current_node))

                if not neighbors:  # 如果没有邻居，停止游走
                    break

                # 如果邻居数量超过阈值，仅采样部分邻居
                if len(neighbors) > sampling_threshold:
                    sampled_neighbors = random.sample(neighbors, max(1, int(len(neighbors) * sampling_ratio)))
                else:
                    sampled_neighbors = neighbors

                # 动态计算转移概率
                probabilities = compute_transition_probs_optimized(graph, current_node, sampled_neighbors, p, q)

                if not probabilities or sum(probabilities) == 0:  # 概率为0时停止游走
                    break

                # 按概率选择下一个节点
                next_node = random.choices(sampled_neighbors, weights=probabilities, k=1)[0]

                # 防止重复访问环路节点
                if next_node in visited:
                    break

                walk.append(next_node)
                visited[next_node] = len(walk)

            walks.append(walk)
    end_time = time.time()
    node2vec_new_time = end_time - start_time
    return walks, node2vec_new_time


def compute_transition_probs_optimized(graph, current_node, neighbors, p, q):
    """
    高效计算从当前节点到邻居节点的转移概率。
    """
    probabilities = []

    # 当前节点的前一个节点（用于 node2vec 的返回策略）
    prev_node = None if len(neighbors) == 0 else neighbors[-1]  # 简单假设最后一个为前节点

    for neighbor in neighbors:
        # 动态调整概率
        if prev_node is not None and neighbor == prev_node:
            probabilities.append(1 / p)  # 返回概率
        elif graph.has_edge(current_node, neighbor):
            probabilities.append(1.0)  # 节点有边连接，设置基本概率
        else:
            probabilities.append(1 / q)  # 偏离策略概率

    # 归一化概率
    total_prob = sum(probabilities)
    if total_prob > 0:
        probabilities = [x / total_prob for x in probabilities]

    return probabilities

#第二次优化
#====================================================================================================================




#==================================================================================
#未进行优化
def node2vec_random_walks(graph, num_walks, walk_length, p=1, q=1):

    print("运行node2vec_random_walks")

    walks = []
    nodes = list(graph.nodes())
    start_time = time.time()

    for _ in range(num_walks):
        for node in nodes:
            walk = [node]
            visited = set([(node, 0)])  # 记录访问过的节点和访问的顺序位置
            while len(walk) < walk_length:
                current_node = walk[-1]
                neighbors = list(graph.neighbors(current_node))
                if not neighbors:
                    break
                next_node = node2vec_walk(neighbors, current_node, p, q, visited)
                if next_node is None:
                    break
                walk.append(next_node)
                visited.add((next_node, len(walk)))
            walks.append(walk)
    end_time = time.time()
    node2vec_old_time = end_time - start_time
    return walks, node2vec_old_time

def node2vec_walk(neighbors, current_node, p, q, visited):
    num_neighbors = len(neighbors)
    if num_neighbors == 0:
        return None

    # 如果邻居只有一个节点，直接返回该邻居
    if num_neighbors == 1:
        return neighbors[0]

    probabilities = [0] * num_neighbors
    for i, neighbor in enumerate(neighbors):
        # 如果邻居已经被访问过，则跳过
        if (neighbor, len(visited)) in visited:
            probabilities[i] = 0
            continue

        # 计算当前节点到邻居节点的转移概率
        if num_neighbors > 1:  # 只有多个邻居时才计算这些转移概率
            if i > 0 and neighbors[i - 1] == neighbor:
                probabilities[i] = (1 - p - q) / (num_neighbors - 1)
            elif (i + 1) % num_neighbors == 0 and neighbors[(i + 1) % num_neighbors] == neighbor:
                probabilities[i] = (1 - p - q) / (num_neighbors - 1)
            else:
                probabilities[i] = 1 / (num_neighbors - 1)

            # 应用 p 和 q 的影响
            if i > 0 and neighbors[i - 1] == neighbor:
                probabilities[i] *= p
            elif (i + 1) % num_neighbors == 0 and neighbors[(i + 1) % num_neighbors] == neighbor:
                probabilities[i] *= q

    total_prob = sum(probabilities)
    # 如果所有概率为0，直接返回 None
    if total_prob == 0:
        return None

    probabilities = [x / total_prob for x in probabilities]  # 归一化概率
    next_node = random.choices(neighbors, weights=probabilities, k=1)[0]
    return next_node

#未进行优化
#==================================================================================




#=========================================================================================
#第一次优化

def precompute_transition_probs(graph, p, q):
    """
    预计算所有边的转移概率，并以稀疏存储方式记录非零值。
    """
    transition_probs = {}
    for node in graph.nodes():
        neighbors = list(graph.neighbors(node))
        if not neighbors:
            continue
        prob_dict = {}
        for neighbor in neighbors:
            prob_dict[neighbor] = 1.0 / len(neighbors)  # 均匀概率初始化
            # 可扩展：应用 p, q 参数的调整逻辑（如需要，按实际算法需求改写）

        # 只存储非零概率值
        transition_probs[node] = prob_dict
    return transition_probs


def node2vec_new0_random_walks(graph, num_walks, walk_length, p=1, q=1, sampling_ratio=1.0):

    print("调用node2vec_new0_random_walks")

    transition_probs = precompute_transition_probs(graph, p, q)

    walks = []
    nodes = list(graph.nodes())

    for _ in range(num_walks):
        for node in nodes:
            walk = [node]
            while len(walk) < walk_length:
                current_node = walk[-1]
                neighbors = list(graph.neighbors(current_node))
                if not neighbors:
                    break

                # 采样邻居以降低计算复杂度
                sampled_neighbors = random.sample(neighbors, max(1, int(len(neighbors) * sampling_ratio)))

                # 获取转移概率（缓存的概率值）
                prob_dict = transition_probs.get(current_node, {})
                probabilities = [prob_dict.get(neighbor, 0) for neighbor in sampled_neighbors]

                if sum(probabilities) == 0:  # 如果概率全为零，停止游走
                    break

                # 归一化概率
                probabilities = [x / sum(probabilities) for x in probabilities]

                # 按概率选择下一个节点
                next_node = random.choices(sampled_neighbors, weights=probabilities, k=1)[0]
                walk.append(next_node)

            walks.append(walk)

    return walks

#第一次优化
#=========================================================================================
