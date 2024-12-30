# 导入必要的库
import pandas as pd
import re

# 读取文件
def load_investment_data(file_path):
    data = []
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            parts = line.strip().split('\t')
            if len(parts) != 2:
                continue
            investor_id, investments_str = parts
            # 清洗数据，移除所有非数字和逗号的字符
            cleaned_data = re.sub(r'[^0-9,\[\]]', '', investments_str)
            # 将清洗后的字符串分割并转换为整数列表
            investments = [int(i) for i in cleaned_data.strip('[]').split(',') if i]
            data.append((int(investor_id), investments))
    return data

# 加载数据
file_path = '投资人的投资轨迹（2015-2020）.txt'
investment_data = load_investment_data(file_path)

# 转换为DataFrame以便于分析
df = pd.DataFrame(investment_data, columns=['InvestorID', 'Investments'])

# 计算每位投资者的投资次数
df['InvestmentCount'] = df['Investments'].apply(len)

# 统计每个投资者的投资路径长度
path_length_distribution = df['InvestmentCount'].value_counts(normalize=True) * 100

# 输出结果
print("Path Length\tPercentage of Meta-Paths")
for path_length, percentage in path_length_distribution.items():
    print(f"{path_length}\t{percentage:.1f}%")

# 如果你想保存结果到CSV文件中
# path_length_distribution.to_csv('path_length_distribution.csv', index_label='Path Length', header=['Percentage of Meta-Paths'])
