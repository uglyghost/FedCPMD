
# 读取文本文件
with open('E:\\FedCPMD\\out\\client_layer\\B_CINIC10_clients_layer.csv', 'r') as file:
    data = file.read()

# 用逗号分隔每个条目，并去除空格
entries = [entry.strip() for entry in data.split(',')]

# 初始化一个列表来存储满足条件的 id
selected_ids = []

# 遍历每个条目，筛选出 xx 为 'fc1' 的 id
for entry in entries:
    # 使用字符串方法检查 xx 是否为 'fc1'
    if "'fc2'" in entry:
        # 如果是，提取 id 并添加到列表中
        selected_ids.append(entry.split(':')[0])

# 对选出的 id 进行排序
selected_ids = sorted(selected_ids)

# 将排序后的 id 列表转换为以逗号间隔的字符串
result = ', '.join(selected_ids)

# 打印结果
print(result)

