import os
import pandas as pd

directory = "E:\\FedCPMD\\out\\FedCPMD_B_50_0.1_0.4"

# 遍历目录中的csv文件
csv_files = [file for file in os.listdir(directory) if file.endswith('.csv')]


# 初始化字典，用于存储平均值
averages_before = {}
averages_after = {}

# 循环处理每个csv文件
for file in csv_files:
    # 读取csv文件
    dataset_path = os.path.join(directory, file)
    df = pd.read_csv(dataset_path)

    # 解析文件名，提取 dataset 和 layer_name
    number, dataset, layer_name = file.split('_')[:3]

    # 取出最后5行数据并计算平均值
    last_15_rows = df.tail(5)
    average_before = round(last_15_rows['test_before'].mean(), 3)
    average_after = round(last_15_rows['test_after'].mean(), 3)

    # 存储平均值到对应的字典中
    averages_before[(dataset, layer_name)] = average_before
    averages_after[(dataset, layer_name)] = average_after

# 生成Markdown表格 - test_before
print("test_before:")

# 计算每列的和，并统计非NA值的个数
sum_values = {layer_name: sum(averages_before.get((dataset, layer_name), 0)
                              for dataset in set([x[0] for x in averages_before.keys()])
                              if averages_before.get((dataset, layer_name), 'NA') != 'NA')
                              for layer_name in set([x[1] for x in averages_before.keys()])}

count_values = {layer_name: sum(1 for dataset in set([x[0] for x in averages_before.keys()])
                                 if averages_before.get((dataset, layer_name), 'NA') != 'NA')
                                 for layer_name in set([x[1] for x in averages_before.keys()])}

# 计算每列的均值
mean_values = {layer_name: round(sum_values[layer_name] / count_values[layer_name], 3)
               for layer_name in sum_values}

averages_before_sort = dict(sorted(averages_before.items(), key=lambda x: (x[0][0], x[0][1])))
averages_after_sort = dict(sorted(averages_after.items(), key=lambda x: (x[0][0], x[0][1])))


print("| Dataset |", end="")
for layer_name in set([x[1] for x in averages_before_sort.keys()]):
    print(f" {layer_name} |", end="")
print()
print("| --- |", end="")
for _ in range(len(set([x[1] for x in averages_before_sort.keys()]))):
    print(" --- |", end="")
print()
for dataset in set([x[0] for x in averages_before_sort.keys()]):
    print(f"| {dataset} |", end="")
    for layer_name in set([x[1] for x in averages_before_sort.keys()]):
        print(f" {averages_before_sort.get((dataset, layer_name), 'NA')} |", end="")
    print()

# 输出每列的均值
print("| Mean |", end="")
for layer_name in set([x[1] for x in averages_before_sort.keys()]):
    print(f" {mean_values[layer_name]} |", end="")
print()


# 生成Markdown表格 - test_after
print("test_after:")

# 计算每列的均值
# 计算每列的和，并统计非NA值的个数
sum_values = {layer_name: sum(averages_after_sort .get((dataset, layer_name), 0)
                              for dataset in set([x[0] for x in averages_after_sort .keys()])
                              if averages_after_sort .get((dataset, layer_name), 'NA') != 'NA')
                              for layer_name in set([x[1] for x in averages_after_sort .keys()])}

count_values = {layer_name: sum(1 for dataset in set([x[0] for x in averages_after_sort .keys()])
                                 if averages_after_sort .get((dataset, layer_name), 'NA') != 'NA')
                                 for layer_name in set([x[1] for x in averages_after_sort .keys()])}

# 计算每列的均值
mean_values = {layer_name: round(sum_values[layer_name] / count_values[layer_name], 3)
               for layer_name in sum_values}

print("| Dataset |", end="")
for layer_name in set([x[1] for x in averages_after_sort .keys()]):
    print(f" {layer_name} |", end="")
print()
print("| --- |", end="")
for _ in range(len(set([x[1] for x in averages_after_sort .keys()]))):
    print(" --- |", end="")
print()
for dataset in set([x[0] for x in averages_after_sort .keys()]):
    print(f"| {dataset} |", end="")
    for layer_name in set([x[1] for x in averages_after_sort .keys()]):
        print(f" {averages_after_sort .get((dataset, layer_name), 'NA')} |", end="")
    print()

# 输出每列的均值
print("| Mean |", end="")
for layer_name in set([x[1] for x in averages_after_sort .keys()]):
    print(f" {mean_values[layer_name]} |", end="")
print()


# import csv
# # 将结果保存为 CSV 文件
# with open('averages_before.csv', 'w', newline='') as csvfile:
#     writer = csv.writer(csvfile)
#     writer.writerow(['Dataset'] + list(set([x[1] for x in averages_before_sort.keys()])))
#     for dataset in set([x[0] for x in averages_before_sort.keys()]):
#         row_data = [dataset] + [averages_before_sort.get((dataset, layer_name), 'NA')
#                                 for layer_name in set([x[1] for x in averages_before_sort.keys()])]
#         writer.writerow(row_data)


# #############################################################################################
# # 加权取mean
# import os
# import pandas as pd
#
# directory = "E:\\FedCPMD\\out\\FedCPMD_W_40_0.1_3_layer"
#
# # 遍历目录中的csv文件
# csv_files = [file for file in os.listdir(directory) if file.endswith('.csv')]
#
# # 初始化字典，用于存储平均值
# averages_before = {}
# averages_after = {}
#
# # 循环处理每个csv文件
# for file in csv_files:
#     # 读取csv文件
#     dataset_path = os.path.join(directory, file)
#     df = pd.read_csv(dataset_path)
#
#     # 解析文件名，提取 number、layer_name 和 dataset
#     number, layer_name, dataset = file.split('_')[:3]
#
#     # 取出最后5行数据并计算平均值
#     last_5_rows = df.tail(7)
#     average_before = round(last_5_rows['test_before'].mean(), 3)
#     average_after = round(last_5_rows['test_after'].mean(), 3)
#
#     # 存储平均值到对应的字典中
#     averages_before[(number, dataset, layer_name)] = average_before
#     averages_after[(number, dataset, layer_name)] = average_after
#
# # 计算加权平均值
# def compute_weighted_average(averages):
#     # 初始化一个字典，用于存储每个 dataset 对应的加权平均值
#     weighted_averages_per_dataset = {}
#
#     # 遍历 averages_after 字典
#     for dataset in set([x[1] for x in averages.keys()]):
#         weighted_sum = 0
#         total_number = 0
#
#         # 遍历每个 dataset 对应的所有 layer_name
#         for layer_name in set([x[2] for x in averages.keys() if x[1] == dataset]):
#             # 获取该 dataset 和 layer_name 对应的 number 和值
#             number = [x[0] for x in averages.keys() if x[1] == dataset and x[2] == layer_name][0]
#             value = averages[(number, dataset, layer_name)]
#
#             # 计算加权和
#             weighted_sum += int(number) * value
#             total_number += int(number)
#
#         # 计算加权平均值
#         weighted_average = round(weighted_sum / total_number if total_number != 0 else 0, 3)
#
#         # 将加权平均值存储到字典中
#         weighted_averages_per_dataset[dataset] = weighted_average
#     return weighted_averages_per_dataset
#
#
# # 计算加权平均值
# weighted_averages_before = compute_weighted_average(averages_before)
# weighted_averages_after = compute_weighted_average(averages_after)
#
# # 输出 Markdown 表格
# def print_markdown_table(title, averages, weighted_averages):
#     new_averages = {}
#     # 遍历原始字典的键值对
#     for key, value in averages.items():
#         # 从原始键中获取 dataset 和 layer_name
#         dataset, layer_name = key[1], key[2]
#         # 使用新的键 (dataset, layer_name) 存储值
#         new_averages[(dataset, layer_name)] = value
#     print(f"{title}:")
#     print("| Dataset |", end="")
#     layer_names = sorted(set([x[2] for x in averages.keys()]))
#     print(" | ".join(layer_names) + " | Weighted Average |")
#     print("| --- |", end="")
#     for _ in layer_names:
#         print(" --- |", end="")
#     print(" --- |")
#     for dataset in sorted(set([x[1] for x in averages.keys()])):
#         print(f"| {dataset} |", end="")
#         for layer_name in layer_names:
#             value = new_averages.get((dataset, layer_name), 'NA')
#             print(f" {value} |", end="")
#         weighted_average = weighted_averages.get(dataset, 'NA')
#         print(f" {weighted_average} |")
#     print()
#
# # 输出 Markdown 表格和加权平均值
# print_markdown_table("test_before", averages_before, weighted_averages_before)
# print_markdown_table("test_after", averages_after, weighted_averages_after)

