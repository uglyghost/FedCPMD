import os
import pandas as pd

# 初始化一个字典来存储平均值
average_values = {}

# 遍历out文件夹，获取其中所有方法名字

length = 5
directory = "E:\\FedCPMD\\out\\FedCPMD_JS_60_0.1_only_decoupling"

# 遍历目录中的csv文件
files = [file for file in os.listdir(directory) if file.endswith('.csv')]
# 初始化字典，用于存储平均值
averages_before = {}
averages_after = {}
# 循环遍历每一个数据集ch
for file in files:
    dataset_path = os.path.join(directory, file)
    df = pd.read_csv(dataset_path)
    last_10_avg_before = round(df['test_before'].tail(length).mean(), 3)
    last_10_avg_after = round(df['test_after'].tail(length).mean(), 3)
    dataset = file.replace("_acc_metrics.csv", "")  # Remove '_acc_metrics.csv' from the dataset name

    # 取出最后5行数据并计算平均值
    last_15_rows = df.tail(5)
    average_before = round(last_15_rows['test_before'].mean(), 3)
    average_after = round(last_15_rows['test_after'].mean(), 3)

    # 存储平均值到对应的字典中
    averages_before[dataset] = average_before
    averages_after[dataset] = average_after

# 输出Markdown表头
print("| Dataset | Average Test Before | Average Test After |")
print("| ------- | ------------------- | ------------------ |")

# 遍历字典，逐行输出数据
for dataset, avg_before in averages_before.items():
    avg_after = averages_after[dataset]
    print(f"| {dataset} | {avg_before} | {avg_after} |")
