import os
import pandas as pd

# 初始化一个字典来存储平均值
average_values = {}

# 遍历out文件夹，获取其中所有方法名字
methods = [d for d in os.listdir('alpha/out_a=1') if os.path.isdir(os.path.join('alpha/out_a=1', d))]

length = 10

# 循环遍历每一个方法
for method in methods:
    method_path = os.path.join('alpha/out_a=1', method)
    datasets = [f for f in os.listdir(method_path) if f.endswith('.csv')]

    # 循环遍历每一个数据集ch
    for dataset in datasets:
        dataset_path = os.path.join('alpha/out_a=1', method, dataset)
        df = pd.read_csv(dataset_path)
        last_10_avg_before = round(df['test_before'].tail(length).mean(), 3)
        last_10_avg_after = round(df['test_after'].tail(length).mean(), 3)
        dataset = dataset.replace("_acc_metrics.csv", "")  # Remove '_acc_metrics.csv' from the dataset name

        if dataset not in average_values:
            average_values[dataset] = {}

        average_values[dataset][method] = {
            'before': last_10_avg_before,
            'after': last_10_avg_after
        }
# 创建Markdown表格
markdown_table = "| Dataset | "
for method in methods:
    markdown_table += f"{method} Before | {method} After | "
markdown_table += "\n|---|"
for method in methods:
    markdown_table += "---|---|"
markdown_table += "\n"

for dataset, methods_dict in average_values.items():
    markdown_table += f"| {dataset} | "
    best_before = max(methods_dict.items(), key=lambda x: x[1]['before'])[0]
    best_after = max(methods_dict.items(), key=lambda x: x[1]['after'])[0]

    for method in methods:
        method_info = methods_dict.get(method, {'before': 'N/A', 'after': 'N/A'})
        before_value = f"{'**' if method == best_before else ''}{method_info['before']}{'***' if method == best_before else ''}"
        after_value = f"{'**' if method == best_after else ''}{method_info['after']}{'***' if method == best_after else ''}"
        markdown_table += f"{before_value} | {after_value} | "

    markdown_table += "\n"

# 输出Markdown表格
print(markdown_table)
