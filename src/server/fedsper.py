from argparse import ArgumentParser, Namespace
from collections import OrderedDict
from copy import deepcopy
from collections import Counter
import torch, random, math
from rich.progress import track

from fedavg_V1 import FedAvgServer, get_fedavg_argparser
from src.config.utils import trainable_params

import re
from collections import defaultdict

import os


def get_pfedsim_argparser() -> ArgumentParser:
    parser = get_fedavg_argparser()
    parser.add_argument("-wr", "--warmup_round", type=float, default=0.1)
    parser.add_argument("--server_momentum", type=float, default=0.9)
    return parser


class FedSoftPerServer(FedAvgServer):
    def __init__(
            self,
            algo: str = "FedCPMD_JS_60_0.1_only_decoupling",
            args: Namespace = None,
            unique_model=False,
            default_trainer=True,
    ):
        if args is None:
            args = get_pfedsim_argparser().parse_args()
        super().__init__(algo, args, unique_model, default_trainer)
        self.test_flag = False
        self.weight_matrix = torch.eye(self.client_num, device=self.device)
        self.startup_round = 0
        self.two_smallest_elements = {}
        self.phase1 = 200


    def extract_weights(self, worker_deltas):
        layer_weights = {}
        layer_index = 0
        index_to_layer = {}
        for worker_delta in worker_deltas:
            for layer, tensor in worker_delta.items():
                if 'weight' in layer and 'bn' not in layer:  # 只关心包含'weight'的键
                    weight = tensor.flatten().cpu().detach().numpy()
                    if layer not in index_to_layer:
                        index_to_layer[layer] = layer_index
                        layer_index += 1
                    indexed_layer = index_to_layer[layer]
                    if indexed_layer in layer_weights:
                        layer_weights[indexed_layer].append(weight)
                    else:
                        layer_weights[indexed_layer] = [weight]
        return layer_weights


    def train(self):
        self.unique_model = True
        fedsper_progress_bar = track(
            range(self.startup_round, self.args.global_epoch),
            "[bold green]Personalizing...",
            console=self.logger.stdout,
        )
        self.client_trainable_params = [
            trainable_params(self.global_params_dict, detach=True)
            for _ in self.train_clients
        ]
        all_clients = set(self.train_clients)  # 存储所有客户端的集合
        selected_clients_set = set()  # 存储已选客户端的集合

        self.client_most_common_layers_count = {}  # 存储每个客户端的个性化层统计信息
        self.count_if_executed = 0  # 初始化计数器
        for E in fedsper_progress_bar:  # 前n轮
            if selected_clients_set != all_clients or self.current_epoch < self.phase1:
                self.count_if_executed += 1  # 每次 if 语句执行时，计数器加一
                client_params_cache = []
                delta_cache = []

                self.current_epoch = E
                self.selected_clients = self.client_sample_stream[E]
                selected_clients_set.update(self.selected_clients)  # 更新已选客户端集合

                if (E + 1) % self.args.verbose_gap == 0:
                    self.logger.log(" " * 30, f"TRAINING EPOCH: {E + 1}", " " * 30)
                if (E + 1) % self.args.test_gap == 0:
                    self.test()

                for client_id in self.selected_clients:
                    self.two_smallest_elements.setdefault(client_id, [])
                    if not self.two_smallest_elements[client_id]:
                        self.trainer.personal_params_name.append(
                            name for name in self.model.state_dict() if
                            any(ele in name for ele in 'fc2')
                        )
                    else:
                        self.trainer.personal_params_name.append(
                            name for name in self.model.state_dict() if
                            any(ele in name for ele in self.two_smallest_elements[client_id])
                        )

                    client_pers_params = self.generate_client_params(client_id, client_id)
                    (
                        client_params,
                        delta,
                        _,
                        self.client_stats[client_id][E],
                    ) = self.trainer.train(
                        client_id=client_id,
                        local_epoch=self.clients_local_epoch[client_id],
                        new_parameters=client_pers_params,
                        return_diff=True,
                        verbose=((E + 1) % self.args.verbose_gap) == 0,
                    )
                    client_params_cache.append(client_params)
                    delta_cache.append(delta)

                # client_most_common_layers = self.find_most_common_layer()  # 计算每轮所有客户端的个性化层
                # print("client_most_common_layers:" + str(client_most_common_layers))
                # for client_id in self.selected_clients:
                #     if client_id not in self.client_most_common_layers_count:
                #         self.client_most_common_layers_count[client_id] = Counter()
                #     fc = client_most_common_layers[client_id]
                #     self.client_most_common_layers_count[client_id][fc] += 1
                #
                # for client_id in self.selected_clients:
                #     # 找出每个client_id对应最大值的fc，并将其存放于two_smallest_elements中
                #     max_fc = max(self.client_most_common_layers_count[client_id],
                #                  key=self.client_most_common_layers_count[client_id].get)
                #     self.two_smallest_elements[client_id] = max_fc
                self.update_client_params(client_params_cache)
                self.log_info()
                # import copy
                # self.metrics1, self.metrics2, self.metrics3, self.metrics4 = copy.deepcopy(self.metrics), copy.deepcopy(self.metrics), copy.deepcopy(self.metrics), copy.deepcopy(self.metrics)

        # if selected_clients_set == all_clients:
        #     clustered_clients = {}
        #     for client_id, layer_name in self.two_smallest_elements.items():
        #         if layer_name not in clustered_clients:
        #             clustered_clients[layer_name] = []
        #         clustered_clients[layer_name].append(client_id)

            ##### 将少于5个clients的小类合并到最大类中
            # longest_layer = max(clustered_clients, key=lambda x: len(clustered_clients[x]))
            # for layer_name, client_ids in clustered_clients.copy().items():
            #     # 判断列表长度是否小于5
            #     if len(client_ids) < 5:
            #         # 如果小于5，则将其归并到最长的 layer_name 中
            #         if layer_name != longest_layer:
            #             clustered_clients[longest_layer].extend(client_ids)
            #         # 删除该键值对
            #         del clustered_clients[layer_name]
            # print("###################聚类完成！！！！！######################")
            # sorted_clustered_clients = dict(sorted(clustered_clients.items()))

            # # 接下来，在每个类中进行100轮联邦学习
            # if self.phase1 > 50:
            #     self.phase1 = 50
            # for E in range(200 - self.phase1):  # E的范围从1到150
            #     # 1. ###################################################################################################
            #     # if E % 4 == 1:  # 如果E只为1的倍数而不是2、3的倍数时
            #         layer_name, clients_in_cluster = next(iter(sorted_clustered_clients.items()))
            #         self.cluster_name1 = layer_name
            #         self.number1 = len(clients_in_cluster)
            #         self.two_smallest_elements.clear()
            #         self.trainer.personal_params_name.clear()
            #         idx_and_client_id = {}
            #         for idx, client_id in enumerate(clients_in_cluster):
            #             idx_and_client_id[client_id] = idx
            #             self.two_smallest_elements[client_id] = layer_name
            #         self.trainer.personal_params_name = [
            #             name for name in self.model.state_dict() if
            #             any(ele in name for ele in set(self.two_smallest_elements.values()))
            #         ]
            #         self.client_sample_cluster_stream = [
            #                 random.sample(
            #                     clients_in_cluster,
            #                     len(clients_in_cluster) if len(clients_in_cluster) <= 5 else
            #                     max(5, math.ceil(len(clients_in_cluster) * self.args.join_ratio))
            #                 )
            #                 for _ in range(200 - self.phase1)
            #             ]
            #         self.selected_clients = self.client_sample_cluster_stream[E]
            #         self.weight_matrix_cluster = torch.eye(len(clients_in_cluster), device=self.device)
            #         self.current_epoch = E + self.count_if_executed
            #         if (self.current_epoch + 1) % self.args.verbose_gap == 0:
            #             self.logger.log(" " * 30, f"TRAINING EPOCH: {self.current_epoch + 1}", " " * 30)
            #         if (self.current_epoch + 1) % self.args.test_gap == 0:
            #             self.test()
            #         client_params_cache = []
            #         delta_cache = []
            #         for client_id in self.selected_clients:
            #             client_pers_params = self.generate_client_params(idx_and_client_id[client_id], client_id)
            #             (
            #                 client_params,
            #                 delta,
            #                 _,
            #                 self.client_stats[client_id][self.current_epoch],
            #             ) = self.trainer.train(
            #                 client_id=client_id,
            #                 local_epoch=self.clients_local_epoch[client_id],
            #                 new_parameters=client_pers_params,
            #                 return_diff=True,
            #                 verbose=((E + 1) % self.args.verbose_gap) == 0,
            #             )
            #             client_params_cache.append(client_params)
            #             delta_cache.append(delta)
            #         self.update_client_params(client_params_cache)
            #         self.update_weight_matrix(delta_cache)
            #         self.log_info1()
            #
            # for E in range(200 - self.phase1):  # E的范围从1到150
            #     # 2. ###################################################################################################
            #     # if E % 4 == 2:  # 如果E只为2的倍数而3不是倍数时
            #         if list(sorted_clustered_clients.items())[1:2]:
            #             layer_name, clients_in_cluster = next(iter(list(sorted_clustered_clients.items())[1:2]))
            #             # print(layer_name, clients_in_cluster)
            #             self.cluster_name2 =layer_name
            #             self.number2 = len(clients_in_cluster)
            #             self.two_smallest_elements.clear()
            #             self.trainer.personal_params_name.clear()
            #             idx_and_client_id = {}
            #             for idx, client_id in enumerate(clients_in_cluster):
            #                 idx_and_client_id[client_id] = idx
            #                 self.two_smallest_elements[client_id] = layer_name
            #             self.trainer.personal_params_name = [
            #                 name for name in self.model.state_dict() if
            #                 any(ele in name for ele in set(self.two_smallest_elements.values()))
            #             ]
            #             self.client_sample_cluster_stream = [
            #                 random.sample(
            #                     clients_in_cluster,
            #                     len(clients_in_cluster) if len(clients_in_cluster) <= 5 else
            #                     max(5, math.ceil(len(clients_in_cluster) * self.args.join_ratio))
            #                 )
            #                 for _ in range(200 - self.phase1)
            #             ]
            #             self.selected_clients = self.client_sample_cluster_stream[E]
            #             self.weight_matrix_cluster = torch.eye(len(clients_in_cluster), device=self.device)
            #             self.current_epoch = E + self.count_if_executed
            #             if (self.current_epoch + 1) % self.args.verbose_gap == 0:
            #                 self.logger.log(" " * 30, f"TRAINING EPOCH: {self.current_epoch + 1}", " " * 30)
            #             if (self.current_epoch + 1) % self.args.test_gap == 0:
            #                 self.test()
            #             client_params_cache = []
            #             delta_cache = []
            #             for client_id in self.selected_clients:
            #                 client_pers_params = self.generate_client_params(idx_and_client_id[client_id], client_id)
            #                 (
            #                     client_params,
            #                     delta,
            #                     _,
            #                     self.client_stats[client_id][self.current_epoch],
            #                 ) = self.trainer.train(
            #                     client_id=client_id,
            #                     local_epoch=self.clients_local_epoch[client_id],
            #                     new_parameters=client_pers_params,
            #                     return_diff=True,
            #                     verbose=((E + 1) % self.args.verbose_gap) == 0,
            #                 )
            #                 client_params_cache.append(client_params)
            #                 delta_cache.append(delta)
            #             self.update_client_params(client_params_cache)
            #             self.update_weight_matrix(delta_cache)
            #             self.log_info2()
            #
            # for E in range(200 - self.phase1):  # E的范围从1到150
            #     # 3. ###################################################################################################
            #     # if E % 4 == 3:  # 如果E只为3的倍数时
            #         if list(sorted_clustered_clients.items())[2:3]:
            #             layer_name, clients_in_cluster = next(iter(list(sorted_clustered_clients.items())[2:3]))
            #             print(layer_name, clients_in_cluster)
            #             self.cluster_name3 =layer_name
            #             self.number3 = len(clients_in_cluster)
            #             self.two_smallest_elements.clear()
            #             self.trainer.personal_params_name.clear()
            #             idx_and_client_id = {}
            #             for idx, client_id in enumerate(clients_in_cluster):
            #                 idx_and_client_id[client_id] = idx
            #                 self.two_smallest_elements[client_id] = layer_name
            #             self.trainer.personal_params_name = [
            #                 name for name in self.model.state_dict() if
            #                 any(ele in name for ele in set(self.two_smallest_elements.values()))
            #             ]
            #             self.client_sample_cluster_stream = [
            #                 random.sample(
            #                     clients_in_cluster,
            #                     len(clients_in_cluster) if len(clients_in_cluster) <= 5 else
            #                     max(5, math.ceil(len(clients_in_cluster) * self.args.join_ratio))
            #                 )
            #                 for _ in range(200 - self.phase1)
            #             ]
            #             self.selected_clients = self.client_sample_cluster_stream[E]
            #             self.weight_matrix_cluster = torch.eye(len(clients_in_cluster), device=self.device)
            #             self.current_epoch = E + self.count_if_executed
            #             if (self.current_epoch + 1) % self.args.verbose_gap == 0:
            #                 self.logger.log(" " * 30, f"TRAINING EPOCH: {self.current_epoch + 1}", " " * 30)
            #             if (self.current_epoch + 1) % self.args.test_gap == 0:
            #                 self.test()
            #             client_params_cache = []
            #             delta_cache = []
            #             for client_id in self.selected_clients:
            #                 client_pers_params = self.generate_client_params(idx_and_client_id[client_id], client_id)
            #                 (
            #                     client_params,
            #                     delta,
            #                     _,
            #                     self.client_stats[client_id][self.current_epoch],
            #                 ) = self.trainer.train(
            #                     client_id=client_id,
            #                     local_epoch=self.clients_local_epoch[client_id],
            #                     new_parameters=client_pers_params,
            #                     return_diff=True,
            #                     verbose=((E + 1) % self.args.verbose_gap) == 0,
            #                 )
            #                 client_params_cache.append(client_params)
            #                 delta_cache.append(delta)
            #             self.update_client_params(client_params_cache)
            #             self.update_weight_matrix(delta_cache)
            #             self.log_info3()

            # for E in range(200 - self.phase1):  # E的范围从1到150
            #     # 4. ########################################################################################################
            #     # if E % 4 == 0:  # 如果E只为3的倍数
            #         if list(sorted_clustered_clients.items())[3:4]:
            #             layer_name, clients_in_cluster = next(iter(list(sorted_clustered_clients.items())[3:4]))
            #             print(layer_name, clients_in_cluster)
            #             self.cluster_name4 = layer_name
            #             self.number4 = len(clients_in_cluster)
            #             self.two_smallest_elements.clear()
            #             self.trainer.personal_params_name.clear()
            #             idx_and_client_id = {}
            #             for idx, client_id in enumerate(clients_in_cluster):
            #                 idx_and_client_id[client_id] = idx
            #                 self.two_smallest_elements[client_id] = layer_name
            #             self.trainer.personal_params_name = [
            #                 name for name in self.model.state_dict() if
            #                 any(ele in name for ele in set(self.two_smallest_elements.values()))
            #             ]
            #             self.client_sample_cluster_stream = [
            #                 random.sample(
            #                     clients_in_cluster,
            #                     len(clients_in_cluster) if len(clients_in_cluster) <= 5 else
            #                     max(5, math.ceil(len(clients_in_cluster) * self.args.join_ratio))
            #                 )
            #                 for _ in range(200 - self.phase1)
            #             ]
            #             self.selected_clients = self.client_sample_cluster_stream[E]
            #             self.weight_matrix_cluster = torch.eye(len(clients_in_cluster), device=self.device)
            #             self.current_epoch = E + self.count_if_executed
            #             if (self.current_epoch + 1) % self.args.verbose_gap == 0:
            #                 self.logger.log(" " * 30, f"TRAINING EPOCH: {self.current_epoch + 1}", " " * 30)
            #             if (self.current_epoch + 1) % self.args.test_gap == 0:
            #                 self.test()
            #             client_params_cache = []
            #             delta_cache = []
            #             for client_id in self.selected_clients:
            #                 client_pers_params = self.generate_client_params(idx_and_client_id[client_id], client_id)
            #                 (
            #                     client_params,
            #                     delta,
            #                     _,
            #                     self.client_stats[client_id][self.current_epoch],
            #                 ) = self.trainer.train(
            #                     client_id=client_id,
            #                     local_epoch=self.clients_local_epoch[client_id],
            #                     new_parameters=client_pers_params,
            #                     return_diff=True,
            #                     verbose=((E + 1) % self.args.verbose_gap) == 0,
            #                 )
            #                 client_params_cache.append(client_params)
            #                 delta_cache.append(delta)
            #             self.update_client_params(client_params_cache)
            #             self.update_weight_matrix(delta_cache)
            #             self.log_info4()


    def find_most_common_layer(self):
        # 存储每个客户端的出现次数最多的层
        client_most_common_layers = {}

        # 打开并读取日志文件的内容
        with open(f'kl_log_{self.args.algo}.txt', 'r') as file:
            lines = file.readlines()
            content = ''.join(lines)
            # 使用正则表达式查找所有层的名称
            for client_id in self.selected_clients:
                layer_counts = defaultdict(int)  # 为每个客户端创建新的计数器
                # 使用正则表达式匹配客户端ID和层的名称
                pattern = rf'{client_id} Layer with minimum contribution to output y: (\w+)'
                matches = re.findall(pattern, content)
                for match in matches:
                    layer_counts[match] += 1

                # 找到客户端出现次数最多的层
                most_common_layer = max(layer_counts, key=layer_counts.get)
                client_most_common_layers[client_id] = most_common_layer

        # 删除 kl_log.txt 文件
        os.remove(f'kl_log_{self.args.algo}.txt')
        # 返回每个客户端出现次数最多的层的字典
        return client_most_common_layers

    @torch.no_grad()
    def generate_client_params(self, idx, client_id):
        new_parameters = OrderedDict(
            zip(
                self.trainable_params_name,
                deepcopy(self.client_trainable_params[client_id]),
            )
        )
        if not self.test_flag:
            if sum(self.weight_matrix[client_id]) > 1:
                weights = self.weight_matrix[client_id].clone()
                weights = torch.exp(weights)
                weights /= weights.sum()
                N_dim = weights.shape[0]
                # # 在CUDA上生成一个N x N的方阵，每个元素值为1/N
                uniform_matrix = torch.full((N_dim,), 1 / N_dim).cpu()
                for i, (name, layer_params) in enumerate(zip(
                        self.trainable_params_name, zip(*self.client_trainable_params)
                )):
                    if idx == client_id:
                        new_parameters[name] = torch.sum(
                            torch.stack(layer_params, dim=-1) * uniform_matrix, dim=-1
                        )
                    else:
                        new_parameters[name] = torch.sum(
                            torch.stack(layer_params, dim=-1) * weights, dim=-1
                        )
        return new_parameters


    @torch.no_grad()
    def update_weight_matrix(self, delta_cache):
        for idx_i, i in enumerate(self.selected_clients):
            client_params_i = delta_cache[idx_i]
            for idx_j, j in enumerate(self.selected_clients[idx_i + 1:]):
                client_params_j = delta_cache[idx_i + idx_j + 1]
                if len(set(self.two_smallest_elements.values())) > 1:
                    sim_ij_classifier = max(
                        0,
                        torch.cosine_similarity(
                            client_params_i["classifier.weight"].cpu(),
                            client_params_j["classifier.weight"].cpu(),
                            dim=-1,
                        ).mean().item()
                    )
                    sim_ij_classifier = torch.tensor(sim_ij_classifier).to('cpu')
                    self.weight_matrix_cluster[idx_i, idx_j] = self.weight_matrix_cluster[
                        idx_j, idx_i] = sim_ij_classifier
                else:
                    if 'classifier' in set(self.two_smallest_elements.values()):
                        classifier_weight_i_cpu = client_params_i["classifier.weight"].cpu()
                        classifier_weight_j_cpu = client_params_j["classifier.weight"].cpu()
                    else:
                        classifier_weight_i_cpu = client_params_i[str(list(set(self.two_smallest_elements.values()))[0]) + ".1.weight"].cpu()
                        classifier_weight_j_cpu = client_params_j[str(list(set(self.two_smallest_elements.values()))[0]) + ".1.weight"].cpu()
                    # 计算余弦相似度
                    sim_ij_classifier = max(
                        0,
                        torch.cosine_similarity(
                            classifier_weight_i_cpu,
                            classifier_weight_j_cpu,
                            dim=-1,
                        ).mean().item()
                    )
                    # 计算流形距离
                    # dist_ij_classifier = ((client_params_i["classifier.weight"]).shape[1]
                    #           - (torch.trace(torch.matmul((client_params_i["classifier.weight"]).t(),
                    #                                       client_params_j["classifier.weight"])
                    #                          )))
                    # sim_ij_classifier = 1 / (1 + torch.exp(dist_ij_classifier))

                    # sim_ij_fc2 = max(
                    #     0,
                    #     torch.cosine_similarity(
                    #         fc2_weight_i_cpu,
                    #         fc2_weight_j_cpu,
                    #         dim=-1,
                    #     ).mean().item()
                    # )

                    sim_ij_classifier = torch.tensor(sim_ij_classifier).to('cpu')
                    self.weight_matrix_cluster[idx_i, idx_j] = self.weight_matrix_cluster[idx_j, idx_i] = sim_ij_classifier

if __name__ == "__main__":
    server = FedSoftPerServer()
    server.run()
