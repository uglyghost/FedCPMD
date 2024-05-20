from torch.nn import BatchNorm2d
from copy import deepcopy
from fedavg_V1 import FedAvgClient
from collections import OrderedDict
import torch
from torch.distributions import Normal


class FedADPlient(FedAvgClient):
    def __init__(self, model, args, logger, device):
        super().__init__(model, args, logger, device)
        self.model_client = deepcopy(model.to(self.device))
        self.model_global = deepcopy(model.to(self.device))

    def voting_personal_layer(self,
                              local_parameters: OrderedDict[str, torch.Tensor],
                              global_parameters: OrderedDict[str, torch.Tensor],
                              params_name=None,
                              name_list=None):
        if params_name is None:
            return None
        local_parameters = {key: value for key, value in local_parameters.items() if 'bias' not in key and 'conv' not in key}
        global_parameters = {key: value for key, value in global_parameters.items() if 'bias' not in key and 'conv' not in key}

        common_keys = set(local_parameters.keys()) & set(global_parameters.keys())
        similarities = OrderedDict()
        for key in common_keys:
            local_values = local_parameters[key]
            global_values = global_parameters[key]
            if len(local_values) == len(global_values):
                # 1. 计算kl散度 (max选classifier，min选fc1)
                # similarity = self.compute_kl_divergence(local_values, global_values)
                # 2. 计算余弦相似度 (max能选中fc2)
                similarity = torch.cosine_similarity(local_values, global_values, dim=-1, ).mean()
                # 3. 计算流形距离 (min选classifier，max选fc1)
                # similarity = (local_values.shape[1] - (torch.trace(torch.matmul(local_values.t(), global_values))))
                # 4. 欧氏距离 (min能选中fc2)
                # similarity = torch.norm(local_values - global_values, dim=-1).mean()
                # 5. 曼哈顿距离 (min选classifier，max选fc1)
                # similarity = torch.mean(torch.abs(local_values - global_values))
                # 6. 闵可夫斯基距离 (max选classifier，min选fc1)
                # similarity = torch.pow(torch.sum(torch.pow(torch.abs(local_values - global_values), 3)), 1/3)
                # 7. 二者差值 (max能选到fc2)
                # similarity = torch.mean(local_values - global_values)
                print("计算得到的相似度值:", similarity)
                similarities[key] = similarity
            else:
                print("Error: 两个字典中键对应的值维度不一致")
        max_key = max(similarities, key=similarities.get)
        if similarities['fc2.1.weight'] == max(similarities.values()):  # 只要fc2最大，即便有多个最大值，也取fc2（先验知识）
            max_key = 'fc2.1.weight'
        return max_key.split('.', 1)[0]
        # return similarities


    def normalize_tensor(self, tensor):
        min_val = torch.min(tensor)
        max_val = torch.max(tensor)
        normalized_tensor = ((tensor - min_val) / (max_val - min_val))
        return normalized_tensor

    def compute_kl_divergence(self, layer_output, y):
        # 归一化y
        y = self.normalize_tensor(y.float())
        # 将张量转换为浮点数类型
        layer_output = self.normalize_tensor(layer_output.float())
        # 拟合高斯分布
        layer_mean, layer_std = layer_output.mean(), layer_output.std()
        y_mean, y_std = y.mean(), y.std()
        # 检查是否有nan值
        if torch.isnan(layer_mean).any() \
                or torch.isnan(layer_std).any() \
                or torch.isnan(y_mean).any() \
                or torch.isnan(y_std).any():
            # 如果有nan值，跳过此次计算并返回一个占位值，如 0
            return 0

        layer_dist = Normal(layer_mean, layer_std)
        y_dist = Normal(y_mean, y_std)
        kl_div = torch.distributions.kl.kl_divergence(layer_dist, y_dist)
        return kl_div.item()

        # # 2. 计算KL散度
        # PXPY_dist = Normal(layer_mean * y_mean, torch.sqrt(torch.pow(layer_std, 2) + torch.pow(y_std, 2) +
        #                                                    torch.pow(layer_std, 2) * torch.pow(y_mean, 2) +
        #                                                    torch.pow(y_std, 2) * torch.pow(layer_mean, 2)))
        # PXY_dist = Normal(0.5 * (layer_mean + y_mean), 0.5 * torch.sqrt(torch.pow(layer_std, 2) + torch.pow(y_std, 2)))
        # kl_div = torch.distributions.kl.kl_divergence(PXY_dist, PXPY_dist)
        # return kl_div.item()

        # 2. 计算JS散度
        # joint_dist = Normal(0.5 * (layer_mean + y_mean), 0.5 * torch.sqrt(torch.pow(layer_std, 2) + torch.pow(y_std, 2)))
        # kl_div_layer = torch.distributions.kl.kl_divergence(layer_dist, joint_dist)
        # kl_div_y = torch.distributions.kl.kl_divergence(y_dist, joint_dist)
        # if kl_div_layer == kl_div_y:
        #     print("*****************************")
        # js_div = 0.5 * (kl_div_layer + kl_div_y)
        # return js_div.item()
