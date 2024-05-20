from typing import Iterator
import torch.nn as nn
import random

import torch
from torch.utils.data import DataLoader

from fedavg import FedAvgClient


device = torch.device('cpu')
if(torch.cuda.is_available()):
    device = torch.device('cpu')
    torch.cuda.empty_cache()
    print("Device set to : " + str(torch.cuda.get_device_name(device)))
else:
    print("Device set to : cpu")
print("============================================================================================")


def trainable_params(model):
    return [p for p in model.parameters() if p.requires_grad]


class NetworkedClient(FedAvgClient):
    def __init__(self, model, args, logger, device):
        super().__init__(model, args, logger, device)
        self.iter_trainloader: Iterator[DataLoader] = None
        self.criterion1 = torch.nn.CrossEntropyLoss(reduction='none').to(self.device)
        self.criterion2 = torch.nn.CrossEntropyLoss(reduction='none').to(self.device)

    def leave_one_out_control_variates(self, losses):
        total_loss = sum(losses)
        loo_losses = [(total_loss - loss) / (len(losses) - 1) for loss in losses]

        # Convert the list of tensor losses to a single tensor
        return torch.stack(loo_losses)

    def compute_correlation_coefficient(self, x, y):
        # 计算向量的均值
        mean_x = torch.mean(x)
        mean_y = torch.mean(y)

        # 计算向量与其均值的差
        xm = x - mean_x
        ym = y - mean_y

        # 计算分子，即协方差的和
        numerator = torch.sum(xm * ym)

        # 计算分母，即两个向量的标准差的乘积
        denominator = torch.sqrt(torch.sum(xm ** 2) * torch.sum(ym ** 2))

        # 计算相关系数
        correlation_coefficient = numerator / denominator

        return correlation_coefficient

    def fit(self):
        """
        The function for specifying operations in local training phase.
        If you wanna implement your method and your method has different local training operations to FedAvg, this method has to be overrided.
        """
        self.model.train()
        for _ in range(self.local_epoch):
            for x, y in self.trainloader:
                # When the current batch size is 1, the batchNorm2d modules in the model would raise error.
                # So the latent size 1 data batches are discarded.
                if len(x) <= 1:
                    continue

                x, y = x.to(self.device), y.to(self.device)
                logit = self.model(x)
                loss = self.criterion1(logit, y)
                processed_losses = self.criterion2(logit, y)

                deps_model = nn.Sequential(
                    nn.Linear(len(loss), 64),
                    nn.Tanh(),
                    nn.Linear(64, 64),
                    nn.Tanh(),
                    nn.Linear(64, len(loss))
                ).to(device)

                # Control Variates 的 leave-one-out 方法处理损失
                # processed_losses = self.leave_one_out_control_variates(processed_losses)
                processed_losses = deps_model(processed_losses)

                # processed_losses = loss

                deps_w = processed_losses.clone()
                # deps_v = torch.exp(deps_w - deps_w.detach()).mean()
                deps_w = (deps_w - deps_w.mean()) / (deps_w.std() + 1e-5)
                deps_v = torch.exp(deps_w - deps_w.detach()).mean()
                alpha = - self.compute_correlation_coefficient(deps_w, loss) / deps_w.std()
                CVor = torch.exp(alpha * (torch.exp(deps_v - deps_v.detach()) - torch.exp(deps_w - deps_w.detach())))

                # 清除已有的梯度
                self.optimizer.zero_grad()
                # 计算总损失
                total_loss = torch.dot(CVor, loss)
                # total_loss = CVor * loss
                # 检查 loss 和 CVor 是否包含 nan
                if torch.isnan(total_loss).any():
                    print("Encountered nan in the total loss. Skipping update.")
                else:
                    total_loss.mean().backward()
                    self.optimizer.step()