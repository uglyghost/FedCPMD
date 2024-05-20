import pickle
from argparse import Namespace
from collections import OrderedDict
from copy import deepcopy
from typing import Dict, List, Tuple, Union
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset
from torchvision import transforms

PROJECT_DIR = Path(__file__).parent.parent.parent.absolute()

from src.config.utils import trainable_params, evalutate_model, Logger
from src.config.models import DecoupledModel
from data.utils.constants import MEAN, STD
from data.utils.datasets import DATASETS
import torch.nn as nn


class FedAvgClient:

    def __init__(
        self,
        model: DecoupledModel,
        args: Namespace,
        logger: Logger,
        device: torch.device,
    ):
        self.fix_layer = 'classifier'
        self.args = args
        self.device = device
        self.client_id: int = None
        self.output_file = None

        # load dataset and clients' data indices
        try:
            partition_path = PROJECT_DIR / "data" / self.args.dataset / "partition.pkl"
            with open(partition_path, "rb") as f:
                partition = pickle.load(f)
        except:
            raise FileNotFoundError(f"Please partition {args.dataset} first.")

        self.data_indices: List[List[int]] = partition["data_indices"]

        # --------- you can define your own data transformation strategy here ------------
        general_data_transform = transforms.Compose(
            [transforms.Normalize(MEAN[self.args.dataset], STD[self.args.dataset])]
        )
        general_target_transform = transforms.Compose([])
        train_data_transform = transforms.Compose([])
        train_target_transform = transforms.Compose([])
        # --------------------------------------------------------------------------------

        self.dataset = DATASETS[self.args.dataset](
            root=PROJECT_DIR / "data" / args.dataset,
            args=args.dataset_args,
            general_data_transform=general_data_transform,
            general_target_transform=general_target_transform,
            train_data_transform=train_data_transform,
            train_target_transform=train_target_transform,
        )

        self.trainloader: DataLoader = None
        self.testloader: DataLoader = None
        self.trainset: Subset = Subset(self.dataset, indices=[])
        self.testset: Subset = Subset(self.dataset, indices=[])
        self.global_testset: Subset = None
        if self.args.global_testset:
            all_testdata_indices = []
            for indices in self.data_indices:
                all_testdata_indices.extend(indices["test"])
            self.global_testset = Subset(self.dataset, all_testdata_indices)

        self.model = model.to(self.device)
        self.local_epoch = self.args.local_epoch
        self.criterion = torch.nn.CrossEntropyLoss().to(self.device)
        self.logger = logger
        self.personal_params_dict: Dict[int, Dict[str, torch.Tensor]] = {}
        self.personal_params_name: List[str] = []
        self.init_personal_params_dict: Dict[str, torch.Tensor] = {
            key: param.clone().detach()
            for key, param in self.model.state_dict(keep_vars=True).items()
            if not param.requires_grad
        }
        self.opt_state_dict = {}

        self.optimizer = torch.optim.SGD(
            params=trainable_params(self.model),
            lr=self.args.local_lr,
            momentum=self.args.momentum,
            weight_decay=self.args.weight_decay,
        )
        self.init_opt_state_dict = deepcopy(self.optimizer.state_dict())

    def write_output(self, message):
        # 如果文件对象不存在，打开文件
        if self.output_file is None:
            self.output_file = open(f'kl_log_{self.args.algo}.txt', 'a')

        # 写入消息
        self.output_file.write(message + '\n')

        # 关闭文件
        self.output_file.close()

        # 将文件对象设置为None，以便下次使用时重新打开
        self.output_file = None

    def load_dataset(self):
        """This function is for loading data indices for No.`self.client_id` client."""
        self.trainset.indices = self.data_indices[self.client_id]["train"]
        self.testset.indices = self.data_indices[self.client_id]["test"]
        self.trainloader = DataLoader(self.trainset, self.args.batch_size)
        if self.args.global_testset:
            self.testloader = DataLoader(self.global_testset, self.args.batch_size)
        else:
            self.testloader = DataLoader(self.testset, self.args.batch_size)

    def train_and_log(self, verbose=False) -> Dict[str, Dict[str, float]]:
        """This function includes the local training and logging process.

        Args:
            verbose (bool, optional): Set to `True` for print logging info onto the stdout (Controled by the server by default). Defaults to False.

        Returns:under
            Dict[str, Dict[str, float]]: The logging info, which contains metric stats.
        """
        before = {
            "train_loss": 0,
            "test_loss": 0,
            "train_correct": 0,
            "test_correct": 0,
            "train_size": 1,
            "test_size": 1,
        }
        after = deepcopy(before)
        before = self.evaluate()
        if self.local_epoch > 0:
            self.fit()
            self.save_state()
            after = self.evaluate()
        if verbose:
            if len(self.trainset) > 0 and self.args.eval_train:
                self.logger.log(
                    "client [{}] (train)  [bold red]loss: {:.4f} -> {:.4f}   [bold blue]acc: {:.2f}% -> {:.2f}%".format(
                        self.client_id,
                        before["train_loss"] / before["train_size"],
                        after["train_loss"] / after["train_size"],
                        before["train_correct"] / before["train_size"] * 100.0,
                        after["train_correct"] / after["train_size"] * 100.0,
                    )
                )
            if len(self.testset) > 0 and self.args.eval_test:
                self.logger.log(
                    "client [{}] (test)  [bold red]loss: {:.4f} -> {:.4f}   [bold blue]acc: {:.2f}% -> {:.2f}%".format(
                        self.client_id,
                        before["test_loss"] / before["test_size"],
                        after["test_loss"] / after["test_size"],
                        before["test_correct"] / before["test_size"] * 100.0,
                        after["test_correct"] / after["test_size"] * 100.0,
                    )
                )
        eval_stats = {"before": before, "after": after}
        return eval_stats


    def set_parameters(self, new_parameters: OrderedDict[str, torch.Tensor]):
        """Load model parameters received from the server.

        Args:
            new_parameters (OrderedDict[str, torch.Tensor]): Parameters of FL model.
        """
        personal_parameters = self.personal_params_dict.get(
            self.client_id, self.init_personal_params_dict
        )
        self.optimizer.load_state_dict(
            self.opt_state_dict.get(self.client_id, self.init_opt_state_dict)
        )
        self.model.load_state_dict(new_parameters, strict=False)
        # personal params would overlap the dummy params from new_parameters from the same layers
        self.model.load_state_dict(personal_parameters, strict=False)

    def save_state(self):
        """Save client model personal parameters and the state of optimizer at the end of local training."""
        self.personal_params_dict[self.client_id] = {
            key: param.clone().detach()
            for key, param in self.model.state_dict(keep_vars=True).items()
            if (not param.requires_grad) or (key in self.personal_params_name)
        }
        self.opt_state_dict[self.client_id] = deepcopy(self.optimizer.state_dict())

    def train(
            self,
            client_id: int,
            local_epoch: int,
            new_parameters: OrderedDict[str, torch.Tensor],
            # name: List[str],  # 使用列表类型的注解
            return_diff=True,
            verbose=False,
    ) -> Tuple[Union[OrderedDict[str, torch.Tensor], List[torch.Tensor]],
    Union[OrderedDict[str, torch.Tensor], List[torch.Tensor]], int, Dict]:
        """
        The funtion for including all operations in client local training phase.
        If you wanna implement your method, consider to override this funciton.

        Args:
            client_id (int): The ID of client.

            local_epoch (int): The number of epochs for performing local training.

            new_parameters (OrderedDict[str, torch.Tensor]): Parameters of FL model.

            return_diff (bool, optional):
            Set as `True` to send the difference between FL model parameters that before and after training;
            Set as `False` to send FL model parameters without any change.  Defaults to True.

            verbose (bool, optional): Set to `True` for print logging info onto the stdout (Controled by the server by default). Defaults to False.

        Returns:
            Tuple[Union[OrderedDict[str, torch.Tensor], List[torch.Tensor]], int, Dict]:
            [The difference / all trainable parameters, the weight of this client, the evaluation metric stats].
        """
        # self.personal_params_name = name
        self.client_id = client_id
        self.local_epoch = local_epoch
        self.load_dataset()
        self.set_parameters(new_parameters)
        eval_stats = self.train_and_log(verbose=verbose)
        if return_diff:
            delta = OrderedDict()
            for (name, p0), p1 in zip(
                    new_parameters.items(), trainable_params(self.model)
            ):
                delta[name] = p0 - p1

            return trainable_params(self.model), delta, len(self.trainset), eval_stats
        else:
            return (
                trainable_params(self.model, detach=True),
                trainable_params(self.model, detach=True),
                len(self.trainset),
                eval_stats,
                # self.result
            )

    def fit(self):
        """
        The function for specifying operations in local training phase.
        If you wanna implement your method and your method has different local training operations to FedAvg, this method has to be overrided.
        """
        # 对于需要梯度的部分
        for param in self.model.parameters():
            param.requires_grad = True
        self.model.train()
        for _ in range(self.local_epoch):
            for x, y in self.trainloader:
                # When the current batch size is 1, the batchNorm2d modules in the model would raise error.
                # So the latent size 1 data batches are discarded.
                if len(x) <= 1:
                    continue

                x, y = x.to(self.device), y.to(self.device)
                if self.model.name == "DecoupledModel":
                    logit = self.model(x)
                else:
                    logit_vae_list = self.model(x)
                    # print(logit_vae_list)
                    logit = logit_vae_list[-1]

                    # # 1. KL(output,y)
                    max_contrib_layer = self.model.determine_max_contribution_layer(logit_vae_list, x, y)
                    # # 2. KL(output,x)
                    # # max_contrib_layer = self.model.determine_max_contribution_layer(logit_vae_list, x)
                    # # print("Layer with minimum contribution to output y:", max_contrib_layer)
                    # 在你需要保存输出的地方，调用write_output()方法
                    message = f"{self.client_id} Layer with minimum contribution to output y: {max_contrib_layer}"
                    self.write_output(message)
                loss = self.criterion(logit, y)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

    # def fit(self):
    #     self.model.train()
    #     for _ in range(self.local_epoch):
    #         for x, y in self.trainloader:
    #             if len(x) <= 1:
    #                 continue
    #
    #             x, y = x.to(self.device), y.to(self.device)
    #             outputs = self.model(x)
    #             loss = self.criterion(outputs[-1], y)
    #
    #             self.optimizer.zero_grad()
    #             loss.backward()
    #             self.optimizer.step()


    def get_layer_contributions(self, inputs):
        outputs = [self.model.layers[0].input]  # 输入层
        for layer in self.model.layers:
            if layer.trainable:  # 仅考虑可训练的层
                outputs.append(layer(inputs))
        return outputs

    @torch.no_grad()
    def evaluate(self, model: torch.nn.Module = None) -> Dict[str, float]:
        """The evaluation function. Would be activated before and after local training if `eval_test = True` or `eval_train = True`.

        Args:
            model (torch.nn.Module, optional): The target model needed evaluation (set to `None` for using `self.model`). Defaults to None.

        Returns:
            Dict[str, float]: The evaluation metric stats.
        """
        # disable train data transform while evaluating
        self.dataset.enable_train_transform = False

        eval_model = self.model if model is None else model
        eval_model.eval()
        train_loss, test_loss = 0, 0
        train_correct, test_correct = 0, 0
        train_sample_num, test_sample_num = 0, 0
        criterion = torch.nn.CrossEntropyLoss(reduction="sum")

        if len(self.testset) > 0 and self.args.eval_test:
            test_loss, test_correct, test_sample_num = evalutate_model(
                model=eval_model,
                dataloader=self.testloader,
                criterion=criterion,
                device=self.device,
            )

        if len(self.trainset) > 0 and self.args.eval_train:
            train_loss, train_correct, train_sample_num = evalutate_model(
                model=eval_model,
                dataloader=self.trainloader,
                criterion=criterion,
                device=self.device,
            )

        self.dataset.enable_train_transform = True

        return {
            "train_loss": train_loss,
            "test_loss": test_loss,
            "train_correct": train_correct,
            "test_correct": test_correct,
            "train_size": float(max(1, train_sample_num)),
            "test_size": float(max(1, test_sample_num)),
        }

    def test(
        self, client_id: int, new_parameters: OrderedDict[str, torch.Tensor]
    ) -> Dict[str, Dict[str, float]]:
        """Test function. Only be activated while in FL test round.

        Args:
            client_id (int): The ID of client.
            new_parameters (OrderedDict[str, torch.Tensor]): The FL model parameters.

        Returns:
            Dict[str, Dict[str, float]]: the evalutaion metrics stats.
        """
        self.client_id = client_id
        self.load_dataset()
        self.set_parameters(new_parameters)

        before = {
            "train_loss": 0,
            "train_correct": 0,
            "train_size": 1.0,
            "test_loss": 0,
            "test_correct": 0,
            "test_size": 1.0,
        }
        after = deepcopy(before)

        before = self.evaluate()
        if self.args.finetune_epoch > 0:
            self.finetune()
            after = self.evaluate()
        return {"before": before, "after": after}

    def finetune(self):
        """
        The fine-tune function. If your method has different fine-tuning opeation, consider to override this.
        This function will only be activated while in FL test round.
        """
        self.model.train()
        for _ in range(self.args.finetune_epoch):
            for x, y in self.trainloader:
                if len(x) <= 1:
                    continue

                x, y = x.to(self.device), y.to(self.device)
                if self.model.name == "DecoupledModel":
                    logit = self.model(x)
                else:
                    logit_vae_list = self.model(x)
                    logit = logit_vae_list[-1]
                    # max_contrib_layer = self.model.determine_max_contribution_layer(logit_vae_list, y)
                    # print("Layer with minimum contribution to output y:", max_contrib_layer)

                loss = self.criterion(logit, y)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                # 需要再次清零梯度，因为我们将进行新的前向/后向传播
                self.optimizer.zero_grad()

                # # 使用x和y计算Fisher信息并确定最大贡献层
                # max_contrib_layer = self.model.determine_max_contribution_layer(x, y)
                # # message = f"{self.client_id} Layer with maximum contribution to output y: {max_contrib_layer}"
                # message = f"Layer with minimum contribution to output y: {max_contrib_layer}"
                # self.write_output(message)

                # # 可能需要再次执行优化器步骤，如果determine_max_contribution_layer中进行了额外的前向/后向传播
                # self.optimizer.step()

    def compute_correlation_coefficient(self,x, y):

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

