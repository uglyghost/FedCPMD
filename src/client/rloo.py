from fedavg import FedAvgClient
import torch

class RLOOClient(FedAvgClient):
    def __init__(self, model, args, logger, device):
        super(RLOOClient, self).__init__(model, args, logger, device)
        self.criterion = torch.nn.CrossEntropyLoss(reduction='none').to(self.device)

    def leave_one_out_control_variates(self, losses):
        total_loss = sum(losses)
        loo_losses = [(total_loss - loss) / (len(losses) - 1) for loss in losses]

        # Convert the list of tensor losses to a single tensor
        return torch.stack(loo_losses)

    def fit(self):
        self.model.train()
        for _ in range(self.local_epoch):
            for x, y in self.trainloader:
                if len(x) <= 1:
                    continue

                x, y = x.to(self.device), y.to(self.device)
                logit = self.model(x)
                loss = self.criterion(logit, y)
                # Control Variates的leave-one-out方法处理损失
                processed_losses = self.leave_one_out_control_variates(loss)

                deps_w = processed_losses.clone()
                deps_v = torch.exp(deps_w - deps_w.detach()).mean()
                CVor = torch.exp((torch.exp(deps_v - deps_v.detach()) - torch.exp(deps_w - deps_w.detach())))

                # 使用处理后的损失来更新模型
                self.optimizer.zero_grad()
                # total_loss = sum(loss) / len(loss)
                total_loss = CVor * loss
                total_loss.mean().backward()
                self.optimizer.step()
