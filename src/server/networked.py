from copy import deepcopy
from argparse import ArgumentParser, Namespace

import torch

from fedavg import FedAvgServer, get_fedavg_argparser
from src.client.networked import NetworkedClient


def get_Networked_argparser() -> ArgumentParser:
    parser = get_fedavg_argparser()
    parser.add_argument("--global_lr", type=float, default=1.0)
    parser.add_argument("--server_momentum", type=float, default=0.9)
    return parser


class NetworkedServer(FedAvgServer):
    def __init__(
        self,
        algo: str = "Networked_non",
        args: Namespace = None,
        unique_model=False,
        default_trainer=True,
    ):
        if args is None:
            args = get_Networked_argparser().parse_args()
        super().__init__(algo, args, unique_model, default_trainer)
        self.trainer = NetworkedClient(
            deepcopy(self.model), self.args, self.logger, self.device
        )
        self.global_optimizer = torch.optim.SGD(
            list(self.global_params_dict.values()),
            lr=1.0,
            # momentum=self.args.server_momentum,
            # nesterov=True,
        )

    def rloo_process(self, values: torch.Tensor, num_clients: int) -> torch.Tensor:
        """
        Process a tensor of values using the Residual Leave-One-Out method on the GPU, applied over the last dimension.

        Args:
        - values (torch.Tensor): Tensor of values with shape [A, B, C, D, E].
        - num_clients (int): Number of clients.

        Returns:
        - Processed tensor of values with the same shape.
        """

        # Sum across the last dimension and expand dimensions for broadcasting
        total = values.sum(dim=-1, keepdim=True)
        # Apply RLOO process efficiently using broadcasting
        rloo_values = (total - values) / (num_clients - 1)
        return rloo_values


    @torch.no_grad()
    def aggregate(self, delta_cache, weight_cache):
        weights = torch.tensor(weight_cache, device=self.device) / sum(weight_cache)

        delta_list = [list(delta.values()) for delta in delta_cache]

        aggregated_delta = []
        for layer_delta in zip(*delta_list):
            aggregated_delta.append(
                torch.sum(torch.stack(layer_delta, dim=-1) * weights, dim=-1)
            )

        # self.global_optimizer.zero_grad()
        for param, diff in zip(self.global_params_dict.values(), aggregated_delta):
            param.grad = diff.data
        # self.global_optimizer.step()

    '''
    @torch.no_grad()
    def aggregate(self, delta_cache, weight_cache):
        num_clients = len(delta_cache)
        weights = torch.tensor(weight_cache, device=self.device) / sum(weight_cache)

        delta_list = [list(delta.values()) for delta in delta_cache]

        aggregated_delta = []
        for layer_delta in zip(*delta_list):
            stacked_delta = torch.stack(layer_delta, dim=-1)
            # Use rloo_process instead of sum
            rloo_values = self.rloo_process(stacked_delta, num_clients)
            aggregated_delta.append(torch.sum(rloo_values * weights, dim=-1))

        for param, diff in zip(self.global_params_dict.values(), aggregated_delta):
            param.grad = diff.data
    '''


if __name__ == "__main__":
    server = NetworkedServer()
    server.run()