import numpy as np
import torch
from torch import nn
from torch.nn.modules.container import Sequential
from networks import SPU, Normalization, FullyConnected
from utils import SPU as spu
from typing import List


class LinearlyBoundedDomain():
    lower: np.ndarray
    upper: np.ndarray
    verbose: bool
    from_deeppoly: bool

    def __init__(
        self,
        lower: np.ndarray,
        upper: np.ndarray,
        verbose: bool = False,
        from_deeppoly: bool = False
    ) -> None:
        self.lower = lower
        self.upper = upper
        self.verbose = verbose
        self.from_deeppoly = from_deeppoly

    def __repr__(self) -> str:
        return f"LinearlyBoundedDomain(lower: {self.lower}, upper: {self.upper}"

    def transform(self, layer: nn.Module):
        if self.verbose:
            if self.from_deeppoly:
                print(
                    f"\t[LinearlyBoundedDomain]: Current layer is of type {layer}")
            else:
                print(
                    f"[LinearlyBoundedDomain]: Current layer is of type {layer}")

        if isinstance(layer, torch.nn.Flatten):
            lower = self.lower.flatten()
            upper = self.upper.flatten()

        elif isinstance(layer, Normalization):
            mean = layer.mean.detach().numpy()
            std = layer.sigma.detach().numpy()

            lower = (self.lower - mean) / std
            upper = (self.upper - mean) / std

        elif isinstance(layer, torch.nn.Linear):
            weights = layer.weight.detach().numpy()
            bias = layer.bias.detach().numpy()

            lower = np.sum(
                np.minimum(
                    weights * self.lower,
                    weights * self.upper
                ),
                axis=1
            ) + bias

            upper = np.sum(
                np.maximum(
                    weights * self.lower,
                    weights * self.upper
                ),
                axis=1
            ) + bias

        elif isinstance(layer, SPU):
            lower_spu = spu(self.lower)
            upper_spu = spu(self.upper)
            lower = np.minimum(lower_spu, upper_spu)
            upper = np.maximum(lower_spu, upper_spu)

        if self.verbose:
            if self.from_deeppoly:
                print(f"\t\t{self.lower.shape} -> {lower.shape}")
            else:
                print(f"\t{self.lower.shape} -> {lower.shape}")

        return LinearlyBoundedDomain(lower=lower, upper=upper, verbose=self.verbose, from_deeppoly=self.from_deeppoly)


class LinearlyBoundedDomainVerifier():
    domains: List[LinearlyBoundedDomain]
    true_label: int
    eps: float
    verbose: bool
    from_deeppoly: bool

    def __init__(
        self,
        net: torch.nn.Module,
        eps: float,
        inputs: torch.Tensor,
        true_label: int,
        verbose: bool,
        from_deeppoly: bool = False
    ) -> None:
        self.eps = eps
        self.true_label = true_label
        self.verbose = verbose
        self.from_deeppoly = from_deeppoly

        layers = [mod for mod in net.modules() if not isinstance(
            mod, (FullyConnected, torch.nn.Sequential))]

        inputs_np = inputs.detach().numpy()

        lower = np.maximum(0, inputs_np - eps)
        upper = np.minimum(1, inputs_np + eps)

        self.domains = [
            LinearlyBoundedDomain(
                lower=lower,
                upper=upper,
                verbose=self.verbose,
                from_deeppoly=from_deeppoly
            )
        ]

        for layer in layers:
            self.domains.append(self.domains[-1].transform(layer))

    def verify(self) -> bool:
        output = self.domains[-1]
        target_lower = output.lower[self.true_label]
        other_scores = output.upper[self.true_label !=
                                    np.arange(len(self.domains[-1].lower))]

        return target_lower > other_scores.max()
