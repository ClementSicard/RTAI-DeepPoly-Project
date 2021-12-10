import numpy as np
import torch
from box import Box
from typing import Tuple, List
import torch
from networks import Normalization, SPU, FullyConnected
from utils import SPU as spu, derivate_SPU
from itertools import product


class DeepPolyTransformer():
    lower_weights: np.ndarray
    lower_bias: np.ndarray
    upper_weights: np.ndarray
    upper_bias: np.ndarray
    layer: torch.nn.Module
    box: Box
    input_shape: Tuple[int]
    layer_shape: Tuple[int]

    def __repr__(self) -> str:
        return f"DeepPolyTransformer(layer: {self.layer})"

    def __init__(
        self,
        lower_weights: np.ndarray,
        lower_bias: np.ndarray,
        upper_weights: np.ndarray,
        upper_bias: np.ndarray,
        box: Box,
        layer: torch.nn.Module = None
    ) -> None:
        self.lower_weights = lower_weights
        self.lower_bias = lower_bias

        self.layer = layer

        self.upper_weights = upper_weights
        self.upper_bias = upper_bias

        self.box = box

        nb_of_input_dimensions = box.lower.ndim
        self.input_shape = self.lower_weights.shape[nb_of_input_dimensions:]
        self.layer_shape = box.upper.shape

    def transform(self, layer: torch.nn.Module):
        if isinstance(layer, torch.nn.Flatten):
            lower_weights = self.lower_weights
            lower_bias = self.lower_bias

            upper_weights = self.upper_weights
            upper_bias = self.upper_bias

        elif isinstance(layer, Normalization):
            mean = layer.mean.detach().numpy()
            std = layer.sigma.detach().numpy()

            lower_weights = np.zeros(2 * self.layer_shape)
            indices = tuple(list(map(lambda x: list(x), zip(
                *product(*map(range, self.layer_shape))))) * 2)

            lower_weights[indices] = 1 / std

            lower_bias = np.ones(self.layer_shape) * (-mean / std)

            upper_weights = lower_weights.copy()
            upper_bias = lower_bias.copy()

        elif isinstance(layer, torch.nn.Linear):
            weights = layer.weight.detach().numpy()
            bias = layer.bias.detach().numpy()

            lower_weights, upper_weights = weights.copy(), weights.copy()
            lower_bias, upper_bias = bias.copy(), bias.copy()

        elif isinstance(layer, SPU):
            """
            Generate indices for the 5 cases
            """

            zero_SPU = np.sqrt(np.abs(spu(0)))

            # 1st case: upper <= 0, lower <= 0
            idx_1 = self.box.upper <= 0

            # 2nd case: lower >= 0
            idx_2 = self.box.lower >= 0

            # 3rd case: upper >= zero_SPU
            idx_3 = self.box.upper >= zero_SPU

            # all the remaining indices
            tmp_idx_4 = np.logical_and(
                np.logical_and(
                    np.logical_not(idx_1),
                    np.logical_not(idx_2)
                ),
                np.logical_not(idx_3),
            )

            # 4th case: the other cases that verify with SPU(l) > SPU(u)
            idx_4 = np.logical_and(
                tmp_idx_4,
                spu(self.box.lower) > spu(
                    self.box.upper)
            )

            # 5th case: all the other cases
            idx_5 = np.logical_and(tmp_idx_4, np.logical_not(idx_4))

            # Initiate weights and bias
            lower_weights = np.zeros(self.layer_shape * 2)
            lower_bias = np.zeros(self.layer_shape)

            upper_weights = np.zeros(self.layer_shape * 2)
            upper_bias = np.zeros(self.layer_shape)

            diag_slope = np.diag((spu(self.box.upper) - spu(
                self.box.lower)) / (self.box.upper - self.box.lower))

            """
            1st case: u <= 0
            """
            # w_l = (spu(u) - spu(l)) / (u - l)
            lower_weights[idx_1] = diag_slope[idx_1]

            # b_l = SPU(l) - w_l * l
            lower_bias[idx_1] = (spu(
                self.box.lower) - lower_weights @ self.box.lower)[idx_1]

            # w_u = 0 (already the case as initiated)

            # b_u = SPU(l)
            upper_bias[idx_1] = spu(self.box.lower)[idx_1]

            """
            2nd case: l >= 0
            """
            # Optimal point(s)
            # a = (u + l) / 2
            optimal_points_2 = (
                self.box.lower + self.box.upper) / 2.0

            # w_l = self.derivative_SPU(a)
            lower_weights[idx_2] = np.diag(
                derivate_SPU(optimal_points_2))[idx_2]

            # b_l = spu(a) - a * self.derivative_SPU(a)
            lower_bias[idx_2] = (spu(
                optimal_points_2) - optimal_points_2 * derivate_SPU(optimal_points_2))[idx_2]

            # w_u = u + l
            upper_weights[idx_2] = np.diag(
                self.box.upper + self.box.lower)[idx_2]

            # b_u = -u * l - spu(0)
            upper_bias[idx_2] = (-self.box.upper *
                                 self.box.lower - spu(0))[idx_2]

            """
            3rd case: u >= SPU^(-1)(0)
            """

            # Optimal point
            # a = max((u + l) / 2, SPU_equals_to_zero)
            optimal_points_3 = np.maximum(
                (self.box.upper + self.box.lower) / 2.0, zero_SPU)

            # w_l = self.derivative_SPU(a)
            lower_weights[idx_3] = np.diag(
                derivate_SPU(optimal_points_3))[idx_3]

            # b_l = spu(a) - a * self.derivative_SPU(a)
            lower_bias[idx_3] = (spu(
                optimal_points_3) - optimal_points_3 @ derivate_SPU(optimal_points_3))[idx_3]

            # w_u = (spu(u) - spu(l)) / (u - l)
            upper_weights[idx_3] = diag_slope[idx_3]

            # b_u = spu(l) - w_u * l
            upper_bias[idx_3] = (spu(
                self.box.lower) - upper_weights @ self.box.lower)[idx_3]

            """
            4th case: all the other indices that verify SPU(l) > SPU(u)
            """
            # w_l = 0.0 (already the case)

            # b_l = spu(0)
            lower_bias[idx_4] = spu(0)

            # w_u = 0 (already the case)

            # b_u = max(spu(l), spu(u))
            upper_bias[idx_4] = np.maximum(
                spu(self.box.lower), spu(self.box.upper))[idx_4]

            """
            5th case: all the other indices
            """
            # w_l = 0.0 (already the case)

            # b_l = spu(0)
            lower_bias[idx_5] = spu(0)

            # w_u = (spu(u) - spu(l)) / (u - l)
            upper_weights[idx_5] = diag_slope[idx_5]

            # b_u = spu(l) - w_u * l
            upper_bias[idx_5] = (spu(
                self.box.lower) - upper_weights @ self.box.lower)[idx_5]

        box = self.box.transform(layer=layer)

        return DeepPolyTransformer(
            lower_weights=lower_weights,
            lower_bias=lower_bias,
            upper_weights=upper_weights,
            upper_bias=upper_bias,
            box=box,
            layer=layer
        )


class DeepPolyVerifier():
    transformers: List[DeepPolyTransformer]
    true_label: int
    eps: float
    layers: List[torch.nn.Module]
    verbose: bool

    def __init__(
        self,
        net: torch.nn.Module,
        inputs: torch.Tensor,
        eps: float,
        true_label: int,
        verbose: bool
    ) -> None:

        self.eps = eps
        self.true_label = true_label
        self.verbose = verbose

        self.layers = [mod for mod in net.modules() if not isinstance(
            mod, (torch.nn.Sequential, FullyConnected))]

        self.transformers = []

        inputs_np = inputs.detach().numpy()

        lower = inputs_np - eps
        upper = inputs_np + eps

        transformer_shape = (*inputs.shape, 0)

        self.transformers = [
            DeepPolyTransformer(
                lower_weights=np.zeros(transformer_shape),
                lower_bias=lower,
                upper_weights=np.zeros(transformer_shape),
                upper_bias=upper,
                box=Box(lower=lower, upper=upper,
                        verbose=self.verbose, from_deeppoly=True),
                layer=None
            )
        ]
        for layer in self.layers:
            self.transformers.append(self.transformers[-1].transform(layer))

    def verify(self) -> bool:
        current_transformer = self.transformers[-1]

        for i in reversed(range(1, len(self.layers))):
            layer = self.layers[i]
            print(f"Layer {i + 1}/{len(self.layers)}: {layer}")

            if isinstance(layer, torch.nn.Flatten):
                break

            current_transformer = self.back_substitution(
                transformer=current_transformer,
                previous_transformer=self.transformers[i]
            )

            # lower_weights = current_transformer.lower_weights
            # lower_bias = current_transformer.lower_bias
            # upper_weights = current_transformer.upper_weights
            # upper_bias = current_transformer.upper_bias

            if self.provable(current_transformer):
                return True

        return self.provable(current_transformer)

    def back_substitution(
        self,
        transformer: DeepPolyTransformer,
        previous_transformer: DeepPolyTransformer
    ) -> DeepPolyTransformer:

        print(f"Transformer: {transformer}")
        print(f"\tWeights shape: {transformer.lower_weights.shape}")
        print(f"\tBounds shape: {transformer.box.lower.shape}")
        print(f"Previous transformer: {previous_transformer}")
        print(f"\tWeights shape: {previous_transformer.lower_weights.shape}")
        print(f"\tBounds shape: {previous_transformer.box.lower.shape}")

        lower_bias = np.zeros(previous_transformer.input_shape)
        lower_weights = np.zeros(
            (
                *transformer.input_shape,
                *previous_transformer.input_shape,
            )
        )

        upper_bias = np.zeros(previous_transformer.input_shape)
        upper_weights = np.zeros(
            (
                *transformer.input_shape,
                *previous_transformer.input_shape,
            )
        )

        for neur in range(transformer.layer_shape[0]):
            if self.verbose:
                print(f"\ti = {neur}")

            # Lower transformer bound
            prev_l_tmp_bias = (previous_transformer.lower_bias * (transformer.lower_weights[neur] >= 0)) +\
                (previous_transformer.upper_bias *
                 (transformer.lower_weights[neur] < 0))

            previous_l_tmp_weights = (previous_transformer.lower_weights.T * (transformer.lower_weights[neur] >= 0)) + (
                previous_transformer.upper_weights.T * (transformer.lower_weights[neur] < 0))

            lower_bias[neur] = transformer.lower_bias[neur] + \
                transformer.lower_weights[neur] @ prev_l_tmp_bias.T

            lower_weights[neur] = previous_l_tmp_weights @ transformer.lower_weights[neur]

            # Upper transformer bound
            previous_u_tmp_bias = (previous_transformer.upper_bias * (transformer.upper_weights[neur] >= 0)) +\
                (previous_transformer.lower_bias *
                 (transformer.upper_weights[neur] < 0))
            previous_u_tmp_weights = (previous_transformer.upper_weights.T * (transformer.upper_weights[neur] >= 0)) +\
                (previous_transformer.lower_weights.T *
                 (transformer.upper_weights[neur] < 0))

            upper_bias[neur] = transformer.upper_bias[neur] + \
                transformer.upper_weights[neur] @ previous_u_tmp_bias.T

            upper_weights[neur] = previous_u_tmp_weights @ transformer.upper_weights[neur]

        print("====" * 20)

        lower = np.sum(
            np.minimum(
                lower_weights.T * previous_transformer.box.lower +
                lower_bias.reshape((-1, 1)),
                lower_weights.T * previous_transformer.box.upper +
                lower_bias.reshape((-1, 1)),
            ),
            axis=1
        )

        upper = np.sum(
            np.maximum(
                upper_weights.T * previous_transformer.box.upper +
                upper_bias.reshape((-1, 1)),
                upper_weights.T * previous_transformer.box.lower +
                upper_bias.reshape((-1, 1))
            ),
            axis=1
        )

        if self.verbose:
            print(
                f"Lower_weights.T shape: {(lower_weights.T).shape}")
            print(
                f"Product shape: {(lower_weights.T * previous_transformer.box.upper).shape}")
            print(
                f"Bias shape: {lower_bias.shape}")
            print(f"Lower shape: {lower.shape}")
            print(f"Upper shape: {upper.shape}")

        box = Box(lower=lower, upper=upper,
                  verbose=self.verbose, from_deeppoly=True)

        return DeepPolyTransformer(
            lower_weights=lower_weights,
            lower_bias=lower_bias,
            upper_weights=upper_weights,
            upper_bias=upper_bias,
            box=box
        )

    def provable(
        self,
        transformer: DeepPolyTransformer
    ) -> bool:

        target_lower = transformer.box.lower[self.true_label]
        other_scores = transformer.box.upper[np.arange(
            len(transformer.box.lower)) != self.true_label]

        return target_lower > other_scores.max()
