import numpy as np
import torch
from lin_bounded_domain import LinearlyBoundedDomain
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
    domain: LinearlyBoundedDomain
    input_shape: Tuple[int]
    layer_shape: Tuple[int]

    def __repr__(self) -> str:
        return f"DeepPolyTransformer(layer: {self.layer}, input_shape: {self.input_shape}, layer_shape: {self.layer_shape})"

    def __init__(
        self,
        lower_weights: np.ndarray,
        lower_bias: np.ndarray,
        upper_weights: np.ndarray,
        upper_bias: np.ndarray,
        domain: LinearlyBoundedDomain,
        layer: torch.nn.Module = None,
        from_flatten_shape: Tuple[int, int, int, int] = None,
    ) -> None:
        self.lower_weights = lower_weights
        self.lower_bias = lower_bias

        self.layer = layer

        self.upper_weights = upper_weights
        self.upper_bias = upper_bias

        self.domain = domain

        if isinstance(self.layer, torch.nn.Flatten):
            assert from_flatten_shape

            self.layer_shape = domain.lower.shape
            self.input_shape = from_flatten_shape

        elif isinstance(self.layer, Normalization):
            self.input_shape = self.lower_weights.shape
            self.layer_shape = domain.upper.shape

        else:
            nb_of_input_dimensions = domain.lower.ndim
            self.input_shape = self.lower_weights.shape[nb_of_input_dimensions:]
            self.layer_shape = domain.upper.shape

    def transform(self, from_layer: torch.nn.Module):
        if self.layer:
            print(f"\tTransform from {self.layer}\n")
        if isinstance(from_layer, torch.nn.Flatten):
            lower_weights = np.eye(self.input_shape[2] ** 2)
            lower_bias = self.lower_bias.reshape((self.input_shape[2] ** 2, 1))

            upper_weights = lower_weights.copy()
            upper_bias = lower_bias.copy()

            print(
                f"\t\tWeights shape: {self.input_shape + (lower_weights.shape[0],)}")
            print(
                f"\t\tBias shape: {upper_bias.shape}")

            return DeepPolyTransformer(
                lower_weights=lower_weights,
                lower_bias=lower_bias,
                upper_weights=upper_weights,
                upper_bias=upper_bias,
                layer=from_layer,
                from_flatten_shape=self.input_shape,
                domain=LinearlyBoundedDomain(
                    lower=self.domain.lower.flatten(),
                    upper=self.domain.upper.flatten(),
                    from_deeppoly=True,
                    verbose=False,
                )
            )

        elif isinstance(from_layer, Normalization):
            mean = from_layer.mean.detach().numpy()
            std = from_layer.sigma.detach().numpy()

            lower = (self.domain.lower - mean) / std
            upper = (self.domain.upper - mean) / std

            print(f"\t\tWeights shape: {self.lower_weights.shape}")
            print(f"\t\tBias shape: {self.lower_bias.shape}")

            return DeepPolyTransformer(
                lower_weights=self.lower_weights,
                lower_bias=self.lower_bias * (- mean / std),
                upper_weights=self.upper_weights,
                upper_bias=self.upper_bias * (- mean / std),
                layer=from_layer,
                domain=LinearlyBoundedDomain(
                    lower=lower,
                    upper=upper,
                    from_deeppoly=True,
                    verbose=True,
                )
            )

        elif isinstance(from_layer, torch.nn.Linear):
            lower_weights = from_layer.weight.detach().numpy().T
            upper_weights = lower_weights.copy()

            lower_bias = from_layer.bias.detach().numpy().reshape(-1, 1)
            upper_bias = lower_bias.copy()

        elif isinstance(from_layer, SPU):
            """
            Generate indices for the 5 cases
            """

            zero_SPU = np.sqrt(np.abs(spu(0)))

            # 1st case: upper <= 0, lower <= 0
            idx_1 = self.domain.upper <= 0

            # 2nd case: lower >= 0
            idx_2 = self.domain.lower >= 0

            # 3rd case: upper >= zero_SPU
            idx_3 = self.domain.upper >= zero_SPU

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
                spu(self.domain.lower) > spu(
                    self.domain.upper)
            )

            # 5th case: all the other cases
            idx_5 = np.logical_and(tmp_idx_4, np.logical_not(idx_4))

            # Initiate weights and bias
            lower_weights = np.zeros(self.layer_shape * 2)
            lower_bias = np.zeros(self.layer_shape)

            upper_weights = lower_weights.copy()
            upper_bias = lower_bias.copy()

            diag_slope = np.diag((spu(self.domain.upper) - spu(
                self.domain.lower)) / (self.domain.upper - self.domain.lower))

            """
            1st case: u <= 0
            """
            # w_l = (spu(u) - spu(l)) / (u - l)
            lower_weights[idx_1] = diag_slope[idx_1]

            # b_l = SPU(l) - w_l * l
            lower_bias[idx_1] = (spu(
                self.domain.lower) - lower_weights @ self.domain.lower)[idx_1]

            # w_u = 0 (already the case as initiated)

            # b_u = SPU(l)
            upper_bias[idx_1] = spu(self.domain.lower)[idx_1]

            """
            2nd case: l >= 0
            """
            # Optimal point(s)
            # a = (u + l) / 2
            optimal_points_2 = (
                self.domain.lower + self.domain.upper) / 2.0

            # w_l = self.derivative_SPU(a)
            lower_weights[idx_2] = np.diag(
                derivate_SPU(optimal_points_2))[idx_2]

            # b_l = spu(a) - a * self.derivative_SPU(a)
            lower_bias[idx_2] = (spu(
                optimal_points_2) - optimal_points_2 * derivate_SPU(optimal_points_2))[idx_2]

            # w_u = u + l
            upper_weights[idx_2] = np.diag(
                self.domain.upper + self.domain.lower)[idx_2]

            # b_u = -u * l - spu(0)
            upper_bias[idx_2] = (-self.domain.upper *
                                 self.domain.lower - spu(0))[idx_2]

            """
            3rd case: u >= SPU^(-1)(0)
            """

            # Optimal point
            # a = max((u + l) / 2, SPU_equals_to_zero)
            optimal_points_3 = np.maximum(
                (self.domain.upper + self.domain.lower) / 2.0, zero_SPU)

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
                self.domain.lower) - upper_weights @ self.domain.lower)[idx_3]

            """
            4th case: all the other indices that verify SPU(l) > SPU(u)
            """
            # w_l = 0.0 (already the case)

            # b_l = spu(0)
            lower_bias[idx_4] = spu(0)

            # w_u = 0 (already the case)

            # b_u = max(spu(l), spu(u))
            upper_bias[idx_4] = np.maximum(
                spu(self.domain.lower), spu(self.domain.upper))[idx_4]

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
                self.domain.lower) - upper_weights @ self.domain.lower)[idx_5]

            # TODO: Check if correct

            lower_weights = lower_weights.T
            upper_weights = upper_weights.T
            lower_bias = lower_bias.reshape(-1, 1)
            upper_bias = upper_bias.reshape(-1, 1)

        domain = self.domain.transform(layer=from_layer)

        print(f"\t\tWeights shape: {lower_weights.shape}")
        print(f"\t\tBias shape: {self.lower_bias.shape}")

        return DeepPolyTransformer(
            lower_weights=lower_weights,
            lower_bias=lower_bias,
            upper_weights=upper_weights,
            upper_bias=upper_bias,
            domain=domain,
            layer=from_layer
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

        transformer_shape = (*inputs.shape,)
        print("===" * 20)
        print("Layer: Input()")

        self.transformers = [
            DeepPolyTransformer(
                lower_weights=np.zeros(transformer_shape),
                lower_bias=lower,
                upper_weights=np.zeros(transformer_shape),
                upper_bias=upper,
                layer=None,
                domain=LinearlyBoundedDomain(
                    lower=lower,
                    upper=upper,
                    verbose=self.verbose,
                    from_deeppoly=True
                ),
            )
        ]
        for layer in self.layers:
            print(f"\nLayer: {layer}")
            self.transformers.append(
                self.transformers[-1].transform(layer))

    def verify(self) -> bool:
        current_transformer = self.transformers[-1]

        if self.provable(current_transformer):
            return True

        print("===" * 20)

        print("\n[Backsubstitution]\n")

        for i in reversed(range(3, len(self.layers))):
            layer = self.layers[i]
            print(f"Layer {i + 1}/{len(self.layers)}: {layer}")

            current_transformer = self.back_substitution(
                transformer=current_transformer,
                previous_transformer=self.transformers[i]
            )

            if self.provable(current_transformer):
                return True

        return self.provable(current_transformer)

    def back_substitution(
        self,
        transformer: DeepPolyTransformer,
        previous_transformer: DeepPolyTransformer,
    ) -> DeepPolyTransformer:
        print(f"\nPrevious transformer: {previous_transformer}")

        if self.verbose:
            print(
                f"\tWeights shape: {previous_transformer.lower_weights.shape}")
            print(f"\tBounds shape: {previous_transformer.domain.lower.shape}")
            print(f"\tLayer shape: {previous_transformer.layer_shape}")
            print(f"\tInput shape: {previous_transformer.input_shape}")

        print(f"\nTransformer: {transformer}")

        if self.verbose:
            print(f"\tWeights shape: {transformer.lower_weights.shape}")
            print(f"\tBounds shape: {transformer.domain.lower.shape}")
            print(f"\tLayer shape: {transformer.layer_shape}")
            print(f"\tInput shape: {transformer.input_shape}")

        lower_bias = np.zeros(
            previous_transformer.input_shape).reshape((-1, 1))
        lower_weights = np.zeros(
            (
                *previous_transformer.layer_shape,
                *transformer.input_shape,
            )
        )

        upper_bias = lower_bias.copy()
        upper_weights = lower_weights.copy()

        print(f"\nWanted weights shape: {upper_weights.shape}")
        print(f"Wanted bias shape: {upper_bias.shape}")

        print()

        for neur in range(transformer.layer_shape[0]):
            print(
                f"Prev lower weights: {previous_transformer.lower_weights.shape}")
            print(
                f"(transformer.lower_weights >= 0): {(transformer.lower_weights[neur]).shape}")

            previous_lower_tmp_bias = previous_transformer.lower_bias * \
                (transformer.lower_weights[:, neur] >= 0).reshape((-1, 1))

            previous_lower_tmp_bias += previous_transformer.upper_bias * \
                (transformer.lower_weights[:, neur] < 0).reshape((-1, 1))

            previous_lower_tmp_weights = previous_transformer.lower_weights * (
                transformer.lower_weights[:, neur] >= 0)

            previous_lower_tmp_weights += previous_transformer.upper_weights * \
                (transformer.lower_weights[:, neur] < 0).reshape((-1, 1))

            print(
                f"Previous lower tmp bias: {previous_lower_tmp_bias.shape}")
            print(
                f"Previous lower tmp weights: {previous_lower_tmp_weights.shape}")

            lower_bias[neur] = transformer.lower_weights[neur] @  previous_lower_tmp_bias[neur]

            lower_bias[neur] += transformer.lower_bias[neur]

            lower_weights[neur] = transformer.lower_weights[neur] @ previous_lower_tmp_weights

            # Upper transformer bound
            previous_upper_tmp_bias = previous_transformer.upper_bias * \
                (transformer.upper_weights[:, neur] >= 0) + previous_transformer.lower_bias * (
                    transformer.upper_weights[:, neur] < 0)

            previous_upper_tmp_weights = (previous_transformer.upper_weights * (transformer.upper_weights[:, neur] >= 0)) +\
                (previous_transformer.lower_weights *
                 (transformer.upper_weights[:, neur] < 0))

            upper_bias[neur] = transformer.upper_bias[neur] + \
                transformer.upper_weights[neur] @ previous_upper_tmp_bias.T

            upper_weights[neur] = transformer.upper_weights[neur] @ previous_upper_tmp_weights

        lower = np.sum(
            np.minimum(
                lower_weights.T * previous_transformer.domain.lower +
                lower_bias,
                lower_weights.T * previous_transformer.domain.upper +
                lower_bias,
            ),
            axis=1
        )

        upper = np.sum(
            np.maximum(
                upper_weights.T * previous_transformer.domain.upper +
                upper_bias,
                upper_weights.T * previous_transformer.domain.lower +
                upper_bias
            ),
            axis=1
        )

        print("====" * 20)

        if self.verbose:
            print(
                f"\t\tWeights shape: {lower_weights.shape}")
            print(
                f"\t\tBias shape: {lower_bias.shape}")
            print(f"\t\tLower shape: {lower.shape}")
            print(f"\t\tUpper shape: {upper.shape}")

        domain = LinearlyBoundedDomain(
            lower=lower,
            upper=upper,
            verbose=self.verbose,
            from_deeppoly=True,
        )

        return DeepPolyTransformer(
            lower_weights=lower_weights,
            lower_bias=lower_bias,
            upper_weights=upper_weights,
            upper_bias=upper_bias,
            domain=domain,
        )

    def provable(
        self,
        transformer: DeepPolyTransformer
    ) -> bool:

        # print("[Provable]")
        # print(f"\tInput shape: {transformer.input_shape}")
        # print(f"\tLayer shape: {transformer.layer_shape}")
        # print(f"\tBounds shape: {transformer.domain.lower.shape}")
        # print()

        # print("Scores:")
        # [print(f"{i}: [{transformer.domain.lower[i]:.3f} , {transformer.domain.upper[i]:.3f}]")
        #  for i in range(transformer.domain.lower.shape[0])]

        target_lower = transformer.domain.lower[self.true_label]
        other_scores = transformer.domain.upper[np.arange(
            len(transformer.domain.lower)) != self.true_label]

        # print(f"\tTarget lower: {target_lower}")
        # print(f"\n\tVerified: {target_lower > other_scores.max()}")

        return target_lower > other_scores.max()
