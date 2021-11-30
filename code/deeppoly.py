from math import ceil
import torch
from torch import overrides
from networks import SPU, Normalization, DerivativeSPU
from typing import List, Tuple


class DeepPolyCertifier:
    net: torch.nn.Module
    eps: torch.float
    inputs: torch.Tensor
    true_label: int
    transformers: List[torch.nn.Module]
    verifier: torch.nn.Module

    def __init__(
        self,
        net: torch.nn.Module,
        eps: float,
        inputs: torch.Tensor,
        true_label: int
    ) -> None:
        self.net = net
        self.eps = eps
        self.inputs = inputs
        self.true_label = true_label
        self._generate_transformers()
        self.verifier = torch.nn.Sequential(*self.transformers)

    def _generate_transformers(self) -> None:
        self.transformers = [DeepPolyInputsTransformer(eps=self.eps)]

        current_transformer = None

        for layer in self.net.layers:
            if isinstance(layer, Normalization):
                current_transformer = DeepPolyNormalizationTransformer()

            elif isinstance(layer, torch.nn.Flatten):
                current_transformer = DeepPolyFlattenTransformer(
                    previous=current_transformer
                )

            elif isinstance(layer, SPU):
                current_transformer = DeepPolySPUTransformer(
                    previous=current_transformer
                )

            elif isinstance(layer, torch.nn.Linear):
                bias = layer.bias.detach()
                weights = layer.weight.detach()

                current_transformer = DeepPolyAffineTransformer(
                    bias=bias,
                    weights=weights,
                    previous=current_transformer
                )
            else:
                raise Exception(f"Unknow type of layer: {type(layer)}")

        self.transformers.append(current_transformer)

    def verify(self):
        return self.verifier(self.inputs)


class DeepPolyInputsTransformer(torch.nn.Module):
    eps: torch.float
    bounds: torch.Tensor

    def __init__(self, eps: torch.float) -> None:
        super(DeepPolyInputsTransformer, self).__init__()
        self.eps = eps

    def forward(self, inputs: torch.Tensor):
        self.bounds = inputs.repeat(2, 1, 1, 1)

        # Add epsilon around the input bounds
        self.bounds += torch.Tensor([[[[-self.eps]]], [[[self.eps]]]])

        # Clamp them back to intersect with [0, 1]^n
        self.bounds = torch.clamp(self.bounds, 0, 1)

        return self.bounds


class DeepPolyNormalizationTransformer(torch.nn.Module):
    previous: torch.nn.Module
    std: torch.Tensor
    mean: torch.Tensor
    bounds: torch.Tensor

    def __init__(self, previous: torch.nn.Module) -> None:
        super(DeepPolyNormalizationTransformer, self).__init__()
        self.previous = previous

        self.std = torch.Tensor([0.3081])
        self.mean = torch.Tensor([0.1307])

    def forward(self, bounds: torch.Tensor):
        self.bounds = torch.div(bounds - self.mean, self.std)
        return self.bounds


class DeepPolyFlattenTransformer(torch.nn.Module):
    previous: torch.nn.Module
    bounds: torch.Tensor

    def __init__(self, previous: torch.nn.Module) -> None:
        super(DeepPolyFlattenTransformer, self).__init__()
        self.previous = previous

    def forward(self, bounds: torch.Tensor):
        return torch.stack(
            tensors=[
                bounds[0, :, :, :].flatten(),
                bounds[1, :, :, :].flatten()
            ],
            dim=1
        )

    def _back_substitution_helper(self, max_steps: int, parameters: Tuple[torch.Tensor, ...] = None):
        self.bounds = self.last._back_substitution(
            max_steps=max_steps, parameters=parameters)
        slicing_idx = int(len(self.bounds) / 2.0)

        self.bounds = torch.stack(
            tensors=[
                self.bounds[: slicing_idx],
                self.bounds[slicing_idx:]
            ],
            dim=1
        )

        return self.bounds


class DeepPolyAffineTransformer(torch.nn.Module):
    previous: torch.nn.Module
    bias: torch.Tensor
    weights: torch.Tensor
    w_max: torch.Tensor
    w_min: torch.Tensor
    bounds: torch.Tensor
    bs_steps: int

    def __init__(self, bias: torch.Tensor = None, weights: torch.Tensor = None, previous: torch.nn.Module = None, bs_steps: int = 0) -> None:
        super(DeepPolyAffineTransformer, self).__init__()

        self.previous = previous

        self.bias = bias
        self.weights = weights

        self.w_max = torch.clamp(self.weights, min=0.0)
        self.w_min = torch.clamp(self.weights, max=0.0)

        # Backsubsitution steps
        self.bs_steps = bs_steps

    def _back_sub_helper(self, max_steps: int, parameters: Tuple[torch.Tensor, ...] = None):
        if parameters:
            assert(len(parameters) == 4)
            w_l, w_u, b_l, b_u = parameters
        else:
            w_l = self.weights
            w_u = self.weights
            b_l = self.bias
            b_u = self.bias

        clamped_w_l_min, clamped_w_l_max = torch.clamp(
            w_l, min=0), torch.clamp(w_l, max=0)
        clamped_w_u_min, clamped_w_u_max = torch.clamp(
            w_u, min=0), torch.clamp(w_u, max=0)

        if max_steps == 0 or not self.previous.previous.previous:
            l_b = torch.matmul(clamped_w_l_min, self.last.bound[:, 0]) + torch.matmul(
                clamped_w_l_max, self.last.bound[:, 1])

            u_b = torch.matmul(clamped_w_u_min, self.last.bound[:, 1]) + torch.matmul(
                clamped_w_u_max, self.last.bound[:, 0])

            return torch.stack([l_b, u_b], dim=1)

        else:
            # TODO: Deal with case after SPU transformer
            updated_w_l = w_l
            updated_w_u = w_u
            updated_b_l = b_l
            updated_b_u = b_u

            return self.previous._back_substitution_helper(parameters=(updated_w_l, updated_w_u, updated_b_l, updated_b_u), max_steps=max_steps - 1)

    def back_substitution(self, max_steps: int):
        updated_bounds = self._back_sub_helper(max_steps=max_steps)

        lower_bound_indices = self.bounds[:, 0] < updated_bounds[:, 0]
        upper_bound_indices = self.bounds[:, 1] > updated_bounds[:, 1]

        # Update bounds with computed indices
        self.bounds[lower_bound_indices,
                    0] = updated_bounds[lower_bound_indices, 0]
        self.bounds[upper_bound_indices,
                    1] = updated_bounds[upper_bound_indices, 1]

    def forward(self, bounds: torch.Tensor) -> torch.Tensor:
        l_b = torch.matmul(
            self.w_max, bounds[:, 0]) + torch.matmul(self.w_min, bounds[:, 1])
        u_b = torch.matmul(
            self.w_max, bounds[:, 1]) + torch.matmul(self.w_min, bounds[:, 0])

        self.bounds = torch.stack([l_b, u_b], dim=1) + self.bias.reshape(-1,
                                                                         1) if self.bias else torch.stack([l_b, u_b], dim=1)

        if self.bs_steps > 0:
            self.back_substitution(self.bs_steps)

        return self.bounds


class DeepPolySPUTransformer(torch.nn.Module):
    previous: torch.nn.Module
    derivative_SPU: torch.nn.Module
    compute_SPU: torch.nn.Module
    bounds: torch.Tensor
    zero_SPU: torch.Tensor

    def __init__(self, previous: torch.nn.Module = None) -> None:
        super(DeepPolySPUTransformer, self).__init__()
        self.previous = previous

        self.derivative_spu = DerivativeSPU()
        self.compute_SPU = SPU()
        self.zero_SPU = torch.sqrt(torch.abs(SPU(torch.tensor(0))))

    def forward(self, bounds: torch.Tensor):
        return self._compute_linear_bounds(bounds)

    def _back_substitution_helper(self):
        pass

    def _compute_linear_bounds(self, bounds: torch.Tensor):
        self.bounds = bounds

        """
        Generate indices for the 5 cases
        """

        # 1st case: upper <= 0, lower <= 0
        idx_1 = self.bounds[:, 1] <= 0

        # 2nd case: lower >= 0
        idx_2 = self.bounds[:, 0] >= 0

        # 3rd case: upper >= zero_SPU
        idx_3 = self.bounds[:, 1] >= self.zero_SPU

        # all the remaining indices
        tmp_idx_4 = torch.logical_and(
            torch.logical_and(
                torch.logical_not(idx_1),
                torch.logical_not(idx_2)
            ),
            torch.logical_not(idx_3),
        )

        # 4th case: the other cases that verify with SPU(l) > SPU(u)
        idx_4 = torch.logical_and(
            tmp_idx_4,
            self.compute_SPU(self.bounds[:, 0]) > self.compute_SPU(
                self.bounds[:, 1])
        )

        # 5th case: all the other cases
        idx_5 = torch.logical_and(tmp_idx_4, torch.logical_not(idx_4))

        # Initiate the weights and bias for all the inputs
        self.bound_weights = torch.zeros(self.bounds.shape[0], 2)
        self.bound_bias = torch.zeros(self.bounds.shape[0], 2)

        """
        1st case: u <= 0
        """
        # w_l = (self.compute_SPU(u) - self.compute_SPU(l)) / (u - l)
        self.bound_weights[idx_1, 0] = (self.compute_SPU(self.bounds[idx_1, 1]) - self.compute_SPU(
            self.bounds[idx_1, 0])) / (self.bounds[idx_1, 1] - self.bounds[idx_1, 0])

        # b_l = SPU(l) - w_l * l
        self.bound_bias[idx_1, 0] = self.compute_SPU(
            self.bounds[idx_1, 0]) - self.bound_weights[idx_1, 0] * self.bounds[idx_1, 0]

        # w_u = 0 (already the case as initiated)

        # b_u = SPU(l)
        self.bound_bias[idx_1, 1] = self.compute_SPU(self.bounds[idx_1, 0])

        """
        2nd case: l >= 0
        """
        # Optimal point(s)
        # a = (u + l) / 2
        optimal_points_2 = (self.bounds[idx_2, 1] + self.bounds[idx_2, 0]) / 2

        # w_l = self.derivative_SPU(a)
        self.bound_weights[idx_2, 0] = self.derivative_SPU(optimal_points_2)

        # b_l = self.compute_SPU(a) - a * self.derivative_SPU(a)
        self.bound_bias[idx_2, 0] = self.compute_SPU(
            optimal_points_2) - optimal_points_2 * self.derivative_SPU(optimal_points_2)

        # w_u = u + l
        self.bound_weights[idx_2, 1] = self.bounds[idx_2,
                                                   1] + self.bounds[idx_2, 0]
        # b_u = -u * l - self.compute_SPU(0)
        self.bound_bias[idx_2, 1] = -self.bounds[idx_2, 1] * \
            self.bounds[idx_2, 0] - self.compute_SPU(0)

        """
        3rd case: u >= SPU^(-1)(0)
        """
        # w_u = (self.compute_SPU(u) - self.compute_SPU(l)) / (u - l)
        self.bound_weights[idx_3, 1] = (self.compute_SPU(self.bounds[idx_3, 1]) - self.compute_SPU(
            self.bounds[idx_3, 0])) / (self.bounds[idx_3, 1] - self.bounds[idx_3, 0])

        # b_u = self.compute_SPU(l) - w_u * l
        self.bound_bias[idx_3, 1] = self.compute_SPU(
            self.bounds[idx_3, 0]) - self.bound_weights[idx_3, 1] * self.bounds[idx_3, 0]

        # Optimal point
        # a = max((u + l) / 2, SPU_equals_to_zero)
        optimal_points_3 = torch.max(
            (self.bounds[idx_3, 1] + self.bounds[idx_3, 0]) / 2, self.zero_SPU)

        # w_l = self.derivative_SPU(a)
        self.bound_weights[idx_3, 0] = self.derivative_SPU(optimal_points_3)
        # b_l = self.compute_SPU(a) - a * self.derivative_SPU(a)
        self.bound_bias[idx_3, 0] = self.compute_SPU(
            optimal_points_3) - optimal_points_3 * self.derivative_SPU(optimal_points_3)

        """
        4th case: all the other indices that verify SPU(l) > SPU(u)
        """

        # w_u = 0 (already the case)

        # b_u = max(self.compute_SPU(l), self.compute_SPU(u))
        self.bound_bias[idx_4, 1] = torch.max(
            self.compute_SPU(self.bounds[idx_4, 0]), self.compute_SPU(self.bounds[idx_4, 1]))

        # w_l = 0.0 (already the case)

        # b_l = self.compute_SPU(0)
        self.bound_bias[idx_4, 0] = self.compute_SPU(0)

        """
        5th case: all the other indices
        """

        # w_u = (self.compute_SPU(u) - self.compute_SPU(l)) / (u - l)
        self.bound_weights[idx_5, 1] = (self.compute_SPU(self.bounds[idx_5, 1]) - self.compute_SPU(
            self.bounds[idx_5, 0])) / (self.bounds[idx_5, 1] - self.bounds[idx_5, 0])
        # b_u = self.compute_SPU(l) - w_u * l
        self.bound_bias[idx_5, 1] = self.compute_SPU(
            self.bounds[idx_5, 0]) - self.bound_weights[idx_5, 1] * self.bounds[idx_5, 0]

        # w_l = 0.0 (already the case)

        # b_l = self.compute_SPU(0)
        self.bound_bias[idx_5, 0] = self.compute_SPU(0)

        """
        Done generating weights and bias for all the indices.

        Compute bound transformation
        """
        self.bounds = self.bound_weights * self.bounds + self.bound_bias

        return self.bounds
