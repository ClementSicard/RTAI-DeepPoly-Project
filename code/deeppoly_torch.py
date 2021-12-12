from networks import Normalization
import torch
from networks import SPU, DerivativeSPU
from typing import List, Tuple
from utils import pprint


class DeepPolyLinearTransformer(torch.nn.Module):
    layer: torch.nn.Module

    def __init__(
        self,
        input_size: int,
        output_size: int,
    ) -> None:
        super().__init__()
        self.layer = torch.nn.Linear(
            input_size, output_size).requires_grad_(False)

    @staticmethod
    def transform_domain(
        weights: torch.Tensor,
        bias: torch.Tensor,
        lower_bounds: torch.Tensor,
        upper_bounds: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Implements swapping of lower and higher bounds
        where the weights are negative and computes the
        forward pass of box bounds. """

        negative_mask = (weights < 0).int()
        positive_mask = (weights >= 0).int()

        negative_weights = torch.mul(negative_mask, weights)
        positive_weights = torch.mul(positive_mask, weights)

        new_lower_bounds = torch.matmul(upper_bounds, negative_weights.t(
        )) + torch.matmul(lower_bounds, positive_weights.t()) + bias

        new_upper_bounds = torch.matmul(lower_bounds, negative_weights.t(
        )) + torch.matmul(upper_bounds, positive_weights.t()) + bias

        # quick check here
        assert (new_lower_bounds <= new_upper_bounds).all(
        ), "Error with the box bounds: low>high"

        return new_lower_bounds, new_upper_bounds

    def forward(
        self,
        x: torch.Tensor,
        lower_bounds: torch.Tensor,
        upper_bounds: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Specific attention must be payed to the transformation
        of box bounds. When working with negative weights low and
        high bound must be swapped.
        """
        weights = self.layer.weight
        bias = self.layer.bias

        lower_bounds, upper_bounds = self.transform_domain(
            weights=weights,
            bias=bias,
            lower_bounds=lower_bounds,
            upper_bounds=upper_bounds,
        )

        out = self.layer(x)

        return out, lower_bounds, upper_bounds


class DeepPolySPUTransformer(torch.nn.Module):
    spu: torch.torch.nn.Module
    derivative_spu: torch.torch.nn.Module
    size: int
    spu_zero: torch.Tensor
    lower_weights: torch.Tensor
    lower_bias: torch.Tensor
    upper_weights: torch.Tensor
    upper_bias: torch.Tensor

    def __init__(
        self,
        input_size: int
    ) -> None:
        super().__init__()
        self.spu = SPU()
        self.derivative_spu = DerivativeSPU()
        self.size = input_size
        self.spu_zero = torch.sqrt(torch.abs(self.spu(torch.tensor(0))))

    def forward(
        self,
        x: torch.Tensor,
        lower_bounds: torch.Tensor,
        upper_bounds: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        input_size = x.size()[0]

        self.lower_weights = torch.eye(input_size)
        self.upper_weights = torch.eye(input_size)

        self.lower_bias = torch.zeros(input_size)
        self.upper_bias = torch.zeros(input_size)

        for i in range(input_size):

            l = lower_bounds[i]
            u = upper_bounds[i]

            # All the points are in the negative half-plane
            if u <= 0:
                w_u = 0
                b_u = self.spu(l)

                w_l = (self.spu(u) - self.spu(l)) / (u - l)
                b_l = self.spu(l) - w_l * l

            elif l >= 0:
                w_u = u + l
                b_u = -u * l - self.spu(0)

                # Optimal point
                a = (u + l) / 2

                w_l = self.derivative_spu(a)
                b_l = self.spu(a) - a * self.derivative_spu(a)

            else:
                if u >= self.spu_zero:
                    w_u = (self.spu(u) - self.spu(l)) / (u - l)
                    b_u = self.spu(l) - w_u * l

                    # Optimal point
                    a = torch.max((u + l) / 2, self.spu_zero)

                    w_l = self.derivative_spu(a)
                    b_l = self.spu(a) - a * self.derivative_spu(a)
                else:
                    if self.spu(l) > self.spu(u):
                        w_u = 0
                        b_u = torch.max(self.spu(l), self.spu(u))

                    else:
                        w_u = (self.spu(u) - self.spu(l)) / (u - l)
                        b_u = self.spu(l) - w_u * l

                    w_l = 0.0
                    b_l = self.spu(0)

            self.lower_weights[i, i] = w_l
            self.lower_bias[i] = b_l

            self.upper_weights[i, i] = w_u
            self.upper_bias[i] = b_u

        out = self.spu(x)

        new_lower_bounds = torch.matmul(
            self.lower_weights, lower_bounds) + self.lower_bias

        new_upper_bounds = torch.matmul(
            self.upper_weights, upper_bounds) + self.upper_bias

        return out, new_lower_bounds, new_upper_bounds


class DeepPolyVerifierTorch(torch.nn.Module):
    """ Abstract version of fully connected network """
    layers: torch.nn.Module
    lower_bounds: torch.Tensor
    upper_bounds: torch.Tensor

    verbose: bool

    def __init__(
        self,
        device: str,
        input_size: int,
        fc_layers: List[int],
        verbose: bool,
    ) -> None:
        super().__init__()
        self.verbose = verbose
        self.device = device

        layers = [Normalization(self.device),
                  torch.nn.Flatten()]

        input_size = input_size * input_size

        for i, output_size in enumerate(fc_layers, 1):
            with torch.no_grad():
                layers.append(DeepPolyLinearTransformer(
                    input_size=input_size, output_size=output_size).to(self.device))

            if i < len(fc_layers):
                layers.append(DeepPolySPUTransformer(
                    input_size=output_size).to(self.device))

            input_size = output_size

        self.layers = torch.nn.Sequential(*layers)

    def forward(
        self,
        x: torch.Tensor,
        lower_bounds: torch.Tensor,
        upper_bounds: torch.Tensor
    ):
        """
            Propagation of abstract area through the network.
            Parameters:
            - x: input
            - low: lower bound on input perturbation (epsilon)
            - high: upper bound on //   //
            note: all the input tensors have shape (1,1,28,28)
            """
        # Normalize
        x = self.layers[0](x)

        # Flattening and squeezing
        x = self.layers[1](x).squeeze()

        lower_bounds = self.layers[0](lower_bounds)
        upper_bounds = self.layers[0](upper_bounds)

        lower_bounds = self.layers[1](lower_bounds).squeeze()
        upper_bounds = self.layers[1](upper_bounds).squeeze()

        self.lower_bounds = [lower_bounds]
        self.upper_bounds = [upper_bounds]

        self.activations = [x]

        # now the rest of the layers
        for i, layer in enumerate(self.layers):
            if self.verbose:
                print(f"\nCurrent layer: {layer}")

            if i in {0, 1}:
                if self.verbose:
                    pprint(f"--> Skipping layer")
                continue

            x, lower_bounds, upper_bounds = layer(
                x, lower_bounds, upper_bounds)

            if type(layer) == DeepPolyLinearTransformer:
                order = i - 2
                while i - order >= 2:
                    lower_bound_tmp, upper_bound_tmp = self.backsubstitute_down_to(
                        layer_index=i,
                        input_size=x.size()[0],
                        order=order,
                    )

                    lower_bounds = torch.maximum(lower_bounds, lower_bound_tmp)
                    upper_bounds = torch.minimum(upper_bounds, upper_bound_tmp)

                    order += 2

            self.lower_bounds += [lower_bounds]
            self.upper_bounds += [upper_bounds]
            self.activations += [x]

        return x, lower_bounds, upper_bounds

    def load_weights(self, net: torch.nn.Module) -> None:
        for i, layer in enumerate(net.layers):
            if type(layer) == torch.nn.Linear:
                self.layers[i].layer.weight = layer.weight
                self.layers[i].layer.weight.requires_grad_(False)
                self.layers[i].layer.bias = layer.bias
                self.layers[i].layer.bias.requires_grad_(False)

    def backsubstitute_down_to(
        self,
        layer_index: int,
        input_size: int,
        order: int = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """ 
        Implements backsubstitution down to the layer at `layer_index`
        """
        if order == None:
            order = len(self.layers) - 2

        l = self.lower_bounds[max(0, layer_index - order - 2)]
        u = self.upper_bounds[max(0, layer_index - order - 2)]

        lower_weights = torch.eye(input_size)
        lower_bias = torch.zeros(input_size)

        upper_weights = torch.eye(input_size)
        upper_bias = torch.zeros(input_size)

        for layer in self.layers[max(
                layer_index - order, 2): layer_index + 1][::-1]:

            # TEST
            upper_weights_tmp = layer.upper_weights if isinstance(
                layer, DeepPolySPUTransformer) else layer.layer.weight
            upper_bias_tmp = layer.upper_bias if isinstance(
                layer, DeepPolySPUTransformer) else layer.layer.bias
            lower_weights_tmp = layer.lower_weights if isinstance(
                layer, DeepPolySPUTransformer) else layer.layer.weight
            lower_bias_tmp = layer.lower_bias if isinstance(
                layer, DeepPolySPUTransformer) else layer.layer.bias

            upper_bias += torch.matmul(upper_weights, upper_bias_tmp)
            lower_bias += torch.matmul(lower_weights, lower_bias_tmp)
            upper_weights = torch.matmul(upper_weights, upper_weights_tmp)
            lower_weights = torch.matmul(lower_weights, lower_weights_tmp)

            # if isinstance(layer, DeepPolyLinearTransformer):
            #     upper_weights_tmp = layer.layer.weight
            #     upper_bias_tmp = layer.layer.bias
            #     lower_weights_tmp = layer.layer.weight
            #     lower_bias_tmp = layer.layer.bias
            #     upper_bias += torch.matmul(upper_weights, upper_bias_tmp)
            #     lower_bias += torch.matmul(lower_weights, lower_bias_tmp)
            #     upper_weights = torch.matmul(upper_weights, upper_weights_tmp)
            #     lower_weights = torch.matmul(lower_weights, lower_weights_tmp)

            # elif isinstance(layer, DeepPolySPUTransformer):
            #     lower_weights_tmp = layer.lower_weights
            #     upper_weights_tmp = layer.upper_weights
            #     upper_bias_tmp = layer.upper_bias

            #     lower_weights, lower_bias_diff = self._backsubstitute_spu(
            #         back_sub_matrix=lower_weights,
            #         upper_weights_spu=upper_weights_tmp,
            #         lower_weights_spu=lower_weights_tmp,
            #         upper_bias=upper_bias_tmp,
            #     )

            #     upper_weights, upper_bias_diff = self._backsubstitute_spu(
            #         back_sub_matrix=upper_weights,
            #         upper_weights_spu=upper_weights_tmp,
            #         lower_weights_spu=lower_weights_tmp,
            #         upper_bias=upper_bias_tmp,
            #         lower_bool=False
            #     )

            #     upper_bias += upper_bias_diff
            #     lower_bias += lower_bias_diff
            # else:
            #     raise Exception("Unknown layer in the forward pass ")

        # finally computing the forward pass on the input ranges
        # note: no bias here (all the biases were already included in W)
        lower_bounds, _ = DeepPolyLinearTransformer.transform_domain(
            lower_weights, lower_bias, l, u)
        _, upper_bounds = DeepPolyLinearTransformer.transform_domain(
            upper_weights, upper_bias, l, u)

        return lower_bounds, upper_bounds

    def _backsubstitute_spu(
        self,
        back_sub_matrix: torch.Tensor,
        lower_weights_spu: torch.Tensor,
        upper_weights_spu: torch.Tensor,
        upper_bias: torch.Tensor,
        lower_bool=True,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Computes matrix multiplication for backsubstitution
        when passing through a SPU layer"""

        input_dimension = back_sub_matrix.size()[0]

        # initialise everything to 0
        weights_matrix = torch.zeros_like(back_sub_matrix)
        bias = torch.zeros(input_dimension)

        # now we want to go into each entry in the back_sub matrix
        # and multiply it by the respective spu weight

        for i in range(input_dimension):  # for each column of the matrix
            if upper_weights_spu[i, i] == 0:
                continue

            current_column = back_sub_matrix[:, i]

            lower = torch.ones_like(current_column) * lower_weights_spu[i, i]
            upper = torch.ones_like(
                current_column) * upper_weights_spu[i, i]

            negative_mask = (current_column < 0).int()
            positive_mask = (current_column >= 0).int()

            if lower_bool:
                tmp = torch.mul(
                    negative_mask, upper) + torch.mul(positive_mask, lower)
                bias += current_column * negative_mask * upper_bias[i]

            else:
                tmp = torch.mul(
                    negative_mask, lower) + torch.mul(positive_mask, upper)
                bias += current_column * positive_mask * upper_bias[i]

            weights_matrix[:, i] = current_column * tmp

        return weights_matrix, bias

    def backsubstitute(
        self,
        true_label: int,
        order: int = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """ Implements backsubstitution
        true_label (int): index (0 to 9) of the right label - used in the last step of backsubstitution
        order (int): defines number of layers to backsubstitute starting from the output.
        """
        if order is None:
            # example: 10 layers, 9 actual lows and highs, 1 for the input, 8 for the rest of the layers
            order = len(self.activations)
        l = self.lower_bounds[-order]
        u = self.upper_bounds[-order]

        nb_of_classes = 10  # we will start from the output
        upper_bias = torch.zeros(nb_of_classes - 1)
        lower_bias = torch.zeros(nb_of_classes - 1)

        # First, we insert the affine layer corresponding to the substractions
        # employed by the verifier to check the correctness of the prediction
        # output_j = logit_i - logit_j, where i is the true_label
        substract_weights = -torch.eye(nb_of_classes - 1)
        substract_weights = torch.cat(
            [
                substract_weights[:, 0: true_label],
                torch.ones(nb_of_classes - 1, 1),
                substract_weights[:, true_label: nb_of_classes]
            ],
            dim=1
        )  # inserting the column of ones for the true label
        # now cumulating the last operation

        lower_weights = substract_weights.clone()
        upper_weights = substract_weights.clone()

        # order = layers -1 --> order -1 = layers -2 --> skipping first two layers
        for layer in self.layers[1 - order:][::-1]:
            if isinstance(layer, DeepPolySPUTransformer):
                lower_weights_tmp = layer.lower_weights
                upper_weights_tmp = layer.upper_weights

                upper_bias_tmp = layer.upper_bias

                lower_weights, lower_bias_delta = self._backsubstitute_spu(
                    back_sub_matrix=lower_weights,
                    lower_weights_spu=lower_weights_tmp,
                    upper_weights_spu=upper_weights_tmp,
                    upper_bias=upper_bias_tmp,
                )

                upper_weights, upper_bias_delta = self._backsubstitute_spu(
                    back_sub_matrix=upper_weights,
                    lower_weights_spu=lower_weights_tmp,
                    upper_weights_spu=upper_weights_tmp,
                    upper_bias=upper_bias_tmp,
                    lower_bool=False
                )

                upper_bias += upper_bias_delta
                lower_bias += lower_bias_delta

            elif isinstance(layer, DeepPolyLinearTransformer):
                upper_weights_tmp = layer.layer.weight
                upper_bias_tmp = layer.layer.bias

                lower_weights_tmp = layer.layer.weight
                lower_bias_tmp = layer.layer.bias

                upper_bias += torch.matmul(upper_weights, upper_bias_tmp)
                lower_bias += torch.matmul(lower_weights, lower_bias_tmp)

                upper_weights = torch.matmul(upper_weights, upper_weights_tmp)
                lower_weights = torch.matmul(lower_weights, lower_weights_tmp)

            else:
                raise Exception("Unknown layer in the forward pass ")

        # finally computing the forward pass on the input ranges
        # note: no bias here (all the biases were already included in W)
        lower_bounds, _ = DeepPolyLinearTransformer.transform_domain(
            lower_weights, lower_bias, l, u)
        _, upper_bounds = DeepPolyLinearTransformer.transform_domain(
            upper_weights, upper_bias, l, u)

        return lower_bounds, upper_bounds
