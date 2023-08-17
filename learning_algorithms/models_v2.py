# Base code from https://github.com/pytorch/rl/blob/dbc7ba0293d2d3321f4285ae2f21ee2c7d5c6dcd/torchrl/modules/models/models.py

from numbers import Number
from typing import List, Optional, Sequence, Tuple, Type, Union
import copy

import numpy as np
import torch
from torch import nn
# from torch.nn import functional as F

from models_utils import DEVICE_TYPING, SquashDims, Squeeze2dLayer, _find_depth



class MLP(nn.Sequential):

    def __init__(
        self,
        in_features: Optional[int] = None,
        out_features: Union[int, Sequence[int]] = None,
        depth: Optional[int] = None,
        num_cells: Optional[Union[Sequence, int]] = None,
        activation_class: Type = nn.Tanh,
        activation_kwargs: Optional[dict] = None,
        last_activation: Optional[Type] = None,
        last_activation_kwargs: Optional[dict] = None,
        norm_class: Optional[Type] = None,
        norm_kwargs: Optional[dict] = None,
        bias_last_layer: bool = True,
        single_bias_last_layer: bool = False,
        layer_class: Type = nn.Linear,
        layer_kwargs: Optional[dict] = None,
        activate_last_layer: bool = False,
        device: DEVICE_TYPING = "cpu",
    ):
        if out_features is None:
            raise ValueError("out_feature must be specified for MLP.")

        default_num_cells = 32
        if num_cells is None:
            if depth is None:
                num_cells = [default_num_cells] * 3
                depth = 3
            else:
                num_cells = [default_num_cells] * depth

        self.in_features = in_features

        _out_features_num = out_features
        if not isinstance(out_features, Number):
            _out_features_num = np.prod(out_features)
        self.out_features = out_features
        self._out_features_num = _out_features_num
        self.activation_class = activation_class
        self.activation_kwargs = (
            activation_kwargs if activation_kwargs is not None else dict()
        )
        self.last_activation = last_activation
        self.last_activation_kwargs = (
            last_activation_kwargs if last_activation_kwargs is not None else dict()
        )
        self.norm_class = norm_class
        self.device = torch.device(device)
        self.norm_kwargs = norm_kwargs if norm_kwargs is not None else dict()
        self.bias_last_layer = bias_last_layer
        self.single_bias_last_layer = single_bias_last_layer
        self.layer_class = layer_class
        self.layer_kwargs = layer_kwargs if layer_kwargs is not None else dict()
        self.layer_kwargs.update({"device": self.device})
        self.activate_last_layer = activate_last_layer
        if single_bias_last_layer:
            raise NotImplementedError

        if not (isinstance(num_cells, Sequence) or depth is not None):
            raise RuntimeError(
                "If num_cells is provided as an integer, \
            depth must be provided too."
            )
        self.num_cells = (
            list(num_cells) if isinstance(num_cells, Sequence) else [num_cells] * depth
        )
        self.depth = depth if depth is not None else len(self.num_cells)
        if not (len(self.num_cells) == depth or depth is None):
            raise RuntimeError(
                "depth and num_cells length conflict, \
            consider matching or specifying a constan num_cells argument together with a a desired depth"
            )
        layers = self._make_net()
        super().__init__(*layers)

    def _make_net(self) -> List[nn.Module]:
        layers = []
        in_features = [self.in_features] + self.num_cells
        out_features = self.num_cells + [self._out_features_num]
        for i, (_in, _out) in enumerate(zip(in_features, out_features)):
            _bias = self.bias_last_layer if i == self.depth else True
            if _in is not None:
                layers.append(
                    self.layer_class(_in, _out, bias=_bias, **self.layer_kwargs)
                )
            else:
                raise Exception('Lazy layers not implemented.')
                # try:
                #     lazy_version = LazyMapping[self.layer_class]
                # except KeyError:
                #     raise KeyError(
                #         f"The lazy version of {self.layer_class.__name__} is not implemented yet. "
                #         "Consider providing the input feature dimensions explicitely when creating an MLP module"
                #     )
                # layers.append(lazy_version(_out, bias=_bias, **self.layer_kwargs))

            if i < self.depth or self.last_activation is None and self.activate_last_layer:
                layers.append(self.activation_class(**self.activation_kwargs))
                if self.norm_class is not None:
                    layers.append(self.norm_class(**self.norm_kwargs))
            elif self.last_activation is not None:
                layers.append(self.last_activation(**self.last_activation_kwargs))
                if self.norm_class is not None:
                    layers.append(self.norm_class(**self.norm_kwargs))
        return layers

    def forward(self, *inputs: Tuple[torch.Tensor]) -> torch.Tensor:
        if len(inputs) > 1:
            inputs = (torch.cat([*inputs], -1),)

        out = super().forward(*inputs)
        if not isinstance(self.out_features, Number):
            out = out.view(*out.shape[:-1], *self.out_features)
        return out
    
          



class ConvNet(nn.Sequential):

    def __init__(
        self,
        in_features: Optional[int] = None,
        depth: Optional[int] = None,
        num_cells: Union[Sequence, int] = [32, 32, 32],
        kernel_sizes: Union[Sequence[Union[int, Sequence[int]]], int] = 3,
        strides: Union[Sequence, int] = 1,
        paddings: Union[Sequence, int] = 0,
        activation_class: Type = nn.ELU,
        activation_kwargs: Optional[dict] = None,
        norm_class: Type = None,
        norm_kwargs: Optional[dict] = None,
        conv_dim: Optional[int] = 3,
        bias_last_layer: bool = True,
        aggregator_class: Type = SquashDims,
        aggregator_kwargs: Optional[dict] = None,
        squeeze_output: bool = False,
        device: DEVICE_TYPING = "cpu",
    ):

        self.in_features = in_features
        self.activation_class = activation_class
        self.activation_kwargs = (
            activation_kwargs if activation_kwargs is not None else dict()
        )
        self.norm_class = norm_class
        self.norm_kwargs = norm_kwargs if norm_kwargs is not None else dict()
        self.bias_last_layer = bias_last_layer

        self.conv_dim = conv_dim
        if self.conv_dim == 2:
            self.conv_layer = nn.Conv2d
            self.conv_layer_lazy = nn.LazyConv2d
            dims2flatten = 3
        elif self.conv_dim == 3:
            self.conv_layer = nn.Conv3d
            self.conv_layer_lazy = nn.LazyConv3d
            dims2flatten = 4
        else:
            raise Exception('Convolutional layer dimensionality not implemented.')
            
        self.aggregator_class = aggregator_class
        self.aggregator_kwargs = (
            aggregator_kwargs if aggregator_kwargs is not None else {"ndims_in": dims2flatten}
        )
        
        self.squeeze_output = squeeze_output
        self.device = torch.device(device)
        # self.single_bias_last_layer = single_bias_last_layer

        depth = _find_depth(depth, num_cells, kernel_sizes, strides, paddings)
        self.depth = depth
        if depth == 0:
            raise ValueError("Null depth is not permitted with ConvNet.")

        for _field, _value in zip(
            ["num_cells", "kernel_sizes", "strides", "paddings"],
            [num_cells, kernel_sizes, strides, paddings],
        ):
            _depth = depth
            setattr(
                self,
                _field,
                (_value if isinstance(_value, Sequence) else [_value] * _depth),
            )
            if not (isinstance(_value, Sequence) or _depth is not None):
                raise RuntimeError(
                    f"If {_field} is provided as an integer, "
                    "depth must be provided too."
                )
            if not (len(getattr(self, _field)) == _depth or _depth is None):
                raise RuntimeError(
                    f"depth={depth} and {_field}={len(getattr(self, _field))} length conflict, "
                    + f"consider matching or specifying a constan {_field} argument together with a a desired depth"
                )

        self.out_features = self.num_cells[-1]

        self.depth = len(self.kernel_sizes)
        layers = self._make_net()
        super().__init__(*layers)

    def _make_net(self) -> nn.Module:
        layers = []
        in_features = [self.in_features] + self.num_cells[: self.depth]
        out_features = self.num_cells + [self.out_features]
        kernel_sizes = self.kernel_sizes
        strides = self.strides
        paddings = self.paddings
        
        for i, (_in, _out, _kernel, _stride, _padding) in enumerate(
            zip(in_features, out_features, kernel_sizes, strides, paddings)
        ):
            _bias = (i < len(in_features) - 1) or self.bias_last_layer
            if _in is not None:
                layers.append(
                    self.conv_layer(
                        _in,
                        _out,
                        kernel_size=_kernel,
                        stride=_stride,
                        bias=_bias,
                        padding=_padding,
                        device=self.device,
                    )
                )
            else:
                layers.append(
                    self.conv_layer_lazy(
                        _out,
                        kernel_size=_kernel,
                        stride=_stride,
                        bias=_bias,
                        padding=_padding,
                        device=self.device,
                    )
                )

            layers.append(self.activation_class(**self.activation_kwargs))
            if self.norm_class is not None:
                layers.append(self.norm_class(**self.norm_kwargs))

        if self.aggregator_class is not None:
            layers.append(self.aggregator_class(**self.aggregator_kwargs))

        if self.squeeze_output:
            layers.append(Squeeze2dLayer())
        return layers


    def get_output_len(self, in_size):
        if self.norm_class is not None:
            norm_kernel_size = self.norm_kwargs.get('kernel_size')
            norm_padding = self.norm_kwargs.get('padding', 0)
            norm_stride = self.norm_kwargs.get('stride', 1)
            
        if not hasattr(in_size, '__len__'):
            curr_size = [in_size for _ in range(self.conv_dim)]
        else:
            curr_size = copy.deepcopy(in_size)
       
        for i in range(self.depth):
            for j in range(len(curr_size)):
                curr_size[j] = np.floor((curr_size[j] + 2*self.paddings[i] - self.kernel_sizes[i])/self.strides[i] + 1)
                if self.norm_class is not None:
                    curr_size[j] = np.floor((curr_size[j] + 2*norm_padding - norm_kernel_size)/norm_stride + 1)   
        total_size = self.num_cells[-1] * np.cumprod(np.array(curr_size))[-1]
        return int(total_size)

    

class BuildConvActorCritic(nn.Module):
    def __init__(self, conv_net, mlp_kwargs, context_dim, action_dim=0, mf_actions_dim=0):
        super(BuildConvActorCritic, self).__init__()
        self.conv_net = conv_net
        out_conv_net_dim = self.conv_net.get_output_len(context_dim)
        mlp_input_dim = out_conv_net_dim + action_dim + mf_actions_dim
        
        mlp_kwargs.update({'in_features' : mlp_input_dim})
        self.mlp = MLP(**mlp_kwargs)
        
    def forward(self, *inputs: Tuple[torch.Tensor]) -> torch.Tensor:
        assert len(inputs) <= 3
        if len(inputs) == 3: # MF Critic
            context, action, mf_actions = inputs
            context = context.unsqueeze(1)
            hidden = self.conv_net.forward(context)
            hidden = torch.cat([hidden, action, mf_actions], -1)
        elif len(inputs) == 2: # Critic
            context, action = inputs
            context = context.unsqueeze(1)
            hidden = self.conv_net.forward(context)
            hidden = torch.cat([hidden, action], -1)
        else:  # Actor
            context = inputs[0].unsqueeze(1)
            hidden = self.conv_net.forward(context)
        hidden = hidden.unsqueeze(0)
        output = self.mlp.forward(hidden)
        output = output.squeeze(0)
        return output
            


                 


                 

                 

                 

                 

                 
