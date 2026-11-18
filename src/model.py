import math
import copy


import torch
import torch.nn as nn
import torch.nn.functional as F


from abc import ABC, abstractmethod
from typing import Tuple




# ============================================================
# Utility
# ============================================================


class Permute(nn.Module):


   def __init__(self, dims: Tuple[int, ...]):
       super().__init__()
       self.dims = dims


   def forward(self, x):
       return x.permute(self.dims)




def apply_output_activation(layers, clamp):
   """Append the requested output activation."""


   if clamp is None or clamp is False:
       return layers


   elif clamp == "clamp":
       layers.append(nn.Hardtanh(0, 1))


   elif clamp == "sigmoid":
       layers.append(nn.Sigmoid())


   elif clamp == "relu":
       layers.append(nn.ReLU())


   elif clamp == "softplus":
       layers.append(nn.Softplus())


   else:
       raise ValueError(
           f"Unknown output activation: {clamp}"
       )


   return layers




def make_activation(nonlin):
   """
   Create an independent activation module.


   Accepts either an nn.Module instance or an nn.Module class.
   """


   if isinstance(nonlin, str):


       if not hasattr(nn, nonlin):
           raise ValueError(
               f"Unknown activation: {nonlin}"
           )


       activation = getattr(nn, nonlin)


       if not isinstance(activation, type):
           raise ValueError(
               f"{nonlin} is not an activation class."
           )


       return activation()


   if isinstance(nonlin, type) and issubclass(nonlin, nn.Module):
       return nonlin()


   if isinstance(nonlin, nn.Module):
       return copy.deepcopy(nonlin)


   raise TypeError(
       "nonlin must be an nn.Module, nn.Module class, "
       "or activation name."
   )




def validate_image_output(d_out):
   """
   Validate image output dimensions.


   Expected format:
       [width, height, channels]
   """


   if isinstance(d_out, int):
       raise ValueError(
           "d_out must be [width, height, channels]."
       )


   if len(d_out) != 3:
       raise ValueError(
           "d_out must have three elements: "
           "[width, height, channels]."
       )


   width, height, channels = d_out


   if width < 1 or height < 1:
       raise ValueError(
           "Image width and height must be positive."
       )


   if channels not in (1, 3, 4):
       raise ValueError(
           "Output channels must be 1, 3, or 4."
       )


   return int(width), int(height), int(channels)




# ============================================================
# MLP
# ============================================================


class MLP(nn.Sequential):


   def __init__(
       self,
       d_in,
       d_out,
       d_hidden=10,
       n_layers=2,
       nonlin=nn.Tanh(),
       dtype=torch.float,
       init_dict=None,
       unflatten=None,
   ):


       if isinstance(d_out, int):
           d_out = [d_out]


       if d_in < 1 or d_hidden < 1:
           raise ValueError(
               "d_in and d_hidden must be positive."
           )


       if n_layers < 1:
           raise ValueError(
               "n_layers must be at least 1."
           )


       layers = []


       for i in range(n_layers):


           layers.append(
               nn.Linear(
                   d_in if i == 0 else d_hidden,
                   d_hidden if i < n_layers - 1 else d_out[0],
                   dtype=dtype,
               )
           )


           if i < n_layers - 1:
               layers.append(
                   make_activation(nonlin)
               )


       if unflatten is not None:
           layers.append(
               nn.Unflatten(-1, unflatten)
           )


       super().__init__(*layers)


       self.d_in = d_in
       self.d_out = d_out


       if init_dict is not None:


           for name, param in self.named_parameters():


               if name in init_dict:
                   param.data.copy_(init_dict[name])




# ============================================================
# Deconvolutional MLP
# ============================================================


class DeconvMLP(nn.Sequential):


   def __init__(
       self,
       d_in,
       d_out,
       d_hidden=256,
       n_layers=4,
       n_channel=32,
       kernel_size=4,
       stride=2,
       padding=1,
       nonlin=nn.ELU(),
       clamp=None,
       batchnorm=False,
   ):


       width, height, channels = validate_image_output(d_out)


       if d_in < 1 or d_hidden < 1 or n_channel < 1:
           raise ValueError(
               "d_in, d_hidden and n_channel must be positive."
           )


       if n_layers < 1:
           raise ValueError(
               "n_layers must be at least 1."
           )


       layers = [
           nn.Linear(
               d_in,
               d_hidden,
           ),
           nn.Unflatten(
               -1,
               (d_hidden, 1, 1),
           ),
       ]


       in_channels = d_hidden


       for i in range(n_layers):


           if i == n_layers - 1:
               out_channels = channels


           else:
               out_channels = max(
                   n_channel // (2 ** i),
                   channels,
               )


           layers.append(
               nn.ConvTranspose2d(
                   in_channels,
                   out_channels,
                   kernel_size=kernel_size,
                   stride=stride,
                   padding=padding,
               )
           )


           if i < n_layers - 1:


               if batchnorm:
                   layers.append(
                       nn.BatchNorm2d(out_channels)
                   )


               layers.append(
                   make_activation(nonlin)
               )


           in_channels = out_channels


       # Guarantee the requested spatial resolution.
       layers.append(
           nn.Upsample(
               size=(height, width),
               mode="nearest",
           )
       )


       apply_output_activation(
           layers,
           clamp,
       )


       # NCHW -> NHWC
       layers.append(
           Permute((0, 2, 3, 1))
       )


       super().__init__(*layers)


       self.d_in = d_in
       self.d_out = [width, height, channels]




# ============================================================
# Upsampling MLP
# ============================================================


class UpsampleMLP(nn.Sequential):
   """
   Original bilinear upsampling decoder.


   Input:
       [B, d_in]


   Fully connected representation:
       d_in -> d_hidden -> ... -> n_channel * 4 * 4


   Initial feature map:
       [B, n_channel, 4, 4]


   Four bilinear upsampling stages:
       4 -> 8 -> 16 -> 32 -> 64


   Output:
       [B, 64, 64, channels]


   n_layers controls the number of fully connected layers.
   It does not control the number of upsampling stages.
   """


   def __init__(
       self,
       d_in: int,
       d_out: list,
       d_hidden: int = 256,
       n_layers: int = 4,
       n_channel: int = 32,
       kernel_size: int = 2,
       nonlin: nn.Module = nn.ELU(),
       clamp=None,
       batchnorm: bool = False,
       **kwargs,
   ):


       super().__init__()


       width, height, channels = validate_image_output(d_out)


       self.d_in = d_in
       self.d_out = [width, height, channels]


       if d_in < 1:
           raise ValueError(
               "d_in must be positive."
           )


       if d_hidden < 1:
           raise ValueError(
               "d_hidden must be positive."
           )


       if n_channel < 1:
           raise ValueError(
               "n_channel must be positive."
           )


       if n_layers < 1:
           raise ValueError(
               "n_layers must be at least 1."
           )


       if kernel_size < 1:
           raise ValueError(
               "kernel_size must be positive."
           )


       # This original architecture is designed for 64x64 images.
       if width != 64 or height != 64:
           raise ValueError(
               "The original UpsampleMLP architecture starts "
               "from 4x4 and performs four upsampling stages. "
               "It therefore produces 64x64 images. "
               f"Received width={width}, height={height}."
           )


       # ----------------------------------------------------
       # Fully connected layers
       # ----------------------------------------------------


       self.append(
           nn.Linear(
               d_in,
               d_hidden,
           )
       )


       for _ in range(n_layers - 2):


           self.append(
               make_activation(nonlin)
           )


           self.append(
               nn.Linear(
                   d_hidden,
                   d_hidden,
               )
           )


       self.append(
           make_activation(nonlin)
       )


       self.append(
           nn.Linear(
               d_hidden,
               n_channel * 4 * 4,
           )
       )


       # Start from a 4x4 feature map.
       self.append(
           nn.Unflatten(
               1,
               (n_channel, 4, 4),
           )
       )


       # ----------------------------------------------------
       # Three intermediate stages:
       # 4x4 -> 8x8 -> 16x16 -> 32x32
       # ----------------------------------------------------


       for _ in range(3):


           self.append(
               make_activation(nonlin)
           )


           self.append(
               nn.UpsamplingBilinear2d(
                   scale_factor=2,
               )
           )


           if batchnorm:


               self.append(
                   nn.BatchNorm2d(n_channel)
               )


           self.append(
               nn.Conv2d(
                   n_channel,
                   n_channel,
                   kernel_size=kernel_size,
                   padding="same",
               )
           )


       # ----------------------------------------------------
       # Final stage:
       # 32x32 -> 64x64
       # ----------------------------------------------------


       self.append(
           make_activation(nonlin)
       )


       self.append(
           nn.UpsamplingBilinear2d(
               scale_factor=2,
           )
       )


       if batchnorm:


           self.append(
               nn.BatchNorm2d(n_channel)
           )


       self.append(
           nn.Conv2d(
               n_channel,
               channels,
               kernel_size=kernel_size,
               padding="same",
           )
       )


       # ----------------------------------------------------
       # Output activation
       # ----------------------------------------------------


       apply_output_activation(
           self,
           clamp,
       )


       # NCHW -> NHWC
       self.append(
           Permute((0, 2, 3, 1))
       )




# ============================================================
# Parallel slot networks
# ============================================================


class ParallelSlots(nn.Module):


   def __init__(self, slot_functions):


       super().__init__()


       if len(slot_functions) == 0:
           raise ValueError(
               "At least one slot network is required."
           )


       self.slot_functions = nn.ModuleList(
           slot_functions
       )


       self.d_in = [
           slot.d_in
           for slot in slot_functions
       ]


       self.d_out = [
           slot.d_out
           for slot in slot_functions
       ]


   def forward(self, x):


       if x.ndim == 1:
           x = x.unsqueeze(0)


       if x.ndim != 2:
           raise ValueError(
               "ParallelSlots expects [B, D] input. "
               f"Received {tuple(x.shape)}."
           )


       expected_dim = sum(self.d_in)


       if x.shape[-1] != expected_dim:
           raise ValueError(
               f"Expected {expected_dim} latent dimensions, "
               f"received {x.shape[-1]}."
           )


       slot_inputs = torch.split(
           x,
           self.d_in,
           dim=-1,
       )


       return [
           slot_function(slot_input)


           for slot_function, slot_input


           in zip(
               self.slot_functions,
               slot_inputs,
           )
       ]


   def __getitem__(self, idx):
       return self.slot_functions[idx]




# ============================================================
# Composition base class
# ============================================================


class Composition(nn.Module, ABC):


   def __init__(self):
       super().__init__()


   @abstractmethod
   def get_d_out(self, slot_d_out):
       pass




# ============================================================
# Add composition
# ============================================================


class Add(Composition):


   def forward(self, x):


       if not isinstance(x, (list, tuple)):
           raise TypeError(
               "Add expects a list or tuple of slot outputs."
           )


       if len(x) == 0:
           raise ValueError(
               "No slot outputs were provided."
           )


       reference_shape = x[0].shape


       for slot in x[1:]:


           if slot.shape != reference_shape:
               raise ValueError(
                   "All slot outputs must have identical shapes."
               )


       x = torch.stack(
           x,
           dim=1,
       )


       return x.sum(dim=1)


   def get_d_out(self, slot_d_out):
       return slot_d_out[0]




# ============================================================
# Occlusion composition
# ============================================================


class OccludeAdd(Composition):


   def __init__(
       self,
       clamp=None,
       add="step",
       ste=False,
       alpha=1,
       color_channel=True,
       threshold=0.0,
       temperature=0.1,
   ):


       super().__init__()


       if temperature <= 0:
           raise ValueError(
               "temperature must be greater than zero."
           )


       self.clamp = clamp
       self.add = add
       self.ste = ste
       self.alpha = alpha
       self.color_channel = color_channel
       self.threshold = threshold
       self.temperature = temperature


       allowed_modes = (
           "step",
           "sigmoid",
           "hardsigmoid",
           "add",
       )


       if add not in allowed_modes:
           raise ValueError(
               f"Unknown addition mode: {add}"
           )


   def _get_mask_value(self, a):


       if self.color_channel:
           return a.mean(
               dim=-1,
               keepdim=True,
           )


       return a


   def _add_step(self, a, b):


       mask_value = self._get_mask_value(a)


       mask = mask_value > self.threshold


       return torch.where(
           mask,
           a,
           b,
       )


   def _add_soft(self, a, b):


       mask_value = self._get_mask_value(a)


       alpha_a = torch.sigmoid(
           (mask_value - self.threshold)
           / self.temperature
       )


       return (
           alpha_a * a
           + (1.0 - alpha_a) * b
       )


   def _add_sigmoid(self, a, b):


       mask_value = self._get_mask_value(a)


       alpha_a = torch.sigmoid(
           mask_value * self.alpha
       )


       alpha_b = torch.sigmoid(
           -mask_value * self.alpha
       )


       return (
           a * alpha_a
           + b * alpha_b
       )


   def _add_hardsigmoid(self, a, b):


       mask_value = self._get_mask_value(a)


       alpha_a = F.hardsigmoid(
           mask_value * self.alpha
       )


       alpha_b = F.hardsigmoid(
           -mask_value * self.alpha
       )


       return (
           a * alpha_a
           + b * alpha_b
       )


   def forward(self, x):


       if not isinstance(x, (list, tuple)):
           raise TypeError(
               "OccludeAdd expects a list or tuple of slot outputs."
           )


       if len(x) == 0:
           raise ValueError(
               "No slot outputs were provided."
           )


       reference_shape = x[0].shape


       for slot in x[1:]:


           if slot.shape != reference_shape:
               raise ValueError(
                   "All slot outputs must have identical shapes. "
                   f"Got {reference_shape} and {slot.shape}."
               )


       x = torch.stack(
           x,
           dim=1,
       )


       out = x[:, 0, ...]


       for slot in range(1, x.shape[1]):


           next_slot = x[:, slot, ...]


           if self.add == "step":


               hard_out = self._add_step(
                   out,
                   next_slot,
               )


               if self.ste:


                   soft_out = self._add_soft(
                       out,
                       next_slot,
                   )


                   # Hard forward pass, soft backward pass.
                   out = (
                       soft_out
                       + (hard_out - soft_out).detach()
                   )


               else:


                   out = hard_out


           elif self.add == "sigmoid":


               out = self._add_sigmoid(
                   out,
                   next_slot,
               )


           elif self.add == "hardsigmoid":


               out = self._add_hardsigmoid(
                   out,
                   next_slot,
               )


           elif self.add == "add":


               out = next_slot + out


       # Optional output activation.
       if self.clamp == "clamp":


           out = torch.clamp(
               out,
               0,
               1,
           )


       elif self.clamp == "relu":


           out = F.relu(out)


       elif self.clamp == "softplus":


           out = F.softplus(out)


       elif self.clamp == "sigmoid":


           out = torch.sigmoid(out)


       elif self.clamp is not None:


           raise ValueError(
               f"Unknown composition clamp: {self.clamp}"
           )


       return out


   def get_d_out(self, slot_d_out):
       return slot_d_out[0]




# ============================================================
# Alpha composition
# ============================================================


class AlphaAdd(Composition):


   def __init__(self, clamp=True):


       super().__init__()


       self.clamp = clamp


   def _alpha_add(self, a, b):


       alpha_a = a[..., -1:]
       alpha_b = b[..., -1:]


       alpha = (
           alpha_a
           + alpha_b * (1.0 - alpha_a)
       )


       rgb = (
           a[..., :-1] * alpha_a
           + b[..., :-1]
           * alpha_b
           * (1.0 - alpha_a)
       ) / alpha.clamp(min=1e-8)


       return torch.cat(
           [
               rgb,
               alpha,
           ],
           dim=-1,
       )


   def forward(self, x):


       if not isinstance(x, (list, tuple)):
           raise TypeError(
               "AlphaAdd expects a list or tuple of slot outputs."
           )


       if len(x) == 0:
           raise ValueError(
               "No slot outputs were provided."
           )


       for slot in x:


           if slot.shape[-1] != 4:
               raise ValueError(
                   "AlphaAdd requires four-channel RGBA slot outputs."
               )


       x = torch.stack(
           x,
           dim=1,
       )


       # Avoid modifying the stacked tensor in place.
       if self.clamp:


           rgb = x[..., :-1]


           alpha = torch.clamp(
               x[..., -1:],
               0,
               1,
           )


           x = torch.cat(
               [
                   rgb,
                   alpha,
               ],
               dim=-1,
           )


       # Transparent background.
       out = torch.cat(
           [
               torch.zeros_like(
                   x[:, 0, ..., :-1]
               ),
               torch.ones_like(
                   x[:, 0, ..., -1:]
               ),
           ],
           dim=-1,
       )


       # Composite from back to front.
       for slot in reversed(range(x.shape[1])):


           out = self._alpha_add(
               x[:, slot, ...],
               out,
           )


       return out[..., :-1]


   def get_d_out(self, slot_d_out):


       return (
           slot_d_out[0][:-1]
           + [3]
       )




# ============================================================
# Complete compositional model
# ============================================================


class CompositionalFunction(nn.Module):


   def __init__(
       self,
       composition,
       slots,
   ):


       super().__init__()


       self.composition = composition
       self.slots = slots


       self.d_in = slots.d_in
       self.d_slots_out = slots.d_out


       self.d_out = composition.get_d_out(
           slots.d_out
       )


   def forward(
       self,
       x,
       return_slot_outputs=False,
   ):


       slot_outputs = self.slots(x)


       out = self.composition(
           slot_outputs
       )


       if return_slot_outputs:
           return out, slot_outputs


       return out


   def get_slot(self, idx):
       return self.slots[idx]
