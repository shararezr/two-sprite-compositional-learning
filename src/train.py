"""
train.py


Training script for compositional two-sprite dSprites.


Compatible with:
   data.py
   model.py


Latent input:
   [B, 10] for BOTH compositional and monolithic models.


Model inputs are normalized labels of directly selected dSprites records.
Evaluation sampling="train" inherits the training sampler and its settings.


Image representation:
   [B, 64, 64, 1]
"""


import copy
import datetime
import random
import warnings
from math import prod
from pathlib import Path


import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


from torch.utils.data import DataLoader
from tqdm import tqdm


from . import model as models




try:
   import wandb
except ImportError:
   wandb = None




from .data import DSpritesDataset, DSpritesTwoSpriteDataset, CachedCompositionDataset




# ================================================================
# Constants
# ================================================================


DEVICE = torch.device(
   "cuda:0" if torch.cuda.is_available() else "cpu"
)


N_SPRITES = 2
LATENT_DIM = 10


IMAGE_SHAPE = (64, 64, 1)
N_PIXELS = prod(IMAGE_SHAPE)




# ================================================================
# Reproducibility
# ================================================================


def set_seed(seed):


   if seed is None:
       return


   random.seed(seed)
   np.random.seed(seed)
   torch.manual_seed(seed)


   if torch.cuda.is_available():
       torch.cuda.manual_seed_all(seed)




# ================================================================
# Model input
# ================================================================


def prepare_model_input(z, monolithic=False):
   """
   Both model types receive flattened latent vectors [B, 10].


   The dataset is responsible for converting discrete factor
   labels into normalized latent coordinates before returning z.


   ParallelSlots splits the flattened input internally.
   """


   if z.ndim != 2 or z.shape[1] != LATENT_DIM:
       raise ValueError(
           f"Expected latent shape [B, {LATENT_DIM}], "
           f"got {tuple(z.shape)}"
       )


   return z




# ================================================================
# Model configuration
# ================================================================


def get_activation(value):


   if isinstance(value, str):


       if not hasattr(nn, value):
           raise ValueError(
               f"Unknown torch.nn activation: {value}"
           )


       activation = getattr(nn, value)


       if not isinstance(activation, type):
           raise ValueError(
               f"{value} is not an activation class."
           )


       return activation()


   if isinstance(value, type) and issubclass(value, nn.Module):
       return value()


   if isinstance(value, nn.Module):
       return copy.deepcopy(value)


   raise TypeError(
       "nonlin must be an activation name, "
       "nn.Module class, or nn.Module instance."
   )




def prepare_model_kwargs(kwargs):


   kwargs = copy.deepcopy(kwargs)


   if "nonlin" in kwargs:
       kwargs["nonlin"] = get_activation(
           kwargs["nonlin"]
       )


   return kwargs




# ================================================================
# Objectness loss
# ================================================================


def get_pairwise_dists(w, h, device=None):


   if device is None:
       device = DEVICE


   x = torch.arange(
       w,
       device=device,
   ).view(1, -1).repeat(h, 1)


   y = torch.arange(
       h,
       device=device,
   ).view(-1, 1).repeat(1, w)


   grid = torch.stack(
       [x, y],
       dim=0,
   ).unsqueeze(0)


   pairwise_diff = (
       grid.view(1, 2, -1).unsqueeze(-1)
       - grid.view(1, 2, 1, -1)
   )


   return torch.sqrt(
       torch.sum(pairwise_diff ** 2, dim=1)
   )




def objectness_score(input, pairwise_dist):


   batch_size = input.shape[0]


   input = input.abs().sum(-1)


   input_min = input.min()
   input_max = input.max()


   input = (
       (input - input_min)
       / (input_max - input_min + 1e-8)
   )


   weighted_pairwise_dist = (
       pairwise_dist
       * input.reshape(batch_size, -1, 1)
       * input.reshape(batch_size, 1, -1)
   )


   return weighted_pairwise_dist.mean()




def objectness_loss_batched(
   input,
   pairwise_dists=None,
   batch_size=16,
):


   if input.shape[0] == 0:
       raise ValueError(
           "Objectness loss received an empty batch."
       )


   if batch_size < 1:
       raise ValueError(
           "Objectness batch_size must be positive."
       )


   if pairwise_dists is None:
       pairwise_dists = get_pairwise_dists(
           input.shape[2],
           input.shape[1],
           device=input.device,
       )


   total_loss = input.new_tensor(0.0)
   total_samples = input.shape[0]


   for chunk in input.split(batch_size, dim=0):


       chunk_loss = objectness_score(
           chunk,
           pairwise_dists,
       )


       total_loss += (
           chunk_loss * chunk.shape[0] / total_samples
       )


   return total_loss




class ObjectnessLoss(nn.Module):


   def __init__(
       self,
       size=(64, 64),
       batch_size=16,
   ):


       super().__init__()


       self.batch_size = batch_size


       self.register_buffer(
           "pairwise_dists",
           get_pairwise_dists(
               size[1],
               size[0],
               device=DEVICE,
           ),
       )


   def forward(self, input):


       return objectness_loss_batched(
           input,
           self.pairwise_dists,
           self.batch_size,
       )




# ================================================================
# Loss functions
# ================================================================


def build_criterion(name, alpha=1.0, **kwargs):


   if name == "L1":
       return nn.L1Loss(**kwargs)


   if name in ("MSE", "L2"):
       return nn.MSELoss(**kwargs)


   if name == "crossentropy":
       raise ValueError(
           "CrossEntropyLoss is not appropriate for "
           "grayscale image regression."
       )


   if name == "MSE+sparsity":


       def criterion(target, prediction):


           return (
               F.mse_loss(target, prediction)
               + alpha * torch.norm(prediction, p=1)
           )


       return criterion


   if name == "MSE+objectness":


       objectness = ObjectnessLoss(
           size=(64, 64),
           batch_size=kwargs.get(
               "objectness_batch_size",
               16,
           ),
       ).to(DEVICE)


       def criterion(target, prediction):


           return (
               F.mse_loss(target, prediction)
               + alpha * objectness(prediction)
           )


       return criterion


   raise ValueError(
       f"Unknown criterion: {name}"
   )




# ================================================================
# Evaluation metrics
# ================================================================


def build_metrics(metric_names):


   n_outputs = N_PIXELS
   metrics = {}


   for name in metric_names:


       if name == "R2Score":


           from torchmetrics import R2Score


           metrics[name] = R2Score(
               num_outputs=n_outputs,
               multioutput="uniform_average",
           ).to(DEVICE)


       elif name == "R2ScoreVW":


           from torchmetrics import R2Score


           metrics[name] = R2Score(
               num_outputs=n_outputs,
               multioutput="variance_weighted",
           ).to(DEVICE)


       elif name == "MSE":


           from torchmetrics import MeanSquaredError


           metrics[name] = MeanSquaredError().to(DEVICE)


       else:


           raise ValueError(
               f"Unknown metric: {name}"
           )


   return metrics




# ================================================================
# Evaluation
# ================================================================


@torch.no_grad()
def evaluate(model, loaders, metric_names, monolithic=False):


   model.eval()


   scores = {}
   diagnostics_printed = False


   def print_stats(name, tensor):


       print(
           f"{name}: "
           f"shape={tuple(tensor.shape)}, "
           f"min={tensor.min().item():.6f}, "
           f"max={tensor.max().item():.6f}, "
           f"mean={tensor.mean().item():.6f}, "
           f"std={tensor.std().item():.6f}"
       )


   for loader_name, loader in loaders.items():


       sum_squared_error = None
       sum_y = None
       sum_y_squared = None


       n_samples = 0


       for batch_idx, (z, target) in enumerate(loader):


           z = z.to(DEVICE)
           target = target.to(DEVICE)


           model_input = prepare_model_input(
               z,
               monolithic,
           )


           prediction = model(model_input)


           if not diagnostics_printed and batch_idx == 0:


               print("\n" + "=" * 65)
               print(f"SLOT DIAGNOSTICS - Dataset: {loader_name}")
               print("=" * 65)


               print("\nMODEL INPUT")
               print_stats("Model input", model_input)


               if hasattr(model, "slots"):


                   slot_outputs = model.slots(model_input)


                   if isinstance(slot_outputs, (list, tuple)):


                       for i, slot in enumerate(slot_outputs):
                           print_stats(f"Slot {i + 1}", slot)


                   elif torch.is_tensor(slot_outputs):


                       print_stats(
                           "Combined slot output",
                           slot_outputs,
                       )


                       if (
                           slot_outputs.ndim >= 2
                           and slot_outputs.shape[0] == model_input.shape[0]
                           and slot_outputs.shape[1] == 2
                       ):


                           for i in range(2):


                               print_stats(
                                   f"Slot {i + 1}",
                                   slot_outputs[:, i],
                               )


                       elif (
                           slot_outputs.ndim >= 2
                           and slot_outputs.shape[0] == 2
                           and slot_outputs.shape[1] == model_input.shape[0]
                       ):


                           for i in range(2):


                               print_stats(
                                   f"Slot {i + 1}",
                                   slot_outputs[i],
                               )


               print("\nTARGET")
               print_stats("Target", target)


               print("\nFINAL PREDICTION")
               print_stats("Prediction", prediction)


               print("=" * 65)
               print("END SLOT DIAGNOSTICS")
               print("=" * 65 + "\n")


               diagnostics_printed = True


           if prediction.shape != target.shape:


               raise ValueError(
                   f"Prediction shape {prediction.shape} "
                   f"does not match target shape {target.shape}"
               )


           prediction = prediction.flatten(1).double()
           target = target.flatten(1).double()


           squared_error = (
               prediction - target
           ).square().sum(dim=0)


           batch_sum = target.sum(dim=0)
           batch_sum_squared = target.square().sum(dim=0)


           if sum_squared_error is None:


               sum_squared_error = squared_error
               sum_y = batch_sum
               sum_y_squared = batch_sum_squared


           else:


               sum_squared_error += squared_error
               sum_y += batch_sum
               sum_y_squared += batch_sum_squared


           n_samples += target.shape[0]


       if n_samples == 0:


           raise ValueError(
               f"Empty evaluation loader: {loader_name}"
           )


       sst = sum_y_squared - sum_y.square() / n_samples
       valid = sst > 1e-12


       if "MSE" in metric_names:


           scores[f"MSE_{loader_name}"] = (
               sum_squared_error.sum()
               / (n_samples * target.shape[1])
           ).item()


       if "R2Score" in metric_names:


           if valid.any():


               r2_per_pixel = (
                   1.0 - sum_squared_error[valid] / sst[valid]
               )


               scores[f"R2Score_{loader_name}"] = (
                   r2_per_pixel.mean().item()
               )


           else:


               scores[f"R2Score_{loader_name}"] = float("nan")


       if "R2ScoreVW" in metric_names:


           total_sst = sst.sum()


           scores[f"R2ScoreVW_{loader_name}"] = (
               (
                   1.0
                   - sum_squared_error.sum() / total_sst
               ).item()
               if total_sst > 1e-12
               else float("nan")
           )


   return scores




# ================================================================
# Training epoch
# ================================================================


def train_epoch(
   model,
   loader,
   optimizer,
   criterion,
   accum_batches=1,
   monolithic=False,
):


   if accum_batches < 1:


       raise ValueError(
           "accum_batches must be at least 1."
       )


   model.train()
   optimizer.zero_grad(set_to_none=True)


   if len(loader) == 0:


       raise ValueError(
           "Training DataLoader is empty."
       )


   total_loss = 0.0
   total_samples = 0
   n_batches = len(loader)


   for batch_idx, (z, target) in enumerate(loader):


       z = z.to(
           DEVICE,
           non_blocking=True,
       )


       target = target.to(
           DEVICE,
           non_blocking=True,
       )


       model_z = prepare_model_input(
           z,
           monolithic=monolithic,
       )


       prediction = model(model_z)


       if prediction.shape != target.shape:


           raise ValueError(
               "Prediction and target shapes differ: "
               f"{tuple(prediction.shape)} vs "
               f"{tuple(target.shape)}"
           )


       loss = criterion(target, prediction)


       current_batch_size = z.shape[0]


       group_start = (
           batch_idx // accum_batches
       ) * accum_batches


       group_end = min(
           group_start + accum_batches,
           n_batches,
       )


       group_size = 0


       for batch_number in range(group_start, group_end):


           start = batch_number * loader.batch_size


           end = min(
               start + loader.batch_size,
               len(loader.dataset),
           )


           group_size += end - start


       if group_size <= 0:


           raise RuntimeError(
               "Invalid gradient accumulation group."
           )


       scaled_loss = loss * (
           current_batch_size / group_size
       )


       scaled_loss.backward()


       total_loss += (
           loss.detach().item() * current_batch_size
       )


       total_samples += current_batch_size


       should_step = (
           (batch_idx + 1) % accum_batches == 0
           or (batch_idx + 1) == n_batches
       )


       if should_step:


           optimizer.step()
           optimizer.zero_grad(set_to_none=True)


   return total_loss / total_samples




# ================================================================
# Model construction
# ================================================================


def build_models(cfg):


   data_cfg = cfg["data"]
   model_cfg = cfg["model"]


   D = data_cfg.get("D", [5, 5])


   M = data_cfg.get(
       "M",
       [
           list(IMAGE_SHAPE),
           list(IMAGE_SHAPE),
       ],
   )


   if list(D) != [5, 5]:


       raise ValueError(
           f"Expected D=[5, 5], got {D}"
       )


   if len(M) != 2:


       raise ValueError(
           "Expected two output shapes in data.M."
       )


   for output_shape in M:


       if tuple(output_shape) != IMAGE_SHAPE:


           raise ValueError(
               f"Expected output shape {IMAGE_SHAPE}, "
               f"got {output_shape}"
           )


   composition_class = getattr(
       models,
       data_cfg.get("C", "OccludeAdd"),
   )


   composition_kwargs = dict(
       data_cfg.get(
           "C_kwargs",
           {
               "add": "sigmoid",
               "color_channel": False,
           },
       )
   )


   target_composition = composition_class(
       **composition_kwargs
   ).to(DEVICE)


   model_class = getattr(
       models,
       model_cfg.get("phi", "UpsampleMLP"),
   )


   model_kwargs = prepare_model_kwargs(
       model_cfg.get("phi_kwargs", {})
   )


   monolithic = model_cfg.get("monolithic", False)


   if monolithic:


       model = model_class(
           sum(D),
           M[0],
           **model_kwargs,
       ).to(DEVICE)


   else:


       if model_cfg.get("reuse_phi", False):


           if D[0] != D[1] or M[0] != M[1]:


               raise ValueError(
                   "Cannot reuse slot networks when their "
                   "input or output dimensions differ."
               )


           shared_slot = model_class(
               D[0],
               M[0],
               **model_kwargs,
           )


           trainable_slots = [
               shared_slot,
               shared_slot,
           ]


       else:


           trainable_slots = [
               model_class(
                   D[i],
                   M[i],
                   **copy.deepcopy(model_kwargs),
               )
               for i in range(N_SPRITES)
           ]


       trainable_parallel = models.ParallelSlots(
           trainable_slots
       )


       # Use the same composition for target and model by default.
       trainable_composition_kwargs = dict(
           data_cfg.get(
               "trainable_C_kwargs",
               composition_kwargs,
           )
       )


       trainable_composition = composition_class(
           **trainable_composition_kwargs
       )


       model = models.CompositionalFunction(
           trainable_composition,
           trainable_parallel,
       ).to(DEVICE)


   return target_composition, model




# ================================================================
# Dataset construction
# ================================================================


def create_source_dataset(
   base_dataset,
   n_samples,
   sampling,
   seed=None,
   **sampling_kwargs,
):


   if sampling == "continuous_random":
       warnings.warn(
           "continuous_random now selects discrete dataset records uniformly; "
           "use sampling: random. This does not test continuous generalization.",
           stacklevel=2,
       )
       sampling = "random"


   return DSpritesTwoSpriteDataset(
       dataset=base_dataset,
       n_samples=n_samples,
       mode=sampling,
       seed=seed,
       **sampling_kwargs,
   )




def create_cached_dataset(
   source,
   target_composition,
   batch_size=256,
   num_workers=0,
):


   if len(source) == 0:


       raise ValueError(
           "Cannot cache an empty source dataset."
       )


   cached = CachedCompositionDataset(
       source=source,
       composition=target_composition,
       device=DEVICE,
       batch_size=batch_size,
       num_workers=num_workers,
   )


   if cached.z.ndim != 2:


       raise ValueError(
           f"Expected 2D latent tensor, got {cached.z.shape}"
       )


   if cached.z.shape[1] != LATENT_DIM:


       raise ValueError(
           f"Expected latent shape [N, {LATENT_DIM}], "
           f"got {tuple(cached.z.shape)}"
       )


   if tuple(cached.targets.shape[1:]) != IMAGE_SHAPE:


       raise ValueError(
           f"Expected target shape [N, {IMAGE_SHAPE}], "
           f"got {tuple(cached.targets.shape)}"
       )


   if len(cached.z) != len(cached.targets):


       raise RuntimeError(
           "Cached latent and target lengths differ."
       )


   cached.source = None
   cached.composition = None


   return cached




def make_loader(
   dataset,
   batch_size,
   shuffle=False,
   num_workers=0,
):


   if len(dataset) == 0:


       raise ValueError(
           "Cannot create a DataLoader for an empty dataset."
       )


   if batch_size < 1:


       raise ValueError(
           "batch_size must be positive."
       )


   return DataLoader(
       dataset,
       batch_size=batch_size,
       shuffle=shuffle,
       num_workers=num_workers,
       pin_memory=(DEVICE.type == "cuda"),
       drop_last=False,
   )




def build_datasets(cfg, target_composition):


   data_cfg = cfg["data"]
   train_cfg = cfg["train"]
   eval_cfg = cfg.get("eval", {})


   base_dataset = DSpritesDataset(
       root=data_cfg["root"],
   )


   cache_batch_size = data_cfg.get(
       "cache_batch_size",
       256,
   )


   num_workers = data_cfg.get(
       "num_workers",
       0,
   )


   base_seed = cfg.get("seed", 42)


   # ------------------------------------------------------------
   # Training dataset
   # ------------------------------------------------------------


   train_seed = train_cfg.get(
       "seed",
       base_seed,
   )


   train_source = create_source_dataset(
       base_dataset,
       n_samples=train_cfg["n_samples"],
       sampling=train_cfg.get("sampling", "random"),
       seed=train_seed,
       **train_cfg.get("sampling_kwargs", {}),
   )


   train_dataset = create_cached_dataset(
       train_source,
       target_composition,
       batch_size=cache_batch_size,
       num_workers=num_workers,
   )


   train_loader = make_loader(
       train_dataset,
       batch_size=train_cfg["batch_size"],
       shuffle=True,
       num_workers=num_workers,
   )


   # ------------------------------------------------------------
   # Evaluation datasets
   # ------------------------------------------------------------


   eval_loaders = {}


   used_seeds = set()


   if train_seed is not None:
       used_seeds.add(int(train_seed))


   for eval_index, (name, dataset_cfg) in enumerate(
       eval_cfg.get("data", {}).items()
   ):


       sampling = dataset_cfg.get(
           "sampling",
           "random",
       )


       n_samples = dataset_cfg.get(
           "n_samples",
           1000,
       )


       default_seed = (
           None
           if base_seed is None
           else int(base_seed) + 10000 + eval_index
       )


       dataset_seed = dataset_cfg.get(
           "seed",
           default_seed,
       )


       if dataset_seed is not None:


           dataset_seed = int(dataset_seed)


           if dataset_seed in used_seeds:


               raise ValueError(
                   f"Evaluation dataset {name!r} uses seed "
                   f"{dataset_seed}, which is already assigned "
                   "to another dataset."
               )


           used_seeds.add(dataset_seed)


       sampling_kwargs = dataset_cfg.get("sampling_kwargs", {})
       if sampling == "train":
           sampling = train_cfg.get("sampling", "random")
           sampling_kwargs = {
               **train_cfg.get("sampling_kwargs", {}),
               **sampling_kwargs,
           }


       source = create_source_dataset(
           base_dataset,
           n_samples=n_samples,
           sampling=sampling,
           seed=dataset_seed,
           **sampling_kwargs,
       )
       eval_dataset = create_cached_dataset(
           source,
           target_composition,
           batch_size=cache_batch_size,
           num_workers=num_workers,
       )


       eval_loaders[name] = make_loader(
           eval_dataset,
           batch_size=eval_cfg.get("batch_size", 128),
           shuffle=False,
           num_workers=num_workers,
       )


   return train_loader, eval_loaders




# ================================================================
# Visualisation
# ================================================================


@torch.no_grad()
def visualize_reconstruction(
   model,
   loader,
   monolithic=False,
   n_examples=4,
):


   model.eval()


   z, target = next(iter(loader))


   n_examples = min(
       n_examples,
       len(z),
   )


   z = z[:n_examples].to(DEVICE)
   target = target[:n_examples]


   prediction = model(
       prepare_model_input(z, monolithic)
   )


   slot_outputs = []


   if not monolithic and hasattr(model, "slots"):


       slot_outputs = model.slots(z)


       if not isinstance(slot_outputs, (list, tuple)):
           slot_outputs = []


   prediction = prediction.cpu()


   slot_outputs = [
       slot.cpu()
       for slot in slot_outputs
   ]


   n_cols = 2 + len(slot_outputs)


   fig, axes = plt.subplots(
       n_examples,
       n_cols,
       figsize=(3 * n_cols, 3 * n_examples),
       squeeze=False,
   )


   for i in range(n_examples):


       axes[i, 0].imshow(
           target[i].squeeze(-1),
           cmap="gray",
           vmin=0,
           vmax=1,
       )


       axes[i, 0].set_title("Target")


       axes[i, 1].imshow(
           prediction[i].squeeze(-1),
           cmap="gray",
           vmin=0,
           vmax=1,
       )


       axes[i, 1].set_title("Prediction")


       for slot_idx, slot in enumerate(slot_outputs):


           axes[i, slot_idx + 2].imshow(
               slot[i].squeeze(-1),
               cmap="gray",
               vmin=0,
               vmax=1,
           )


           axes[i, slot_idx + 2].set_title(
               f"Slot {slot_idx + 1}"
           )


       for ax in axes[i]:
           ax.axis("off")


   fig.tight_layout()


   return fig




@torch.no_grad()
def visualize_mse_on_grid(
   model,
   loader,
   monolithic=False,
):


   model.eval()


   all_mse = []
   all_z = []


   for z, target in loader:


       z = z.to(DEVICE)
       target = target.to(DEVICE)


       prediction = model(
           prepare_model_input(
               z,
               monolithic=monolithic,
           )
       )


       mse = (
           prediction - target
       ).pow(2).flatten(1).mean(dim=1)


       all_mse.append(mse.cpu())
       all_z.append(z.cpu())


   all_mse = torch.cat(all_mse)
   all_z = torch.cat(all_z)


   x_position = all_z[:, 3].numpy()
   y_position = all_z[:, 4].numpy()
   mse_values = all_mse.numpy()


   fig, ax = plt.subplots(figsize=(7, 6))


   scatter = ax.scatter(
       x_position,
       y_position,
       c=mse_values,
       s=12,
   )


   ax.set_xlabel("First sprite x position")
   ax.set_ylabel("First sprite y position")


   ax.set_title(
       "Reconstruction MSE across first-sprite position"
   )


   fig.colorbar(
       scatter,
       ax=ax,
       label="MSE",
   )


   fig.tight_layout()


   return fig




# ================================================================
# Checkpointing
# ================================================================


def save_checkpoint(
   path,
   model,
   optimizer=None,
   scheduler=None,
   epoch=None,
   best_scores=None,
   cfg=None,
):


   checkpoint = {
       "model": model.state_dict(),
       "epoch": epoch,
       "best_scores": best_scores,
       "config": cfg,
       "torch_rng_state": torch.get_rng_state(),
       "numpy_rng_state": np.random.get_state(),
       "python_rng_state": random.getstate(),
   }


   if optimizer is not None:
       checkpoint["optimizer"] = optimizer.state_dict()


   if scheduler is not None:
       checkpoint["scheduler"] = scheduler.state_dict()


   if torch.cuda.is_available():
       checkpoint["cuda_rng_state"] = torch.cuda.get_rng_state_all()


   torch.save(checkpoint, path)




def load_checkpoint(
   path,
   model,
   optimizer=None,
   scheduler=None,
   restore_rng=True,
):


   try:


       checkpoint = torch.load(
           path,
           map_location="cpu",
           weights_only=False,
       )


   except TypeError:


       checkpoint = torch.load(
           path,
           map_location="cpu",
       )


   if isinstance(checkpoint, dict) and "model" in checkpoint:


       model.load_state_dict(
           checkpoint["model"]
       )


       if optimizer is not None and "optimizer" in checkpoint:
           optimizer.load_state_dict(checkpoint["optimizer"])


       if scheduler is not None and "scheduler" in checkpoint:
           scheduler.load_state_dict(checkpoint["scheduler"])


       if restore_rng:


           if "torch_rng_state" in checkpoint:
               torch.set_rng_state(
                   checkpoint["torch_rng_state"].cpu()
               )


           if (
               torch.cuda.is_available()
               and "cuda_rng_state" in checkpoint
           ):
               torch.cuda.set_rng_state_all(
                   checkpoint["cuda_rng_state"]
               )


           if "numpy_rng_state" in checkpoint:
               np.random.set_state(
                   checkpoint["numpy_rng_state"]
               )


           if "python_rng_state" in checkpoint:
               random.setstate(
                   checkpoint["python_rng_state"]
               )


       return (
           checkpoint.get("epoch", -1) + 1,
           checkpoint.get("best_scores", {}) or {},
       )


   model.load_state_dict(checkpoint)


   return 0, {}




# ================================================================
# Main experiment
# ================================================================


def run(**cfg):


   set_seed(cfg.get("seed"))


   now_str = datetime.datetime.now().strftime(
       "%Y%m%d-%H%M%S"
   )


   save_dir = (
       Path(cfg["save_dir"])
       / f'{cfg["save_name"]}_{now_str}'
   )


   save_dir.mkdir(
       parents=True,
       exist_ok=False,
   )


   cfg["save_dir"] = str(save_dir)


   import yaml


   with (save_dir / "config.yaml").open("w") as file:


       yaml.safe_dump(
           cfg,
           file,
           sort_keys=False,
       )


   if cfg["train"].get("use_cudnn_backend", False):
       torch.backends.cudnn.benchmark = True


   # ------------------------------------------------------------
   # Models
   # ------------------------------------------------------------


   target_composition, model = build_models(cfg)


   monolithic = cfg["model"].get(
       "monolithic",
       False,
   )


   # ------------------------------------------------------------
   # Datasets
   # ------------------------------------------------------------


   train_loader, eval_loaders = build_datasets(
       cfg,
       target_composition,
   )


   # ------------------------------------------------------------
   # Loss
   # ------------------------------------------------------------


   criterion = build_criterion(
       cfg["train"]["loss"],
       **cfg["train"].get("loss_kwargs", {}),
   )


   # ------------------------------------------------------------
   # Optimizer
   # ------------------------------------------------------------


   optimizer_class = getattr(
       torch.optim,
       cfg["train"]["optimizer"],
   )


   optimizer = optimizer_class(
       model.parameters(),
       **cfg["train"].get("optimizer_kwargs", {}),
   )


   # ------------------------------------------------------------
   # Scheduler
   # ------------------------------------------------------------


   scheduler = None


   scheduler_name = cfg["train"].get("scheduler")


   if scheduler_name:


       scheduler_class = getattr(
           torch.optim.lr_scheduler,
           scheduler_name,
       )


       scheduler = scheduler_class(
           optimizer,
           **cfg["train"].get("scheduler_kwargs", {}),
       )


   # ------------------------------------------------------------
   # Checkpoint loading
   # ------------------------------------------------------------


   start_epoch = 0
   best_scores = {}


   checkpoint_path = cfg["train"].get("checkpoint")


   if checkpoint_path:


       start_epoch, best_scores = load_checkpoint(
           checkpoint_path,
           model,
           optimizer=optimizer,
           scheduler=scheduler,
           restore_rng=True,
       )


       print(
           f"Resuming from checkpoint: {checkpoint_path}"
       )


       print(
           f"Starting at epoch {start_epoch + 1}"
       )


   # ------------------------------------------------------------
   # W&B
   # ------------------------------------------------------------


   wandb_run = None
   wandb_cfg = cfg.get("wandb", {})


   if wandb_cfg.get("enabled", False):


       if wandb is None:


           raise ImportError(
               "W&B is enabled but wandb is not installed."
           )


       wandb_run = wandb.init(
           project=wandb_cfg.get("project", "CGFP"),
           config=cfg,
       )


       if wandb_cfg.get("watch", False):


           wandb.watch(
               model,
               log="all",
               log_freq=wandb_cfg.get("watch_freq", 1000),
           )


   # ------------------------------------------------------------
   # Evaluation configuration
   # ------------------------------------------------------------


   eval_cfg = cfg.get("eval", {})


   do_eval = bool(eval_cfg) and bool(eval_loaders)


   metric_names = eval_cfg.get(
       "metrics",
       ["MSE", "R2ScoreVW"],
   )


   # ------------------------------------------------------------
   # Visualization configuration
   # ------------------------------------------------------------


   vis_cfg = cfg.get("visualization", {})
   do_vis = bool(vis_cfg)


   vis_freq = vis_cfg.get("freq", 1)


   if vis_freq < 1:


       raise ValueError(
           "Visualization frequency must be positive."
       )


   history = []


   # ============================================================
   # Training
   # ============================================================


   try:


       for epoch in tqdm(
           range(start_epoch, cfg["train"]["epochs"]),
           desc="Epochs",
       ):


           log = {}


           train_loss = train_epoch(
               model,
               train_loader,
               optimizer,
               criterion,
               accum_batches=cfg["train"].get(
                   "accum_batches",
                   1,
               ),
               monolithic=monolithic,
           )


           log["loss"] = train_loss


           # ----------------------------------------------------
           # Evaluation
           # ----------------------------------------------------


           should_evaluate = (
               do_eval
               and (epoch + 1) % eval_cfg.get("freq", 1) == 0
           )


           if should_evaluate:


               scores = evaluate(
                   model,
                   eval_loaders,
                   metric_names,
                   monolithic=monolithic,
               )


               log.update(scores)


               print(
                   f"\nEpoch {epoch + 1}/{cfg['train']['epochs']} "
                   f"| Training loss: {train_loss:.6f}"
               )


               for metric_name, value in scores.items():


                   print(
                       f"  {metric_name}: {value:.6f}"
                   )


               # Track the best All MSE when available.
               if "MSE_All" in scores:


                   current = scores["MSE_All"]
                   previous = best_scores.get(
                       "MSE_All",
                       float("inf"),
                   )


                   if current < previous:


                       best_scores["MSE_All"] = current


                       save_checkpoint(
                           save_dir / "best_checkpoint.pt",
                           model,
                           optimizer,
                           scheduler,
                           epoch + 1,
                           best_scores,
                           cfg,
                       )


           # ----------------------------------------------------
           # Scheduler
           # ----------------------------------------------------


           if scheduler is not None:


               if isinstance(
                   scheduler,
                   torch.optim.lr_scheduler.ReduceLROnPlateau,
               ):


                   scheduler.step(train_loss)


               else:


                   scheduler.step()


           learning_rate = optimizer.param_groups[0]["lr"]


           log["learning_rate"] = learning_rate


           # ----------------------------------------------------
           # Visualizations
           # ----------------------------------------------------


           if do_vis and (epoch + 1) % vis_freq == 0:


               for vis_name, vis_data_cfg in (
                   vis_cfg.get("data", {}).items()
               ):


                   if vis_name not in eval_loaders:


                       print(
                           f"Skipping visualization '{vis_name}': "
                           "no matching evaluation loader."
                       )


                       continue


                   vis_type = vis_data_cfg["type"]


                   if vis_type == "heatmap":


                       fig = visualize_mse_on_grid(
                           model,
                           eval_loaders[vis_name],
                           monolithic=monolithic,
                       )


                   elif vis_type == "reconstruction":


                       fig = visualize_reconstruction(
                           model,
                           eval_loaders[vis_name],
                           monolithic=monolithic,
                       )


                   else:


                       raise ValueError(
                           f"Unsupported visualization type: {vis_type}"
                       )


                   if wandb_run is not None:
                       log[vis_name] = wandb.Image(fig)


                   fig.savefig(
                       save_dir
                       / f"{vis_name}_epoch_{epoch + 1}.png",
                       bbox_inches="tight",
                   )


                   plt.close(fig)


           # ----------------------------------------------------
           # Logging
           # ----------------------------------------------------


           history.append(log)


           if wandb_run is not None:
               wandb.log(log, step=epoch)


       # ========================================================
       # FINAL EVALUATION
       # ========================================================


       print("\n")
       print("=" * 65)
       print("FINAL TRAINING AND EVALUATION RESULTS")
       print("=" * 65)


       model.eval()


       final_train_scores = evaluate(
           model,
           {"Train": train_loader},
           ["MSE"],
           monolithic=monolithic,
       )


       final_train_mse = final_train_scores["MSE_Train"]


       print(
           f"\nFinal training MSE: "
           f"{final_train_mse:.8f}"
       )


       final_eval_scores = {}


       if eval_loaders:


           final_eval_scores = evaluate(
               model,
               eval_loaders,
               ["MSE"],
               monolithic=monolithic,
           )


           print("\nFinal evaluation MSE:")


           for metric_name, value in final_eval_scores.items():


               print(
                   f"  {metric_name}: {value:.8f}"
               )


       else:


           print(
               "\nNo evaluation datasets were configured."
           )


       final_r2_scores = {}


       requested_r2 = [
           name
           for name in metric_names
           if name in ("R2Score", "R2ScoreVW")
       ]


       if eval_loaders and requested_r2:


           final_r2_scores = evaluate(
               model,
               eval_loaders,
               requested_r2,
               monolithic=monolithic,
           )


           print("\nFinal R2 metrics:")


           for metric_name, value in final_r2_scores.items():


               print(
                   f"  {metric_name}: {value:.8f}"
               )


       print("\n" + "=" * 65)
       print("FINAL EVALUATION COMPLETE")
       print("=" * 65 + "\n")


       final_results = {
           "final_train_mse": final_train_mse,
           **final_eval_scores,
           **final_r2_scores,
       }


       history.append(final_results)


       with (save_dir / "final_results.txt").open("w") as file:


           file.write(
               "FINAL TRAINING AND EVALUATION RESULTS\n"
           )


           file.write("=" * 50 + "\n\n")


           file.write(
               f"Final training MSE: {final_train_mse:.8f}\n\n"
           )


           file.write("Final evaluation metrics:\n")


           for metric_name, value in final_eval_scores.items():


               file.write(
                   f"{metric_name}: {value:.8f}\n"
               )


           for metric_name, value in final_r2_scores.items():


               file.write(
                   f"{metric_name}: {value:.8f}\n"
               )


       # Always save a final checkpoint.
       save_checkpoint(
           save_dir / "final_checkpoint.pt",
           model,
           optimizer,
           scheduler,
           cfg["train"]["epochs"],
           best_scores,
           cfg,
       )


       if wandb_run is not None:
           wandb.log(final_results)


       print(
           f"Final results saved to: "
           f"{save_dir / 'final_results.txt'}"
       )


   finally:


       if wandb_run is not None:
           wandb.finish()


   return model, history
