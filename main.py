"""Edit DEFAULT_CONFIG or pass a YAML override to run an experiment."""


import argparse
from copy import deepcopy
from pathlib import Path


from src.train import run
from src.utils import deep_update, load_config




DEFAULT_CONFIG = {
   "save_dir": "./experiments",
   "save_name": "two_sprite_dsprites_original_cgfp",
   "seed": 42,
   "data": {
       "root": "/mnt/primary/compositional-ood-generalization/data/dsprites",
       "D": [5, 5],
       "M": [
           [64, 64, 1],
           [64, 64, 1],
       ],
       "C": "OccludeAdd",
       "C_kwargs": {
           "add": "sigmoid",
           "color_channel": False,
       },
       "trainable_C_kwargs": {
           "add": "sigmoid",
           "color_channel": False,
       },
       "cache_batch_size": 256,
       "num_workers": 0,
   },
   "model": {
       "monolithic": False,
       "reuse_phi": False,
       "phi": "UpsampleMLP",
       "phi_kwargs": {
           "d_hidden": 256,
           "n_channel": 32,
           "kernel_size": 2,
           "nonlin": "ELU",
           "clamp": None,
       },
   },
   "train": {
       "n_samples": 100000,
       "sampling": "orthogonal",
       "sampling_kwargs": {},
       "batch_size": 512,
       "epochs": 2000,
       "accum_batches": 1,
       "loss": "MSE",
       "loss_kwargs": {},
       "optimizer": "AdamW",
       "optimizer_kwargs": {
           "lr": 0.001,
           "weight_decay": 1e-5,
       },
       "scheduler": "CosineAnnealingLR",
       "scheduler_kwargs": {
           "T_max": 2000,
           "eta_min": 1e-5,
       },
       # Optional path to a checkpoint to resume.
       "checkpoint": None,
       "use_cudnn_backend": True,
   },
   "eval": {
       "freq": 100,
       "batch_size": 1000,
       "metrics": [
           "MSE",
           "R2Score",
           "R2ScoreVW",
       ],
       "save_scores": {
           "MSE_All": "min",
       },
       "data": {
           "ID": {
               "sampling": "train",
               "n_samples": 10000,
               "seed": 10042,
           },
           "All": {
               "sampling": "random",
               "n_samples": 10000,
               "seed": 20042,
           },
       },
   },
   "visualization": {
       "freq": 100,
       "data": {
           "ID": {
               "type": "reconstruction",
           },
           "All": {
               "type": "reconstruction",
           },
       },
   },
   "wandb": {
       "enabled": False,
       "project": "CGFP",
       "watch": False,
       "watch_freq": 100,
   },
}




def main(argv=None):
   parser = argparse.ArgumentParser(description="Train a two-sprite experiment.")
   parser.add_argument("--config", type=Path, help="YAML overrides for DEFAULT_CONFIG.")
   args = parser.parse_args(argv)


   config = deepcopy(DEFAULT_CONFIG)
   if args.config is not None:
       config = deep_update(config, load_config(args.config))
   run(**config)




if __name__ == "__main__":
   main()
