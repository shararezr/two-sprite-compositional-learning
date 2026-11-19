
"""
Extract hidden representations from trained two-sprite dSprites models.

Compositional models:
    Saves representations from slot 1 and slot 2 separately.

Monolithic models:
    Saves the shared representation from the single decoder.

Three representation levels are extracted:

    initial:
        Output of the final fully connected layer, immediately
        before the first Unflatten layer.

    intermediate:
        Output immediately after the first intermediate Conv2d layer.

    final:
        Output immediately after the last intermediate Conv2d layer,
        immediately before the final image-producing convolution.

Convolutional representations are adaptively average-pooled to 4x4
before being flattened.

All checkpoints are evaluated on the same latent inputs, stored in
common_latents.npy. The evaluation inputs are independent of the
training sampling strategy.
"""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils
from torch.utils.data import DataLoader, TensorDataset

from . import train
from .data import DSpritesDataset


FACTOR_SIZES = np.array([3, 6, 1, 32, 32])

FACTOR_NAMES = [
    "shape",
    "scale",
    "colour",
    "x_position",
    "y_position",
]

N_SLOTS = 2
LATENT_DIM = 10

POOL_SIZE = (4, 4)

REPRESENTATION_LEVELS = [
    "initial",
    "intermediate",
    "final",
]


# ------------------------------------------------------------
# Hashing and reproducibility
# ------------------------------------------------------------

def array_sha256(array):
    """Calculate a SHA-256 hash of an array's contents."""

    array = np.ascontiguousarray(array)

    return hashlib.sha256(
        array.tobytes()
    ).hexdigest()


def file_sha256(path):
    """Calculate a SHA-256 hash of a file."""

    digest = hashlib.sha256()

    with open(path, "rb") as f:

        for chunk in iter(
            lambda: f.read(1024 * 1024),
            b"",
        ):
            digest.update(chunk)

    return digest.hexdigest()


# ------------------------------------------------------------
# Loading checkpoints and models
# ------------------------------------------------------------

def load_trained_model(checkpoint_path, device):
    """Load a checkpoint and reconstruct its model using train.py."""

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=False,
    )

    config = checkpoint["config"]

    # build_models returns both the target composition and the model.
    _, model = train.build_models(config)

    model.load_state_dict(checkpoint["model"])

    model.to(device)
    model.eval()

    monolithic = config.get(
        "model", {}
    ).get("monolithic", False)

    architecture = (
        "monolithic" if monolithic else "compositional"
    )

    return model, config, checkpoint, architecture


# ------------------------------------------------------------
# Common evaluation latent inputs
# ------------------------------------------------------------

def to_numpy(value):
    """Convert a NumPy array or tensor to a NumPy array."""

    if torch.is_tensor(value):
        return value.detach().cpu().numpy()

    return np.asarray(value)


def validate_latents(latents):
    """Validate normalized, discrete two-sprite dSprites latents."""

    latents = np.asarray(latents)

    if latents.ndim != 2 or latents.shape[1] != LATENT_DIM:

        raise ValueError(
            f"Expected latent inputs with shape [N, {LATENT_DIM}], "
            f"but received {latents.shape}."
        )

    if latents.shape[0] == 0:

        raise ValueError(
            "The latent input array is empty."
        )

    if not np.isfinite(latents).all():

        raise ValueError(
            "Latent inputs contain NaN or infinite values."
        )

    if np.any(latents < 0) or np.any(latents > 1):

        raise ValueError(
            "Expected normalized latent values in [0, 1]."
        )

    # Every latent coordinate must lie on its discrete dSprites grid.
    sizes = np.tile(
        FACTOR_SIZES,
        N_SLOTS,
    )

    scaled = latents * (sizes - 1)

    nearest_grid_values = np.rint(scaled)

    if not np.allclose(
        scaled,
        nearest_grid_values,
        atol=1e-5,
    ):

        raise ValueError(
            "Latent inputs contain values outside the discrete "
            "dSprites factor grid."
        )

    # Colour has cardinality one and must remain zero.
    colour_indices = [2, 7]

    if not np.allclose(
        latents[:, colour_indices],
        0.0,
        atol=1e-6,
    ):

        raise ValueError(
            "Colour latent dimensions must be zero for both sprites."
        )


def create_common_latents(
    config,
    latent_path,
    n_samples=10000,
    seed=12345,
):
    """
    Create a common evaluation set if one does not already exist.

    The same saved latent inputs should be reused for every checkpoint.
    """

    latent_path = Path(latent_path)

    if latent_path.suffix != ".npy":

        raise ValueError(
            "Creating new common latent inputs requires a .npy path. "
            "Use load_latents() to load an existing .npz file."
        )

    latent_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    if latent_path.exists():

        print(
            f"Loading existing common latent inputs: {latent_path}"
        )

        latents = np.load(latent_path)

    else:

        print(
            "Creating common random evaluation latent inputs..."
        )

        data_config = config["data"]

        base_dataset = DSpritesDataset(
            root=data_config["root"]
        )

        source_dataset = train.create_source_dataset(
            base_dataset=base_dataset,
            n_samples=n_samples,
            sampling="random",
            seed=seed,
        )

        latents = to_numpy(
            source_dataset.z
        ).astype(np.float32)

        validate_latents(latents)

        np.save(
            latent_path,
            latents,
        )

        print(
            f"Saved common latent inputs to: {latent_path}"
        )

    validate_latents(latents)

    return latents.astype(np.float32)


def load_latents(latent_path):
    """Load an existing common latent input file."""

    latent_path = Path(latent_path)

    if latent_path.suffix == ".npy":

        latents = np.load(latent_path)

    elif latent_path.suffix == ".npz":

        with np.load(latent_path) as loaded:

            if "latents" not in loaded:

                raise KeyError(
                    "The NPZ file must contain an array named 'latents'."
                )

            latents = loaded["latents"].copy()

    else:

        raise ValueError(
            "Latent input file must be .npy or .npz."
        )

    validate_latents(latents)

    return latents.astype(np.float32)


# ------------------------------------------------------------
# Factor labels
# ------------------------------------------------------------

def get_factor_labels(latents):
    """
    Convert normalized latent values into discrete dSprites labels.

    Model-facing order:
        [shape, scale, colour, x_position, y_position] for each slot.
    """

    sizes = np.tile(
        FACTOR_SIZES,
        N_SLOTS,
    )

    labels = np.rint(
        np.clip(latents, 0.0, 1.0) * (sizes - 1)
    ).astype(np.int64)

    return labels


def get_factor_names():
    """Return names for all ten slot-specific latent dimensions."""

    names = []

    for slot in range(N_SLOTS):

        for factor in FACTOR_NAMES:

            names.append(
                f"slot{slot + 1}_{factor}"
            )

    return names


# ------------------------------------------------------------
# Identifying representation layers
# ------------------------------------------------------------

def find_representation_layers(decoder):
    """
    Identify the three representation points in an UpsampleMLP decoder.

    Initial:
        Final Linear layer before the first Unflatten.

    Intermediate:
        First Conv2d layer after Unflatten.

    Final:
        Last Conv2d feature layer before the final image-producing
        Conv2d layer.
    """

    children = list(
        decoder.named_children()
    )

    unflatten_position = None

    for i, (_, module) in enumerate(children):

        if isinstance(module, nn.Unflatten):

            unflatten_position = i
            break

    if unflatten_position is None:

        raise ValueError(
            "Could not find an Unflatten layer in the decoder. "
            "This implementation expects an UpsampleMLP-style decoder."
        )

    linear_layers = [
        (name, module)
        for name, module in children[:unflatten_position]
        if isinstance(module, nn.Linear)
    ]

    if not linear_layers:

        raise ValueError(
            "Could not find a Linear layer before Unflatten."
        )

    initial_name, initial_layer = linear_layers[-1]

    convolutional_layers = [
        (name, module)
        for name, module in children[unflatten_position + 1:]
        if isinstance(module, nn.Conv2d)
    ]

    if len(convolutional_layers) < 2:

        raise ValueError(
            "Expected at least two Conv2d layers after Unflatten, "
            f"but found {len(convolutional_layers)}."
        )

    intermediate_name, intermediate_layer = (
        convolutional_layers[0]
    )

    final_name, final_layer = (
        convolutional_layers[-2]
    )

    return {
        "initial": {
            "module": initial_layer,
            "name": initial_name,
        },
        "intermediate": {
            "module": intermediate_layer,
            "name": intermediate_name,
        },
        "final": {
            "module": final_layer,
            "name": final_name,
        },
    }


# ------------------------------------------------------------
# Processing captured activations
# ------------------------------------------------------------

def process_activation(output, level):
    """
    Convert a captured activation into a representation matrix.

    Initial:
        Keep the fully connected output unchanged.

    Intermediate and final:
        Apply adaptive average pooling to 4x4, then flatten.
    """

    if not torch.is_tensor(output):

        raise TypeError(
            "Expected a tensor output from the representation layer."
        )

    if level == "initial":

        return output.detach().cpu().numpy()

    if level in ("intermediate", "final"):

        if output.ndim != 4:

            raise ValueError(
                f"Expected a 4D convolutional activation for {level}, "
                f"but received shape {tuple(output.shape)}."
            )

        pooled = F.adaptive_avg_pool2d(
            output,
            POOL_SIZE,
        )

        flattened = pooled.flatten(
            start_dim=1
        )

        return flattened.detach().cpu().numpy()

    raise ValueError(
        f"Unknown representation level: {level}"
    )


# ------------------------------------------------------------
# Register hooks for compositional models
# ------------------------------------------------------------

def register_compositional_hooks(model):
    """
    Register hooks for all three representation levels in each slot.

    If reuse_phi=True, both slots may use the same decoder object.
    Its invocations are assigned to slot 1 and slot 2 in call order.
    """

    slot_decoders = model.slots.slot_functions

    captured = {}

    handles = []

    layer_groups_by_level = {}

    layer_names = {}

    for level in REPRESENTATION_LEVELS:

        captured[level] = {
            "slot1": [],
            "slot2": [],
        }

        layer_groups = {}

        for slot_index, decoder in enumerate(slot_decoders):

            layers = find_representation_layers(decoder)

            layer_info = layers[level]

            layer = layer_info["module"]
            layer_name = layer_info["name"]

            layer_id = id(layer)

            if layer_id not in layer_groups:

                layer_groups[layer_id] = {
                    "layer": layer,
                    "slot_indices": [],
                    "call_count": 0,
                    "module_name": layer_name,
                }

            layer_groups[layer_id]["slot_indices"].append(
                slot_index
            )

        layer_groups_by_level[level] = list(
            layer_groups.values()
        )

        layer_names[level] = {
            "slot1": None,
            "slot2": None,
        }

        for group in layer_groups.values():

            for slot_index in group["slot_indices"]:

                layer_names[level][
                    f"slot{slot_index + 1}"
                ] = group["module_name"]

            def make_hook(group_state, representation_level):

                def hook(module, inputs, output):

                    call_index = group_state["call_count"]

                    group_state["call_count"] += 1

                    slot_indices = group_state["slot_indices"]

                    if call_index >= len(slot_indices):

                        raise RuntimeError(
                            "Unexpected number of calls to a shared "
                            "decoder."
                        )

                    slot_index = slot_indices[call_index]

                    key = f"slot{slot_index + 1}"

                    representation = process_activation(
                        output,
                        representation_level,
                    )

                    captured[representation_level][key].append(
                        representation
                    )

                return hook

            handle = group["layer"].register_forward_hook(
                make_hook(group, level)
            )

            handles.append(handle)

    return (
        captured,
        handles,
        layer_groups_by_level,
        layer_names,
    )


# ------------------------------------------------------------
# Register hooks for monolithic models
# ------------------------------------------------------------

def register_monolithic_hooks(model):
    """Register hooks for all three representation levels."""

    layers = find_representation_layers(model)

    captured = {
        level: {"shared": []}
        for level in REPRESENTATION_LEVELS
    }

    handles = []

    layer_groups_by_level = {}

    layer_names = {}

    for level in REPRESENTATION_LEVELS:

        layer_info = layers[level]

        layer = layer_info["module"]

        layer_names[level] = {
            "shared": layer_info["name"]
        }

        def make_hook(representation_level):

            def hook(module, inputs, output):

                representation = process_activation(
                    output,
                    representation_level,
                )

                captured[representation_level]["shared"].append(
                    representation
                )

            return hook

        handle = layer.register_forward_hook(
            make_hook(level)
        )

        handles.append(handle)

    return (
        captured,
        handles,
        layer_groups_by_level,
        layer_names,
    )


# ------------------------------------------------------------
# Representation extraction
# ------------------------------------------------------------

def extract_representations(
    model,
    latents,
    architecture,
    device,
    batch_size=256,
):
    """
    Extract all three representation levels for supplied latent inputs.
    """

    if batch_size <= 0:

        raise ValueError(
            "batch_size must be greater than zero."
        )

    latent_tensor = torch.as_tensor(
        latents,
        dtype=torch.float32,
    )

    dataset = TensorDataset(
        latent_tensor
    )

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
    )

    if architecture == "compositional":

        (
            captured,
            handles,
            layer_groups_by_level,
            layer_names,
        ) = register_compositional_hooks(model)

    elif architecture == "monolithic":

        (
            captured,
            handles,
            layer_groups_by_level,
            layer_names,
        ) = register_monolithic_hooks(model)

    else:

        raise ValueError(
            f"Unknown architecture: {architecture}"
        )

    try:

        with torch.no_grad():

            for (z,) in loader:

                z = z.to(device)

                # Reset shared-decoder invocation counters for
                # every representation level and every batch.
                for groups in layer_groups_by_level.values():

                    for group in groups:

                        group["call_count"] = 0

                model(
                    train.prepare_model_input(
                        z,
                        monolithic=(
                            architecture == "monolithic"
                        ),
                    )
                )

    finally:

        for handle in handles:

            handle.remove()

    representations = {}

    for level in REPRESENTATION_LEVELS:

        for key, batches in captured[level].items():

            if not batches:

                raise RuntimeError(
                    f"No representations were captured for "
                    f"{level}_{key}."
                )

            representation = np.concatenate(
                batches,
                axis=0,
            )

            if representation.shape[0] != len(latents):

                raise RuntimeError(
                    f"Expected {len(latents)} representations for "
                    f"{level}_{key}, but extracted "
                    f"{representation.shape[0]}."
                )

            if not np.isfinite(representation).all():

                raise RuntimeError(
                    f"Non-finite activations detected in {level}_{key}."
                )

            output_key = f"{level}_{key}"

            representations[output_key] = representation

    return representations, layer_names


# ------------------------------------------------------------
# Saving representations and metadata
# ------------------------------------------------------------

def save_representations(
    representations,
    layer_names,
    latents,
    output_dir,
    architecture,
    checkpoint,
    config,
    checkpoint_path,
    batch_size,
    seed,
):
    """Save representation matrices and associated metadata."""

    output_dir = Path(output_dir)

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    labels = get_factor_labels(latents)

    factor_names = get_factor_names()

    sample_ids = np.arange(
        len(latents)
    )

    for name, representation in representations.items():

        output_path = output_dir / f"{name}.npz"

        np.savez_compressed(
            output_path,
            representations=representation,
            latents=latents,
            labels=labels,
            factor_names=np.asarray(factor_names),
            sample_ids=sample_ids,
        )

        print(
            f"Saved {name}: "
            f"{representation.shape} -> {output_path}"
        )

    metadata = {
        "architecture": architecture,

        "checkpoint_path": str(checkpoint_path),

        "checkpoint_sha256": file_sha256(
            checkpoint_path
        ),

        "epoch": checkpoint.get("epoch"),

        "n_samples": int(len(latents)),

        "evaluation_latents_sha256": array_sha256(
            latents
        ),

        "evaluation_seed": int(seed),

        "extraction_batch_size": int(batch_size),

        "representation_shapes": {
            name: list(values.shape)
            for name, values in representations.items()
        },

        "representation_levels": {
            "initial": (
                "Final Linear output immediately before Unflatten"
            ),
            "intermediate": (
                "Output immediately after the first intermediate Conv2d"
            ),
            "final": (
                "Output immediately after the last intermediate Conv2d, "
                "before the final image-producing Conv2d"
            ),
        },

        "representation_module_names": layer_names,

        "convolutional_pooling": {
            "method": "adaptive average pooling",
            "output_size": list(POOL_SIZE),
        },

        "factor_names": factor_names,

        "factor_sizes": FACTOR_SIZES.tolist(),

        "latent_dimension": LATENT_DIM,

        "number_of_slots": N_SLOTS,

        "model_config": config.get("model", {}),

        "data_config": config.get("data", {}),

        "training_sampling": config.get(
            "train", {}
        ).get("sampling"),

        "software_versions": {
            "numpy": np.__version__,
            "pytorch": torch.__version__,
        },
    }

    metadata_path = output_dir / "metadata.json"

    with open(metadata_path, "w") as f:

        json.dump(
            metadata,
            f,
            indent=4,
            default=str,
        )

    print(
        f"Saved metadata to {metadata_path}"
    )


# ------------------------------------------------------------
# Main
# ------------------------------------------------------------

def main():

    parser = argparse.ArgumentParser(
        description=(
            "Extract hidden representations from a trained "
            "two-sprite dSprites model."
        )
    )

    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to the trained checkpoint.",
    )

    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Directory in which representations will be saved.",
    )

    parser.add_argument(
        "--latents",
        type=str,
        default="common_latents.npy",
        help="Path to the shared evaluation latent inputs.",
    )

    parser.add_argument(
        "--n-samples",
        type=int,
        default=10000,
        help=(
            "Number of evaluation samples if the latent file "
            "is created."
        ),
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=12345,
        help="Seed used when creating common evaluation inputs.",
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=256,
        help="Extraction batch size.",
    )

    args = parser.parse_args()

    if args.batch_size <= 0:

        parser.error(
            "--batch-size must be greater than zero."
        )

    if args.n_samples <= 0:

        parser.error(
            "--n-samples must be greater than zero."
        )

    checkpoint_path = Path(
        args.checkpoint
    )

    if not checkpoint_path.is_file():

        parser.error(
            f"Checkpoint does not exist: {checkpoint_path}"
        )

    device = train.DEVICE

    print(
        f"Using device: {device}"
    )

    # Load model and checkpoint.
    model, config, checkpoint, architecture = load_trained_model(
        checkpoint_path,
        device,
    )

    print(
        f"Detected architecture: {architecture}"
    )

    # Load or create the shared latent inputs.
    latent_path = Path(
        args.latents
    )

    if latent_path.exists():

        latents = load_latents(
            latent_path
        )

    else:

        latents = create_common_latents(
            config=config,
            latent_path=latent_path,
            n_samples=args.n_samples,
            seed=args.seed,
        )

    print(
        f"Number of evaluation samples: {len(latents)}"
    )

    print(
        "Evaluation latent SHA-256:",
        array_sha256(latents),
    )

    # Extract representations.
    representations, layer_names = extract_representations(
        model=model,
        latents=latents,
        architecture=architecture,
        device=device,
        batch_size=args.batch_size,
    )

    # Save representations and metadata.
    save_representations(
        representations=representations,
        layer_names=layer_names,
        latents=latents,
        output_dir=args.output,
        architecture=architecture,
        checkpoint=checkpoint,
        config=config,
        checkpoint_path=checkpoint_path,
        batch_size=args.batch_size,
        seed=args.seed,
    )

    print(
        "Representation extraction complete."
    )


if __name__ == "__main__":

    main()
