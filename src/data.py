# ============================================================
# Version 2 (change the Diagonal)
# ============================================================


import numpy as np
import torch

from math import sqrt
from pathlib import Path
from torch.utils.data import Dataset, DataLoader


# ============================================================
# Constants
# ============================================================

# Model-facing factor order:
# [shape, scale, colour, x_position, y_position]

FACTOR_SIZES = (3, 6, 1, 32, 32)
N_FACTORS = 5
N_SLOTS = 2
FIXED_ORIENTATION = 0


# ============================================================
# Continuous-to-discrete label conversion
# ============================================================

def continuous_to_factor_labels(z, factor_sizes=FACTOR_SIZES):
    """
    Convert continuous latent coordinates to discrete dSprites
    factor labels for image lookup.

    Accepts NumPy arrays or PyTorch tensors.

    Input:
        [..., 5]

    Returns:
        NumPy int64 array with shape [..., 5].
    """
    if torch.is_tensor(z):
        z = z.detach().cpu().numpy()

    z = np.asarray(z, dtype=np.float32)
    sizes = np.asarray(factor_sizes, dtype=np.int64)

    if z.ndim == 0 or z.shape[-1] != N_FACTORS:
        raise ValueError(
            f"Expected latent coordinates with final dimension "
            f"{N_FACTORS}, but received shape {z.shape}."
        )

    if sizes.shape != (N_FACTORS,):
        raise ValueError(
            f"factor_sizes must contain exactly {N_FACTORS} values."
        )

    if np.any(sizes < 1):
        raise ValueError("All factor sizes must be at least 1.")

    labels = np.rint(
        np.clip(z, 0.0, 1.0) * (sizes - 1)
    ).astype(np.int64)

    # Colour has only one possible value.
    labels[..., 2] = 0

    return labels


def quantize_latents(z, factor_sizes=FACTOR_SIZES):
    """
    Quantize continuous latent coordinates to normalized discrete
    dSprites factor values.

    Supported shapes:
        [..., 5]  : individual sprite latents
        [..., 10] : flattened two-sprite latents

    Returns the same type/device as the input.
    """
    is_tensor = torch.is_tensor(z)

    if is_tensor:
        original_device = z.device
        original_dtype = z.dtype
        z_array = z.detach().cpu().numpy()
    else:
        z_array = np.asarray(z)

    z_array = np.asarray(z_array, dtype=np.float32)
    original_shape = z_array.shape

    sizes = np.asarray(factor_sizes, dtype=np.int64)

    if sizes.shape != (N_FACTORS,):
        raise ValueError(
            f"factor_sizes must contain exactly {N_FACTORS} values."
        )

    if np.any(sizes < 1):
        raise ValueError("All factor sizes must be at least 1.")

    if z_array.ndim == 0:
        raise ValueError("Latents must have at least one dimension.")

    if z_array.shape[-1] == N_FACTORS:
        pass
    elif z_array.shape[-1] == N_SLOTS * N_FACTORS:
        z_array = z_array.reshape(
            *z_array.shape[:-1],
            N_SLOTS,
            N_FACTORS
        )
    else:
        raise ValueError(
            "Expected final latent dimension "
            f"{N_FACTORS} or {N_SLOTS * N_FACTORS}, "
            f"but received shape {original_shape}."
        )

    labels = np.rint(
        np.clip(z_array, 0.0, 1.0) * (sizes - 1)
    ).astype(np.int64)

    labels[..., 2] = 0

    denominators = np.maximum(sizes - 1, 1)

    quantized = (
        labels.astype(np.float32) /
        denominators.astype(np.float32)
    )

    quantized[..., 2] = 0.0
    quantized = quantized.reshape(original_shape)

    if is_tensor:
        return torch.as_tensor(
            quantized,
            dtype=original_dtype,
            device=original_device
        )

    return quantized.astype(np.float32)


# ============================================================
# dSprites dataset
# ============================================================

class DSpritesDataset:
    """
    dSprites dataset adapted for two-sprite compositional learning.

    Original dSprites latent order:
        [colour, shape, scale, orientation, x, y]

    Model-facing latent order:
        [shape, scale, colour, x, y]

    Orientation is fixed to zero for image lookup.
    Colour is retained as a factor, but always equals zero.
    """

    FACTOR_SIZES = FACTOR_SIZES
    FIXED_ORIENTATION = FIXED_ORIENTATION

    def __init__(self, root):
        root = Path(root)

        if root.is_dir():
            files = list(root.glob("*.npz"))

            if not files:
                raise FileNotFoundError(
                    f"No dSprites .npz file found in {root}"
                )

            data_path = files[0]
        else:
            data_path = root

        data = np.load(data_path, encoding="latin1")

        self.images = data["imgs"]
        original_latents = data["latents_classes"].astype(np.int64)

        # Keep only orientation == 0.
        keep = (
            original_latents[:, 3] == self.FIXED_ORIENTATION
        )

        self.image_indices = np.flatnonzero(keep)
        selected = original_latents[keep]

        # Reorder:
        # [colour, shape, scale, orientation, x, y]
        # to:
        # [shape, scale, colour, x, y]
        self.latents = selected[:, [1, 2, 0, 4, 5]]

        self.label_to_image_index = {
            tuple(label): int(image_index)
            for label, image_index in zip(
                self.latents,
                self.image_indices
            )
        }

        # Map labels to indices in the orientation-filtered dataset.
        self.label_to_record_id = {
            tuple(label): int(record_id)
            for record_id, label in enumerate(self.latents)
        }

        expected = int(np.prod(self.FACTOR_SIZES))

        if len(self.latents) != expected:
            raise ValueError(
                f"Expected {expected} orientation-filtered images, "
                f"but found {len(self.latents)}."
            )

    def __len__(self):
        return len(self.latents)

    def continuous_to_labels(self, z):
        return continuous_to_factor_labels(
            z,
            self.FACTOR_SIZES
        )

    def quantize(self, z):
        return quantize_latents(
            z,
            self.FACTOR_SIZES
        )

    def get_image_indices(self, factor_labels):
        labels = np.asarray(
            factor_labels,
            dtype=np.int64
        )

        if labels.ndim == 1:
            labels = labels[None, :]

        if labels.ndim != 2 or labels.shape[1] != N_FACTORS:
            raise ValueError(
                f"Expected factor labels with shape [N, {N_FACTORS}]."
            )

        indices = []

        for label in labels:
            key = tuple(label.tolist())

            if key not in self.label_to_image_index:
                raise ValueError(
                    f"Invalid dSprites factor combination: {key}"
                )

            indices.append(self.label_to_image_index[key])

        return np.asarray(indices, dtype=np.int64)

    def get_images_by_indices(self, indices):
        indices = np.asarray(indices, dtype=np.int64)

        if indices.ndim == 0:
            return self.images[int(indices)]

        return self.images[indices]

    def get_images(self, factor_labels):
        indices = self.get_image_indices(factor_labels)
        return self.get_images_by_indices(indices)

    def get_images_from_continuous(self, z):
        labels = self.continuous_to_labels(z)
        return self.get_images(labels)

    def __getitem__(self, index):
        image = self.images[
            int(self.image_indices[int(index)])
        ]

        label = self.latents[int(index)]

        return image, label


# ============================================================
# Gap constraints
# ============================================================

def _normalise_gap_constraints(gap=None, gaps=None, gap_dims=None):
    """
    Minimum between-sprite separation, in normalized label coordinates.

    This preserves the existing gap-sampling definition.
    """
    if gaps is not None:
        gap = gaps

    if gap is None:
        return {}

    if isinstance(gap, dict):
        return {
            int(dim): float(value)
            for dim, value in gap.items()
        }

    if np.isscalar(gap):
        if gap_dims is None:
            gap_dims = [0, 1, 3, 4]

        return {
            int(dim): float(gap)
            for dim in gap_dims
        }

    values = list(gap)

    if gap_dims is None:
        gap_dims = range(len(values))

    return {
        int(dim): float(values[dim])
        for dim in gap_dims
        if dim < len(values) and values[dim] is not None
    }


# ============================================================
# Original-style diagonal sampling
# ============================================================

def _sample_diagonal_continuous(
    n_samples,
    delta,
    rng,
    max_attempts=100000,
):
    """
    Adaptation of the original diagonal sampler.

    For every pair:

        z_base ~ Uniform([0, 1]^5)

        z_on_diag = [z_base, z_base]

        z_pair = z_on_diag + orthogonal_noise * radius

    The perturbation is perpendicular to the diagonal, so the
    two slots move in opposite directions.

    The perturbation is generated independently for each factor,
    following the structure of the original implementation.

    Returns:
        [N, 2, 5] continuous latent coordinates.

    Note:
        dSprites colour is fixed, so colour is reset to zero.
        The other four factors are perturbed.
    """
    max_delta = sqrt(2) / 2

    if (
        not np.isfinite(delta)
        or delta <= 0
        or delta >= max_delta
    ):
        raise ValueError(
            f"delta must be greater than 0 and smaller than "
            f"{max_delta:.6f}, but got {delta}."
        )

    if n_samples == 0:
        return np.empty(
            (0, N_SLOTS, N_FACTORS),
            dtype=np.float32
        )

    samples = []
    filled = 0
    rejected = 0

    while filled < n_samples:
        remaining = n_samples - filled

        count = min(
            max(4 * remaining, 256),
            8192,
            max_attempts - rejected
        )

        if count <= 0:
            raise ValueError(
                "Could not sample enough diagonal points within "
                "max_attempts. Check delta."
            )

        # Shared base point on the diagonal.
        z_in_slot = rng.random(
            (count, N_FACTORS)
        ).astype(np.float32)

        z_on_diag = np.repeat(
            z_in_slot[:, None, :],
            N_SLOTS,
            axis=1
        )

        # Generate noise using the same construction as the
        # original implementation.
        noise = rng.normal(
            size=(count, N_SLOTS + 2, N_FACTORS)
        ).astype(np.float32)

        noise_norm = np.linalg.norm(
            noise,
            axis=1,
            keepdims=True
        )

        noise = noise / np.maximum(
            noise_norm,
            1e-12
        )

        # Remove the two extra vectors.
        noise = noise[:, :N_SLOTS, :]

        # Project noise onto the hyperplane perpendicular
        # to the diagonal.
        numerator = (
            noise * z_on_diag
        ).sum(axis=1, keepdims=True)

        denominator = (
            z_on_diag ** 2
        ).sum(axis=1, keepdims=True)

        ort_vec = (
            noise
            - z_on_diag
            * numerator
            / np.maximum(denominator, 1e-12)
        )

        # Normalize the perturbation direction per factor.
        ort_norm = np.linalg.norm(
            ort_vec,
            axis=1,
            keepdims=True
        )

        ort_vec = ort_vec / np.maximum(
            ort_norm,
            1e-12
        )

        # For two slots, the exponent is 1 / (2 - 1) = 1.
        # This follows the original radius sampling.
        radius = (
            rng.random((count, 1, N_FACTORS))
            ** (1.0 / (N_SLOTS - 1))
        ) * delta

        z_pair = z_on_diag + ort_vec * radius

        # dSprites colour has only one possible value.
        z_pair[:, :, 2] = 0.0

        # Reject samples outside the unit cube.
        valid = np.all(
            (z_pair >= 0.0) & (z_pair <= 1.0),
            axis=(1, 2)
        )

        accepted = z_pair[valid]

        if len(accepted) > 0:
            accepted = accepted[:remaining]
            samples.append(accepted)
            filled += len(accepted)
            rejected = 0
        else:
            rejected += count

        if rejected >= max_attempts:
            raise ValueError(
                "No valid diagonal samples accepted within "
                "max_attempts. Try a smaller delta."
            )

    return np.concatenate(samples, axis=0).astype(np.float32)


def _sample_diagonal_record_pairs(
    dataset,
    n_samples,
    delta,
    rng,
    max_attempts=100000,
):
    """
    Sample continuous diagonal latents, quantize them to dSprites
    labels, and convert those labels to dataset record IDs.

    Returns:
        [N, 2] indices into dataset.latents.
    """
    continuous = _sample_diagonal_continuous(
        n_samples=n_samples,
        delta=delta,
        rng=rng,
        max_attempts=max_attempts
    )

    if n_samples == 0:
        return np.empty((0, N_SLOTS), dtype=np.int64)

    labels = dataset.continuous_to_labels(
        continuous.reshape(-1, N_FACTORS)
    )

    labels = labels.reshape(
        n_samples,
        N_SLOTS,
        N_FACTORS
    )

    pairs = np.empty(
        (n_samples, N_SLOTS),
        dtype=np.int64
    )

    for i in range(n_samples):
        for slot in range(N_SLOTS):
            key = tuple(labels[i, slot].tolist())

            if key not in dataset.label_to_record_id:
                raise RuntimeError(
                    f"Sampled dSprites label does not exist: {key}"
                )

            pairs[i, slot] = dataset.label_to_record_id[key]

    return pairs


# ============================================================
# Sampling existing sprite records
# ============================================================

def sample_record_pairs(
    dataset,
    n_samples,
    mode="random",
    seed=None,
    planes=None,
    delta=0.1,
    gap=None,
    gaps=None,
    gap_dims=None,
    grid_size=10,
    max_points=1000000,
    max_attempts=100000,
):
    """
    Return [N, 2] indices into dataset.latents.

    Modes
    -----
    random:
        Independently sample two records uniformly, with replacement.

    orthogonal:
        Equal-probability planes, each varying one slot uniformly.
        Example:
            planes=[
                {"slot": 0, "record_id": 7},
                {"slot": 1, "record_id": 7}
            ]

    diagonal:
        Original-style diagonal sampling:
        sample a shared continuous base vector, perturb the two slots
        in opposite directions perpendicular to the diagonal, reject
        out-of-cube points, then quantize to dSprites labels.

    random_gap / orthogonal_gap:
        Additionally enforce minimum separation on selected factors.

    grid:
        Uniformly sample records whose labels lie on a discrete grid.

    full_grid / face_grid:
        Sample pairs without replacement.
        If n_samples=None, enumerate all pairs.

    Record IDs refer to the orientation-filtered dataset, not the
    original NPZ row indices.
    """
    modes = {
        "random",
        "random_gap",
        "orthogonal",
        "orthogonal_gap",
        "diagonal",
        "grid",
        "full_grid",
        "face_grid"
    }

    if mode not in modes:
        raise ValueError(
            f"Unknown sampling mode {mode!r}. "
            f"Available modes: {sorted(modes)}"
        )

    if n_samples is None:
        if mode not in ("full_grid", "face_grid"):
            raise ValueError(
                "n_samples is required for this sampling mode."
            )
    elif (
        not isinstance(n_samples, (int, np.integer))
        or n_samples < 0
    ):
        raise ValueError(
            "n_samples must be a non-negative integer."
        )

    if len(dataset) == 0:
        raise ValueError("Cannot sample an empty dataset.")

    if max_attempts < 1:
        raise ValueError("max_attempts must be positive.")

    rng = np.random.default_rng(seed)

    labels = dataset.latents
    sizes = np.asarray(dataset.FACTOR_SIZES)

    normalized = (
        labels / np.maximum(sizes - 1, 1)
    )

    # --------------------------------------------------------
    # Diagonal sampling
    # --------------------------------------------------------

    if mode == "diagonal":
        if n_samples == 0:
            return np.empty(
                (0, N_SLOTS),
                dtype=np.int64
            )

        return _sample_diagonal_record_pairs(
            dataset=dataset,
            n_samples=n_samples,
            delta=delta,
            rng=rng,
            max_attempts=max_attempts
        )

    # --------------------------------------------------------
    # Grid sampling
    # --------------------------------------------------------

    if mode in ("grid", "full_grid", "face_grid"):
        if (
            not isinstance(grid_size, (int, np.integer))
            or grid_size < 1
        ):
            raise ValueError(
                "grid_size must be a positive integer."
            )

        keep = np.ones(
            len(dataset),
            dtype=bool
        )

        for dim, size in enumerate(sizes):
            levels = np.rint(
                np.linspace(
                    0,
                    size - 1,
                    min(grid_size, size)
                )
            ).astype(int)

            keep &= np.isin(
                labels[:, dim],
                levels
            )

        if mode == "face_grid":
            keep &= (
                np.count_nonzero(labels, axis=1) <= 2
            )

        records = np.flatnonzero(keep)

        if len(records) == 0:
            raise ValueError(
                "No existing records satisfy the grid settings."
            )

        if mode == "grid":
            return rng.choice(
                records,
                size=(n_samples, N_SLOTS)
            )

        total = len(records) ** 2

        count = (
            total
            if n_samples is None
            else n_samples
        )

        if count > total:
            raise ValueError(
                "Requested more samples than the grid contains."
            )

        if count > max_points:
            raise ValueError(
                f"Requested {count:,} pairs. "
                f"Maximum allowed is {max_points:,}."
            )

        selected = (
            np.arange(total)
            if n_samples is None
            else rng.choice(
                total,
                size=count,
                replace=False
            )
        )

        return np.column_stack(
            (
                records[selected // len(records)],
                records[selected % len(records)]
            )
        )

    # --------------------------------------------------------
    # Orthogonal sampling setup
    # --------------------------------------------------------

    anchors = []

    if mode in ("orthogonal", "orthogonal_gap"):
        if planes is None:
            planes = [
                {"slot": 0, "record_id": 0},
                {"slot": 1, "record_id": 0}
            ]

        if not planes:
            raise ValueError(
                "At least one orthogonal plane is required."
            )

        for plane in planes:
            slot = plane["slot"]

            if slot not in (0, 1):
                raise ValueError(
                    "Orthogonal plane slot must be 0 or 1."
                )

            if "record_id" in plane:
                record_id = plane["record_id"]

                if (
                    not isinstance(record_id, (int, np.integer))
                    or not 0 <= record_id < len(dataset)
                ):
                    raise ValueError(
                        "Orthogonal record_id must identify "
                        "an existing record."
                    )
            else:
                fixed = np.asarray(
                    plane["fixed"],
                    dtype=float
                )

                if fixed.shape != (N_FACTORS,):
                    raise ValueError(
                        "Each plane must specify five fixed coordinates."
                    )

                matches = np.flatnonzero(
                    np.all(
                        np.isclose(
                            normalized,
                            fixed,
                            rtol=0,
                            atol=1e-7
                        ),
                        axis=1
                    )
                )

                if len(matches) != 1:
                    raise ValueError(
                        "Fixed coordinates must identify an existing "
                        "record; use record_id."
                    )

                record_id = int(matches[0])

            anchors.append((slot, record_id))

    # --------------------------------------------------------
    # Gap constraints
    # --------------------------------------------------------

    constraints = {}

    if mode in ("random_gap", "orthogonal_gap"):
        constraints = _normalise_gap_constraints(
            gap,
            gaps,
            gap_dims
        )

        for dim, minimum in constraints.items():
            if (
                not 0 <= dim < N_FACTORS
                or not np.isfinite(minimum)
                or not 0 <= minimum <= 1
            ):
                raise ValueError(
                    "Gap constraints require valid factor dimensions "
                    "and distances in [0, 1]."
                )

            if sizes[dim] == 1 and minimum > 0:
                raise ValueError(
                    f"Cannot enforce a positive gap on fixed factor {dim}."
                )

    if n_samples == 0:
        return np.empty(
            (0, N_SLOTS),
            dtype=np.int64
        )

    # --------------------------------------------------------
    # Random / orthogonal / gap sampling
    # --------------------------------------------------------

    result = np.empty(
        (n_samples, N_SLOTS),
        dtype=np.int64
    )

    filled = 0
    rejected = 0

    while filled < n_samples:
        count = min(
            max(4 * (n_samples - filled), 256),
            8192,
            max_attempts - rejected
        )

        if count <= 0:
            raise ValueError(
                "No record pairs accepted within max_attempts; "
                "check sampling constraints."
            )

        pairs = rng.integers(
            len(dataset),
            size=(count, N_SLOTS)
        )

        if anchors:
            choices = rng.integers(
                len(anchors),
                size=count
            )

            for i, (slot, record_id) in enumerate(anchors):
                pairs[
                    choices == i,
                    1 - slot
                ] = record_id

        keep = np.ones(
            count,
            dtype=bool
        )

        for dim, minimum in constraints.items():
            separation = np.abs(
                normalized[pairs[:, 0], dim]
                - normalized[pairs[:, 1], dim]
            )

            keep &= (
                separation + 1e-12 >= minimum
            )

        accepted = pairs[keep][
            :n_samples - filled
        ]

        result[
            filled:filled + len(accepted)
        ] = accepted

        filled += len(accepted)

        if len(accepted) > 0:
            rejected = 0
        else:
            rejected += count

        if rejected >= max_attempts:
            raise ValueError(
                "No record pairs accepted within max_attempts; "
                "check sampling constraints."
            )

    return result


# ============================================================
# Two-sprite dataset
# ============================================================

class DSpritesTwoSpriteDataset(Dataset):
    """
    Select existing sprite pairs and return (images, normalized labels).

    images: [2, 64, 64, 1]
    z:      [10]

    record_ids stores the selected orientation-filtered record indices.
    """

    def __init__(
        self,
        dataset,
        n_samples,
        mode="random",
        seed=None,
        **sampling_kwargs
    ):
        self.dataset = dataset

        self.record_ids = sample_record_pairs(
            dataset,
            n_samples,
            mode=mode,
            seed=seed,
            **sampling_kwargs
        )

        labels = dataset.latents[self.record_ids]
        image_indices = dataset.image_indices[self.record_ids]

        images = dataset.get_images_by_indices(
            image_indices
        )

        self.images = torch.from_numpy(
            images.astype(np.float32)
        ).unsqueeze(-1)

        denominators = np.maximum(
            np.asarray(dataset.FACTOR_SIZES) - 1,
            1
        )

        z = (
            labels.astype(np.float32)
            / denominators
        )

        self.z = torch.from_numpy(
            z.reshape(
                len(self.record_ids),
                N_SLOTS * N_FACTORS
            ).astype(np.float32)
        )

    def __len__(self):
        return len(self.z)

    def __getitem__(self, index):
        return (
            self.images[int(index)],
            self.z[int(index)]
        )


# ============================================================
# Cached composition targets
# ============================================================

class CachedCompositionDataset(Dataset):
    """
    Precompute composition targets once.

    Returns:
        z, target

    z contains normalized labels of selected records.

    The composition module should already be on the requested
    device before constructing this dataset.
    """

    def __init__(
        self,
        source,
        composition,
        device=None,
        batch_size=256,
        num_workers=0
    ):
        self.source = source
        self.composition = composition

        if device is None:
            try:
                device = next(
                    composition.parameters()
                ).device
            except (StopIteration, AttributeError):
                device = torch.device("cpu")

        self.device = torch.device(device)

        original_training_state = getattr(
            composition,
            "training",
            None
        )

        if hasattr(composition, "eval"):
            composition.eval()

        targets = []

        loader = DataLoader(
            source,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers
        )

        try:
            with torch.no_grad():
                for images, _ in loader:
                    images = images.to(self.device)

                    # images: [B, 2, 64, 64, 1]
                    target = composition(
                        [
                            images[:, 0],
                            images[:, 1]
                        ]
                    )

                    if isinstance(target, (tuple, list)):
                        raise TypeError(
                            "Expected composition() to return "
                            "a single tensor."
                        )

                    targets.append(
                        target.detach().cpu()
                    )

        finally:
            if (
                original_training_state is not None
                and hasattr(composition, "train")
            ):
                composition.train(
                    original_training_state
                )

        if not targets:
            raise ValueError(
                "Cannot cache composition targets from an empty dataset."
            )

        self.targets = torch.cat(
            targets,
            dim=0
        )

        self.z = source.z.detach().cpu()
        self.record_ids = source.record_ids

        if len(self.z) != len(self.targets):
            raise RuntimeError(
                "Cached target count does not match "
                "the source dataset length."
            )

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, index):
        index = int(index)

        return (
            self.z[index],
            self.targets[index]
        )# ============================================================
# Version 2 (change the Diagonal)
# ============================================================


import numpy as np
import torch

from math import sqrt
from pathlib import Path
from torch.utils.data import Dataset, DataLoader


# ============================================================
# Constants
# ============================================================

# Model-facing factor order:
# [shape, scale, colour, x_position, y_position]

FACTOR_SIZES = (3, 6, 1, 32, 32)
N_FACTORS = 5
N_SLOTS = 2
FIXED_ORIENTATION = 0


# ============================================================
# Continuous-to-discrete label conversion
# ============================================================

def continuous_to_factor_labels(z, factor_sizes=FACTOR_SIZES):
    """
    Convert continuous latent coordinates to discrete dSprites
    factor labels for image lookup.

    Accepts NumPy arrays or PyTorch tensors.

    Input:
        [..., 5]

    Returns:
        NumPy int64 array with shape [..., 5].
    """
    if torch.is_tensor(z):
        z = z.detach().cpu().numpy()

    z = np.asarray(z, dtype=np.float32)
    sizes = np.asarray(factor_sizes, dtype=np.int64)

    if z.ndim == 0 or z.shape[-1] != N_FACTORS:
        raise ValueError(
            f"Expected latent coordinates with final dimension "
            f"{N_FACTORS}, but received shape {z.shape}."
        )

    if sizes.shape != (N_FACTORS,):
        raise ValueError(
            f"factor_sizes must contain exactly {N_FACTORS} values."
        )

    if np.any(sizes < 1):
        raise ValueError("All factor sizes must be at least 1.")

    labels = np.rint(
        np.clip(z, 0.0, 1.0) * (sizes - 1)
    ).astype(np.int64)

    # Colour has only one possible value.
    labels[..., 2] = 0

    return labels


def quantize_latents(z, factor_sizes=FACTOR_SIZES):
    """
    Quantize continuous latent coordinates to normalized discrete
    dSprites factor values.

    Supported shapes:
        [..., 5]  : individual sprite latents
        [..., 10] : flattened two-sprite latents

    Returns the same type/device as the input.
    """
    is_tensor = torch.is_tensor(z)

    if is_tensor:
        original_device = z.device
        original_dtype = z.dtype
        z_array = z.detach().cpu().numpy()
    else:
        z_array = np.asarray(z)

    z_array = np.asarray(z_array, dtype=np.float32)
    original_shape = z_array.shape

    sizes = np.asarray(factor_sizes, dtype=np.int64)

    if sizes.shape != (N_FACTORS,):
        raise ValueError(
            f"factor_sizes must contain exactly {N_FACTORS} values."
        )

    if np.any(sizes < 1):
        raise ValueError("All factor sizes must be at least 1.")

    if z_array.ndim == 0:
        raise ValueError("Latents must have at least one dimension.")

    if z_array.shape[-1] == N_FACTORS:
        pass
    elif z_array.shape[-1] == N_SLOTS * N_FACTORS:
        z_array = z_array.reshape(
            *z_array.shape[:-1],
            N_SLOTS,
            N_FACTORS
        )
    else:
        raise ValueError(
            "Expected final latent dimension "
            f"{N_FACTORS} or {N_SLOTS * N_FACTORS}, "
            f"but received shape {original_shape}."
        )

    labels = np.rint(
        np.clip(z_array, 0.0, 1.0) * (sizes - 1)
    ).astype(np.int64)

    labels[..., 2] = 0

    denominators = np.maximum(sizes - 1, 1)

    quantized = (
        labels.astype(np.float32) /
        denominators.astype(np.float32)
    )

    quantized[..., 2] = 0.0
    quantized = quantized.reshape(original_shape)

    if is_tensor:
        return torch.as_tensor(
            quantized,
            dtype=original_dtype,
            device=original_device
        )

    return quantized.astype(np.float32)


# ============================================================
# dSprites dataset
# ============================================================

class DSpritesDataset:
    """
    dSprites dataset adapted for two-sprite compositional learning.

    Original dSprites latent order:
        [colour, shape, scale, orientation, x, y]

    Model-facing latent order:
        [shape, scale, colour, x, y]

    Orientation is fixed to zero for image lookup.
    Colour is retained as a factor, but always equals zero.
    """

    FACTOR_SIZES = FACTOR_SIZES
    FIXED_ORIENTATION = FIXED_ORIENTATION

    def __init__(self, root):
        root = Path(root)

        if root.is_dir():
            files = list(root.glob("*.npz"))

            if not files:
                raise FileNotFoundError(
                    f"No dSprites .npz file found in {root}"
                )

            data_path = files[0]
        else:
            data_path = root

        data = np.load(data_path, encoding="latin1")

        self.images = data["imgs"]
        original_latents = data["latents_classes"].astype(np.int64)

        # Keep only orientation == 0.
        keep = (
            original_latents[:, 3] == self.FIXED_ORIENTATION
        )

        self.image_indices = np.flatnonzero(keep)
        selected = original_latents[keep]

        # Reorder:
        # [colour, shape, scale, orientation, x, y]
        # to:
        # [shape, scale, colour, x, y]
        self.latents = selected[:, [1, 2, 0, 4, 5]]

        self.label_to_image_index = {
            tuple(label): int(image_index)
            for label, image_index in zip(
                self.latents,
                self.image_indices
            )
        }

        # Map labels to indices in the orientation-filtered dataset.
        self.label_to_record_id = {
            tuple(label): int(record_id)
            for record_id, label in enumerate(self.latents)
        }

        expected = int(np.prod(self.FACTOR_SIZES))

        if len(self.latents) != expected:
            raise ValueError(
                f"Expected {expected} orientation-filtered images, "
                f"but found {len(self.latents)}."
            )

    def __len__(self):
        return len(self.latents)

    def continuous_to_labels(self, z):
        return continuous_to_factor_labels(
            z,
            self.FACTOR_SIZES
        )

    def quantize(self, z):
        return quantize_latents(
            z,
            self.FACTOR_SIZES
        )

    def get_image_indices(self, factor_labels):
        labels = np.asarray(
            factor_labels,
            dtype=np.int64
        )

        if labels.ndim == 1:
            labels = labels[None, :]

        if labels.ndim != 2 or labels.shape[1] != N_FACTORS:
            raise ValueError(
                f"Expected factor labels with shape [N, {N_FACTORS}]."
            )

        indices = []

        for label in labels:
            key = tuple(label.tolist())

            if key not in self.label_to_image_index:
                raise ValueError(
                    f"Invalid dSprites factor combination: {key}"
                )

            indices.append(self.label_to_image_index[key])

        return np.asarray(indices, dtype=np.int64)

    def get_images_by_indices(self, indices):
        indices = np.asarray(indices, dtype=np.int64)

        if indices.ndim == 0:
            return self.images[int(indices)]

        return self.images[indices]

    def get_images(self, factor_labels):
        indices = self.get_image_indices(factor_labels)
        return self.get_images_by_indices(indices)

    def get_images_from_continuous(self, z):
        labels = self.continuous_to_labels(z)
        return self.get_images(labels)

    def __getitem__(self, index):
        image = self.images[
            int(self.image_indices[int(index)])
        ]

        label = self.latents[int(index)]

        return image, label


# ============================================================
# Gap constraints
# ============================================================

def _normalise_gap_constraints(gap=None, gaps=None, gap_dims=None):
    """
    Minimum between-sprite separation, in normalized label coordinates.

    This preserves the existing gap-sampling definition.
    """
    if gaps is not None:
        gap = gaps

    if gap is None:
        return {}

    if isinstance(gap, dict):
        return {
            int(dim): float(value)
            for dim, value in gap.items()
        }

    if np.isscalar(gap):
        if gap_dims is None:
            gap_dims = [0, 1, 3, 4]

        return {
            int(dim): float(gap)
            for dim in gap_dims
        }

    values = list(gap)

    if gap_dims is None:
        gap_dims = range(len(values))

    return {
        int(dim): float(values[dim])
        for dim in gap_dims
        if dim < len(values) and values[dim] is not None
    }


# ============================================================
# Original-style diagonal sampling
# ============================================================

def _sample_diagonal_continuous(
    n_samples,
    delta,
    rng,
    max_attempts=100000,
):
    """
    Adaptation of the original diagonal sampler.

    For every pair:

        z_base ~ Uniform([0, 1]^5)

        z_on_diag = [z_base, z_base]

        z_pair = z_on_diag + orthogonal_noise * radius

    The perturbation is perpendicular to the diagonal, so the
    two slots move in opposite directions.

    The perturbation is generated independently for each factor,
    following the structure of the original implementation.

    Returns:
        [N, 2, 5] continuous latent coordinates.

    Note:
        dSprites colour is fixed, so colour is reset to zero.
        The other four factors are perturbed.
    """
    max_delta = sqrt(2) / 2

    if (
        not np.isfinite(delta)
        or delta <= 0
        or delta >= max_delta
    ):
        raise ValueError(
            f"delta must be greater than 0 and smaller than "
            f"{max_delta:.6f}, but got {delta}."
        )

    if n_samples == 0:
        return np.empty(
            (0, N_SLOTS, N_FACTORS),
            dtype=np.float32
        )

    samples = []
    filled = 0
    rejected = 0

    while filled < n_samples:
        remaining = n_samples - filled

        count = min(
            max(4 * remaining, 256),
            8192,
            max_attempts - rejected
        )

        if count <= 0:
            raise ValueError(
                "Could not sample enough diagonal points within "
                "max_attempts. Check delta."
            )

        # Shared base point on the diagonal.
        z_in_slot = rng.random(
            (count, N_FACTORS)
        ).astype(np.float32)

        z_on_diag = np.repeat(
            z_in_slot[:, None, :],
            N_SLOTS,
            axis=1
        )

        # Generate noise using the same construction as the
        # original implementation.
        noise = rng.normal(
            size=(count, N_SLOTS + 2, N_FACTORS)
        ).astype(np.float32)

        noise_norm = np.linalg.norm(
            noise,
            axis=1,
            keepdims=True
        )

        noise = noise / np.maximum(
            noise_norm,
            1e-12
        )

        # Remove the two extra vectors.
        noise = noise[:, :N_SLOTS, :]

        # Project noise onto the hyperplane perpendicular
        # to the diagonal.
        numerator = (
            noise * z_on_diag
        ).sum(axis=1, keepdims=True)

        denominator = (
            z_on_diag ** 2
        ).sum(axis=1, keepdims=True)

        ort_vec = (
            noise
            - z_on_diag
            * numerator
            / np.maximum(denominator, 1e-12)
        )

        # Normalize the perturbation direction per factor.
        ort_norm = np.linalg.norm(
            ort_vec,
            axis=1,
            keepdims=True
        )

        ort_vec = ort_vec / np.maximum(
            ort_norm,
            1e-12
        )

        # For two slots, the exponent is 1 / (2 - 1) = 1.
        # This follows the original radius sampling.
        radius = (
            rng.random((count, 1, N_FACTORS))
            ** (1.0 / (N_SLOTS - 1))
        ) * delta

        z_pair = z_on_diag + ort_vec * radius

        # dSprites colour has only one possible value.
        z_pair[:, :, 2] = 0.0

        # Reject samples outside the unit cube.
        valid = np.all(
            (z_pair >= 0.0) & (z_pair <= 1.0),
            axis=(1, 2)
        )

        accepted = z_pair[valid]

        if len(accepted) > 0:
            accepted = accepted[:remaining]
            samples.append(accepted)
            filled += len(accepted)
            rejected = 0
        else:
            rejected += count

        if rejected >= max_attempts:
            raise ValueError(
                "No valid diagonal samples accepted within "
                "max_attempts. Try a smaller delta."
            )

    return np.concatenate(samples, axis=0).astype(np.float32)


def _sample_diagonal_record_pairs(
    dataset,
    n_samples,
    delta,
    rng,
    max_attempts=100000,
):
    """
    Sample continuous diagonal latents, quantize them to dSprites
    labels, and convert those labels to dataset record IDs.

    Returns:
        [N, 2] indices into dataset.latents.
    """
    continuous = _sample_diagonal_continuous(
        n_samples=n_samples,
        delta=delta,
        rng=rng,
        max_attempts=max_attempts
    )

    if n_samples == 0:
        return np.empty((0, N_SLOTS), dtype=np.int64)

    labels = dataset.continuous_to_labels(
        continuous.reshape(-1, N_FACTORS)
    )

    labels = labels.reshape(
        n_samples,
        N_SLOTS,
        N_FACTORS
    )

    pairs = np.empty(
        (n_samples, N_SLOTS),
        dtype=np.int64
    )

    for i in range(n_samples):
        for slot in range(N_SLOTS):
            key = tuple(labels[i, slot].tolist())

            if key not in dataset.label_to_record_id:
                raise RuntimeError(
                    f"Sampled dSprites label does not exist: {key}"
                )

            pairs[i, slot] = dataset.label_to_record_id[key]

    return pairs


# ============================================================
# Sampling existing sprite records
# ============================================================

def sample_record_pairs(
    dataset,
    n_samples,
    mode="random",
    seed=None,
    planes=None,
    delta=0.1,
    gap=None,
    gaps=None,
    gap_dims=None,
    grid_size=10,
    max_points=1000000,
    max_attempts=100000,
):
    """
    Return [N, 2] indices into dataset.latents.

    Modes
    -----
    random:
        Independently sample two records uniformly, with replacement.

    orthogonal:
        Equal-probability planes, each varying one slot uniformly.
        Example:
            planes=[
                {"slot": 0, "record_id": 7},
                {"slot": 1, "record_id": 7}
            ]

    diagonal:
        Original-style diagonal sampling:
        sample a shared continuous base vector, perturb the two slots
        in opposite directions perpendicular to the diagonal, reject
        out-of-cube points, then quantize to dSprites labels.

    random_gap / orthogonal_gap:
        Additionally enforce minimum separation on selected factors.

    grid:
        Uniformly sample records whose labels lie on a discrete grid.

    full_grid / face_grid:
        Sample pairs without replacement.
        If n_samples=None, enumerate all pairs.

    Record IDs refer to the orientation-filtered dataset, not the
    original NPZ row indices.
    """
    modes = {
        "random",
        "random_gap",
        "orthogonal",
        "orthogonal_gap",
        "diagonal",
        "grid",
        "full_grid",
        "face_grid"
    }

    if mode not in modes:
        raise ValueError(
            f"Unknown sampling mode {mode!r}. "
            f"Available modes: {sorted(modes)}"
        )

    if n_samples is None:
        if mode not in ("full_grid", "face_grid"):
            raise ValueError(
                "n_samples is required for this sampling mode."
            )
    elif (
        not isinstance(n_samples, (int, np.integer))
        or n_samples < 0
    ):
        raise ValueError(
            "n_samples must be a non-negative integer."
        )

    if len(dataset) == 0:
        raise ValueError("Cannot sample an empty dataset.")

    if max_attempts < 1:
        raise ValueError("max_attempts must be positive.")

    rng = np.random.default_rng(seed)

    labels = dataset.latents
    sizes = np.asarray(dataset.FACTOR_SIZES)

    normalized = (
        labels / np.maximum(sizes - 1, 1)
    )

    # --------------------------------------------------------
    # Diagonal sampling
    # --------------------------------------------------------

    if mode == "diagonal":
        if n_samples == 0:
            return np.empty(
                (0, N_SLOTS),
                dtype=np.int64
            )

        return _sample_diagonal_record_pairs(
            dataset=dataset,
            n_samples=n_samples,
            delta=delta,
            rng=rng,
            max_attempts=max_attempts
        )

    # --------------------------------------------------------
    # Grid sampling
    # --------------------------------------------------------

    if mode in ("grid", "full_grid", "face_grid"):
        if (
            not isinstance(grid_size, (int, np.integer))
            or grid_size < 1
        ):
            raise ValueError(
                "grid_size must be a positive integer."
            )

        keep = np.ones(
            len(dataset),
            dtype=bool
        )

        for dim, size in enumerate(sizes):
            levels = np.rint(
                np.linspace(
                    0,
                    size - 1,
                    min(grid_size, size)
                )
            ).astype(int)

            keep &= np.isin(
                labels[:, dim],
                levels
            )

        if mode == "face_grid":
            keep &= (
                np.count_nonzero(labels, axis=1) <= 2
            )

        records = np.flatnonzero(keep)

        if len(records) == 0:
            raise ValueError(
                "No existing records satisfy the grid settings."
            )

        if mode == "grid":
            return rng.choice(
                records,
                size=(n_samples, N_SLOTS)
            )

        total = len(records) ** 2

        count = (
            total
            if n_samples is None
            else n_samples
        )

        if count > total:
            raise ValueError(
                "Requested more samples than the grid contains."
            )

        if count > max_points:
            raise ValueError(
                f"Requested {count:,} pairs. "
                f"Maximum allowed is {max_points:,}."
            )

        selected = (
            np.arange(total)
            if n_samples is None
            else rng.choice(
                total,
                size=count,
                replace=False
            )
        )

        return np.column_stack(
            (
                records[selected // len(records)],
                records[selected % len(records)]
            )
        )

    # --------------------------------------------------------
    # Orthogonal sampling setup
    # --------------------------------------------------------

    anchors = []

    if mode in ("orthogonal", "orthogonal_gap"):
        if planes is None:
            planes = [
                {"slot": 0, "record_id": 0},
                {"slot": 1, "record_id": 0}
            ]

        if not planes:
            raise ValueError(
                "At least one orthogonal plane is required."
            )

        for plane in planes:
            slot = plane["slot"]

            if slot not in (0, 1):
                raise ValueError(
                    "Orthogonal plane slot must be 0 or 1."
                )

            if "record_id" in plane:
                record_id = plane["record_id"]

                if (
                    not isinstance(record_id, (int, np.integer))
                    or not 0 <= record_id < len(dataset)
                ):
                    raise ValueError(
                        "Orthogonal record_id must identify "
                        "an existing record."
                    )
            else:
                fixed = np.asarray(
                    plane["fixed"],
                    dtype=float
                )

                if fixed.shape != (N_FACTORS,):
                    raise ValueError(
                        "Each plane must specify five fixed coordinates."
                    )

                matches = np.flatnonzero(
                    np.all(
                        np.isclose(
                            normalized,
                            fixed,
                            rtol=0,
                            atol=1e-7
                        ),
                        axis=1
                    )
                )

                if len(matches) != 1:
                    raise ValueError(
                        "Fixed coordinates must identify an existing "
                        "record; use record_id."
                    )

                record_id = int(matches[0])

            anchors.append((slot, record_id))

    # --------------------------------------------------------
    # Gap constraints
    # --------------------------------------------------------

    constraints = {}

    if mode in ("random_gap", "orthogonal_gap"):
        constraints = _normalise_gap_constraints(
            gap,
            gaps,
            gap_dims
        )

        for dim, minimum in constraints.items():
            if (
                not 0 <= dim < N_FACTORS
                or not np.isfinite(minimum)
                or not 0 <= minimum <= 1
            ):
                raise ValueError(
                    "Gap constraints require valid factor dimensions "
                    "and distances in [0, 1]."
                )

            if sizes[dim] == 1 and minimum > 0:
                raise ValueError(
                    f"Cannot enforce a positive gap on fixed factor {dim}."
                )

    if n_samples == 0:
        return np.empty(
            (0, N_SLOTS),
            dtype=np.int64
        )

    # --------------------------------------------------------
    # Random / orthogonal / gap sampling
    # --------------------------------------------------------

    result = np.empty(
        (n_samples, N_SLOTS),
        dtype=np.int64
    )

    filled = 0
    rejected = 0

    while filled < n_samples:
        count = min(
            max(4 * (n_samples - filled), 256),
            8192,
            max_attempts - rejected
        )

        if count <= 0:
            raise ValueError(
                "No record pairs accepted within max_attempts; "
                "check sampling constraints."
            )

        pairs = rng.integers(
            len(dataset),
            size=(count, N_SLOTS)
        )

        if anchors:
            choices = rng.integers(
                len(anchors),
                size=count
            )

            for i, (slot, record_id) in enumerate(anchors):
                pairs[
                    choices == i,
                    1 - slot
                ] = record_id

        keep = np.ones(
            count,
            dtype=bool
        )

        for dim, minimum in constraints.items():
            separation = np.abs(
                normalized[pairs[:, 0], dim]
                - normalized[pairs[:, 1], dim]
            )

            keep &= (
                separation + 1e-12 >= minimum
            )

        accepted = pairs[keep][
            :n_samples - filled
        ]

        result[
            filled:filled + len(accepted)
        ] = accepted

        filled += len(accepted)

        if len(accepted) > 0:
            rejected = 0
        else:
            rejected += count

        if rejected >= max_attempts:
            raise ValueError(
                "No record pairs accepted within max_attempts; "
                "check sampling constraints."
            )

    return result


# ============================================================
# Two-sprite dataset
# ============================================================

class DSpritesTwoSpriteDataset(Dataset):
    """
    Select existing sprite pairs and return (images, normalized labels).

    images: [2, 64, 64, 1]
    z:      [10]

    record_ids stores the selected orientation-filtered record indices.
    """

    def __init__(
        self,
        dataset,
        n_samples,
        mode="random",
        seed=None,
        **sampling_kwargs
    ):
        self.dataset = dataset

        self.record_ids = sample_record_pairs(
            dataset,
            n_samples,
            mode=mode,
            seed=seed,
            **sampling_kwargs
        )

        labels = dataset.latents[self.record_ids]
        image_indices = dataset.image_indices[self.record_ids]

        images = dataset.get_images_by_indices(
            image_indices
        )

        self.images = torch.from_numpy(
            images.astype(np.float32)
        ).unsqueeze(-1)

        denominators = np.maximum(
            np.asarray(dataset.FACTOR_SIZES) - 1,
            1
        )

        z = (
            labels.astype(np.float32)
            / denominators
        )

        self.z = torch.from_numpy(
            z.reshape(
                len(self.record_ids),
                N_SLOTS * N_FACTORS
            ).astype(np.float32)
        )

    def __len__(self):
        return len(self.z)

    def __getitem__(self, index):
        return (
            self.images[int(index)],
            self.z[int(index)]
        )


# ============================================================
# Cached composition targets
# ============================================================

class CachedCompositionDataset(Dataset):
    """
    Precompute composition targets once.

    Returns:
        z, target

    z contains normalized labels of selected records.

    The composition module should already be on the requested
    device before constructing this dataset.
    """

    def __init__(
        self,
        source,
        composition,
        device=None,
        batch_size=256,
        num_workers=0
    ):
        self.source = source
        self.composition = composition

        if device is None:
            try:
                device = next(
                    composition.parameters()
                ).device
            except (StopIteration, AttributeError):
                device = torch.device("cpu")

        self.device = torch.device(device)

        original_training_state = getattr(
            composition,
            "training",
            None
        )

        if hasattr(composition, "eval"):
            composition.eval()

        targets = []

        loader = DataLoader(
            source,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers
        )

        try:
            with torch.no_grad():
                for images, _ in loader:
                    images = images.to(self.device)

                    # images: [B, 2, 64, 64, 1]
                    target = composition(
                        [
                            images[:, 0],
                            images[:, 1]
                        ]
                    )

                    if isinstance(target, (tuple, list)):
                        raise TypeError(
                            "Expected composition() to return "
                            "a single tensor."
                        )

                    targets.append(
                        target.detach().cpu()
                    )

        finally:
            if (
                original_training_state is not None
                and hasattr(composition, "train")
            ):
                composition.train(
                    original_training_state
                )

        if not targets:
            raise ValueError(
                "Cannot cache composition targets from an empty dataset."
            )

        self.targets = torch.cat(
            targets,
            dim=0
        )

        self.z = source.z.detach().cpu()
        self.record_ids = source.record_ids

        if len(self.z) != len(self.targets):
            raise RuntimeError(
                "Cached target count does not match "
                "the source dataset length."
            )

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, index):
        index = int(index)

        return (
            self.z[index],
            self.targets[index]
        )