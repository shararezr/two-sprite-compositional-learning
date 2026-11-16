import numpy as np
import torch

from pathlib import Path
from itertools import product
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

    Accepts:
        NumPy arrays or PyTorch tensors.

    Input:
        [..., 5]

    Returns:
        NumPy int64 array with shape [..., 5].

    The continuous coordinates themselves are not modified.
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
    Quantize continuous latent coordinates to the normalized
    discrete dSprites factor values.

    This is the model-input counterpart to
    continuous_to_factor_labels().

    Supported shapes:
        [..., 5]  : individual sprite latents
        [..., 10] : flattened two-sprite latents

    Returns:
        A NumPy array or torch.Tensor with the same shape and
        corresponding type/device as the input.

    Example:
        A continuous scale coordinate of 0.43 is converted to
        the nearest valid normalized scale coordinate:

            label = round(0.43 * (6 - 1)) = 2
            model_input = 2 / (6 - 1) = 0.4

    Fixed colour is always zero.
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

    # Support flattened two-slot latents as well as individual slots.
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

    # Convert continuous coordinates to discrete labels.
    labels = np.rint(
        np.clip(z_array, 0.0, 1.0) * (sizes - 1)
    ).astype(np.int64)

    # Colour has only one possible value.
    labels[..., 2] = 0

    # Convert integer labels back to normalized model inputs.
    denominators = np.maximum(sizes - 1, 1)

    quantized = (
        labels.astype(np.float32) /
        denominators.astype(np.float32)
    )

    quantized[..., 2] = 0.0

    # Restore the original shape.
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

        expected = int(np.prod(self.FACTOR_SIZES))

        if len(self.latents) != expected:
            raise ValueError(
                f"Expected {expected} orientation-filtered images, "
                f"but found {len(self.latents)}."
            )

    def __len__(self):
        return len(self.latents)

    def continuous_to_labels(self, z):
        """
        Convert continuous coordinates to discrete labels
        for image lookup only.

        z:
            [..., 5]

        Returns:
            [..., 5] NumPy integer labels.
        """
        return continuous_to_factor_labels(
            z,
            self.FACTOR_SIZES
        )

    def quantize(self, z):
        """
        Convert continuous coordinates to normalized discrete
        factor coordinates for model input.
        """
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

            indices.append(
                self.label_to_image_index[key]
            )

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
# Continuous sampling utilities
# ============================================================

def _sample_random(n_samples, factor_sizes=None, rng=None, **kwargs):
    """
    Uniform continuous sampling in [0, 1]^10.

    Returns:
        [n_samples, 2, 5]
    """
    rng = np.random.default_rng() if rng is None else rng

    z = rng.uniform(
        0.0,
        1.0,
        size=(n_samples, 2, N_FACTORS)
    )

    # Colour has only one value.
    z[:, :, 2] = 0.0

    return z.astype(np.float32)


# ============================================================
# Random sampling with gaps
# ============================================================

def _normalise_gap_constraints(
    gap=None,
    gaps=None,
    gap_dims=None
):
    """
    Interpret gap constraints in normalized [0, 1] coordinates.

    Examples:
        gap=0.2
        gaps={0: 0.2, 1: 0.1}
        gaps=[0.2, 0.1, 0.0, 0.2, 0.2]

    Dimensions refer to the five factors within each sprite.

    By default, the fixed colour dimension is excluded.
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
            gap_dims = [
                dim for dim in range(N_FACTORS)
                if FACTOR_SIZES[dim] > 1
            ]

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


def _satisfies_gap(z0, z1, constraints):
    for dim, minimum_gap in constraints.items():

        if not 0 <= dim < N_FACTORS:
            raise ValueError(
                f"Invalid gap dimension: {dim}"
            )

        if FACTOR_SIZES[dim] == 1 and minimum_gap > 0:
            raise ValueError(
                f"Cannot enforce a positive gap on fixed factor "
                f"dimension {dim}."
            )

        if abs(z0[dim] - z1[dim]) < minimum_gap:
            return False

    return True


def _sample_random_with_gap(
    n_samples,
    factor_sizes=None,
    rng=None,
    gap=None,
    gaps=None,
    gap_dims=None,
    max_attempts=100000,
    **kwargs
):
    """
    Uniformly sample continuous pairs subject to minimum
    separation constraints.
    """
    rng = np.random.default_rng() if rng is None else rng

    constraints = _normalise_gap_constraints(
        gap=gap,
        gaps=gaps,
        gap_dims=gap_dims
    )

    result = np.empty(
        (n_samples, 2, N_FACTORS),
        dtype=np.float32
    )

    for i in range(n_samples):

        for _ in range(max_attempts):

            pair = rng.uniform(
                0.0,
                1.0,
                size=(2, N_FACTORS)
            )

            pair[:, 2] = 0.0

            if _satisfies_gap(
                pair[0],
                pair[1],
                constraints
            ):
                result[i] = pair
                break

        else:
            raise RuntimeError(
                "Could not satisfy random gap constraints. "
                "Reduce the gap or increase max_attempts."
            )

    return result


# ============================================================
# Orthogonal sampling
# ============================================================

def _sample_orthogonal(
    n_samples,
    factor_sizes=None,
    rng=None,
    planes=None,
    **kwargs
):
    """
    Sample from orthogonal planes.

    A plane specifies:
        - which sprite slot varies;
        - the fixed continuous coordinates of the other slot.
    """
    rng = np.random.default_rng() if rng is None else rng

    if planes is None:
        zero = [0.0] * N_FACTORS

        planes = [
            {"slot": 0, "fixed": zero},
            {"slot": 1, "fixed": zero}
        ]

    result = np.empty(
        (n_samples, 2, N_FACTORS),
        dtype=np.float32
    )

    for i in range(n_samples):

        plane = planes[
            int(rng.integers(len(planes)))
        ]

        slot = int(plane["slot"])

        if slot not in (0, 1):
            raise ValueError(
                "Orthogonal plane slot must be 0 or 1."
            )

        fixed = np.asarray(
            plane["fixed"],
            dtype=np.float32
        )

        if fixed.shape != (N_FACTORS,):
            raise ValueError(
                "Each plane must specify five fixed coordinates."
            )

        if np.any(fixed < 0.0) or np.any(fixed > 1.0):
            raise ValueError(
                "Fixed coordinates must be in [0, 1]."
            )

        pair = rng.uniform(
            0.0,
            1.0,
            size=(2, N_FACTORS)
        )

        pair[1 - slot] = fixed

        # Colour is constant.
        pair[:, 2] = 0.0

        result[i] = pair

    return result


def _sample_orthogonal_with_gap(
    n_samples,
    factor_sizes=None,
    rng=None,
    planes=None,
    gap=None,
    gaps=None,
    gap_dims=None,
    max_attempts=100000,
    **kwargs
):
    """
    Orthogonal sampling with additional gap constraints.
    """
    rng = np.random.default_rng() if rng is None else rng

    constraints = _normalise_gap_constraints(
        gap=gap,
        gaps=gaps,
        gap_dims=gap_dims
    )

    result = np.empty(
        (n_samples, 2, N_FACTORS),
        dtype=np.float32
    )

    for i in range(n_samples):

        for _ in range(max_attempts):

            pair = _sample_orthogonal(
                1,
                rng=rng,
                planes=planes
            )[0]

            if _satisfies_gap(
                pair[0],
                pair[1],
                constraints
            ):
                result[i] = pair
                break

        else:
            raise RuntimeError(
                "Could not find an orthogonal sample satisfying "
                "the gap constraints."
            )

    return result


# ============================================================
# Diagonal sampling
# ============================================================

def _sample_diagonal(
    n_samples,
    factor_sizes=None,
    rng=None,
    delta=0.2,
    max_attempts=100000,
    **kwargs
):
    """
    Sample around the diagonal subspace:

        z_0 == z_1

    The orthogonal perturbation has the form:

        z_0 = base + perturbation / sqrt(2)
        z_1 = base - perturbation / sqrt(2)
    """
    rng = np.random.default_rng() if rng is None else rng

    if delta < 0:
        raise ValueError("delta must be non-negative.")

    result = np.empty(
        (n_samples, 2, N_FACTORS),
        dtype=np.float32
    )

    active_dims = [
        d for d in range(N_FACTORS)
        if FACTOR_SIZES[d] > 1
    ]

    dimension = len(active_dims)

    for i in range(n_samples):

        for _ in range(max_attempts):

            # Base point on the diagonal.
            base = rng.uniform(
                0.0,
                1.0,
                size=N_FACTORS
            )

            base[2] = 0.0

            # Random direction in the active orthogonal space.
            direction = rng.normal(
                size=dimension
            )

            norm = np.linalg.norm(direction)

            if norm == 0:
                continue

            direction = direction / norm

            # Uniform radius within a dimension-dimensional ball.
            radius = delta * (
                rng.uniform() ** (1.0 / dimension)
            )

            perturbation = np.zeros(
                N_FACTORS,
                dtype=np.float32
            )

            perturbation[active_dims] = (
                radius * direction
            )

            z0 = base + perturbation / np.sqrt(2.0)
            z1 = base - perturbation / np.sqrt(2.0)

            z0[2] = 0.0
            z1[2] = 0.0

            if (
                np.all(z0 >= 0.0)
                and np.all(z0 <= 1.0)
                and np.all(z1 >= 0.0)
                and np.all(z1 <= 1.0)
            ):
                result[i, 0] = z0
                result[i, 1] = z1
                break

        else:
            raise RuntimeError(
                "Could not sample a valid diagonal point. "
                "Try reducing delta."
            )

    return result


# ============================================================
# Grid utilities
# ============================================================

def _get_grid(factor_sizes=None, grid_size=10):
    """
    Generate evenly spaced continuous grid levels in [0, 1].
    """
    if grid_size < 1:
        raise ValueError("grid_size must be at least 1.")

    return [
        np.linspace(
            0.0,
            1.0,
            int(grid_size)
        )
        if dim != 2
        else np.array([0.0])
        for dim in range(N_FACTORS)
    ]


def _get_face_grids(
    factor_sizes=None,
    grid_size=10
):
    """
    Generate two-dimensional faces of the continuous latent cube.

    Two dimensions vary; the other dimensions are fixed at zero.
    """
    levels = _get_grid(
        factor_sizes,
        grid_size
    )

    faces = []

    for dim_a in range(N_FACTORS):

        for dim_b in range(dim_a + 1, N_FACTORS):

            face = []

            for value_a, value_b in product(
                levels[dim_a],
                levels[dim_b]
            ):
                point = np.zeros(
                    N_FACTORS,
                    dtype=np.float32
                )

                point[dim_a] = value_a
                point[dim_b] = value_b

                face.append(point)

            faces.append(
                np.asarray(face, dtype=np.float32)
            )

    return faces


# ============================================================
# Grid sampling
# ============================================================

def _sample_grid(
    n_samples,
    factor_sizes=None,
    rng=None,
    grid_size=10,
    **kwargs
):
    """
    Sample continuous values from evenly spaced grid levels.
    """
    rng = np.random.default_rng() if rng is None else rng

    levels = _get_grid(
        factor_sizes,
        grid_size
    )

    result = np.zeros(
        (n_samples, 2, N_FACTORS),
        dtype=np.float32
    )

    for i in range(n_samples):

        for slot in range(2):

            for dim in range(N_FACTORS):

                result[i, slot, dim] = rng.choice(
                    levels[dim]
                )

    return result


def _sample_full_grid(
    n_samples=None,
    factor_sizes=None,
    grid_size=10,
    max_points=1000000,
    **kwargs
):
    """
    Enumerate the complete Cartesian grid for both sprites.
    """
    levels = _get_grid(
        factor_sizes,
        grid_size
    )

    single_grid = np.asarray(
        list(product(*levels)),
        dtype=np.float32
    )

    total = len(single_grid) ** 2

    if total > max_points:
        raise ValueError(
            f"Full grid contains {total:,} pairs. "
            f"Maximum allowed is {max_points:,}."
        )

    pairs = np.asarray(
        list(product(single_grid, repeat=2)),
        dtype=np.float32
    )

    if n_samples is not None:

        if n_samples > len(pairs):
            raise ValueError(
                "Requested more samples than the full grid contains."
            )

        pairs = pairs[:n_samples]

    return pairs


def _sample_face_grid(
    n_samples=None,
    factor_sizes=None,
    rng=None,
    grid_size=10,
    max_points=1000000,
    **kwargs
):
    """
    Sample or enumerate pairs from the union of two-dimensional
    grid faces.
    """
    rng = np.random.default_rng() if rng is None else rng

    faces = _get_face_grids(
        factor_sizes,
        grid_size
    )

    points = np.unique(
        np.concatenate(faces, axis=0),
        axis=0
    )

    total = len(points) ** 2

    if total > max_points:
        raise ValueError(
            f"Face grid contains {total:,} pairs. "
            f"Maximum allowed is {max_points:,}."
        )

    if n_samples is None:
        return np.asarray(
            list(product(points, repeat=2)),
            dtype=np.float32
        )

    if n_samples > total:
        raise ValueError(
            "Requested more samples than the face grid contains."
        )

    selected = rng.choice(
        total,
        size=n_samples,
        replace=False
    )

    result = np.empty(
        (n_samples, 2, N_FACTORS),
        dtype=np.float32
    )

    for i, index in enumerate(selected):

        a = index // len(points)
        b = index % len(points)

        result[i, 0] = points[a]
        result[i, 1] = points[b]

    return result


# ============================================================
# Main continuous sampling interface
# ============================================================

def sample_latents(
    n_samples,
    n_slots=2,
    slot_dim=5,
    mode="random",
    factor_sizes=FACTOR_SIZES,
    seed=None,
    **kwargs
):
    """
    Sample continuous latent coordinates.

    Returns:
        [n_samples, n_slots * slot_dim]

    Values remain continuous. Quantization is applied later
    when constructing the dSprites dataset.
    """
    if n_samples < 0:
        raise ValueError("n_samples must be non-negative.")

    if n_slots != 2 or slot_dim != 5:
        raise ValueError(
            "This implementation expects two slots "
            "with five factors each."
        )

    rng = np.random.default_rng(seed)

    samplers = {
        "random": _sample_random,
        "random_gap": _sample_random_with_gap,
        "orthogonal": _sample_orthogonal,
        "orthogonal_gap": _sample_orthogonal_with_gap,
        "diagonal": _sample_diagonal,
        "grid": _sample_grid,
        "full_grid": _sample_full_grid,
        "face_grid": _sample_face_grid
    }

    if mode not in samplers:
        raise ValueError(
            f"Unknown sampling mode '{mode}'. "
            f"Available modes: {list(samplers.keys())}"
        )

    sampler = samplers[mode]

    samples = sampler(
        n_samples=n_samples,
        factor_sizes=factor_sizes,
        rng=rng,
        **kwargs
    )

    samples = np.asarray(
        samples,
        dtype=np.float32
    )

    if samples.ndim != 3 or samples.shape[1:] != (2, 5):
        raise ValueError(
            "Sampler must return an array of shape [N, 2, 5]."
        )

    return torch.from_numpy(
        samples.reshape(len(samples), -1)
    ).float()


# ============================================================
# Two-sprite dataset
# ============================================================

class DSpritesTwoSpriteDataset(Dataset):
    """
    Returns:
        images: [2, 64, 64, 1]
        z:      [10]

    z contains normalized, quantized dSprites factor coordinates.

    continuous_z retains the original continuous samples for
    diagnostics and analysis.

    Image lookup and model inputs therefore use matching
    discrete factor configurations.
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

        # Sample original continuous latent coordinates.
        self.continuous_z = sample_latents(
            n_samples=n_samples,
            n_slots=2,
            slot_dim=5,
            mode=mode,
            factor_sizes=dataset.FACTOR_SIZES,
            seed=seed,
            **sampling_kwargs
        )

        # Reshape to [N, 2, 5] for image lookup.
        continuous = self.continuous_z.reshape(
            len(self.continuous_z),
            2,
            5
        )

        # Convert continuous coordinates to discrete image labels.
        labels = continuous_to_factor_labels(
            continuous.reshape(-1, 5),
            dataset.FACTOR_SIZES
        )

        # Retrieve images corresponding to those labels.
        image_indices = dataset.get_image_indices(
            labels
        )

        images = dataset.get_images_by_indices(
            image_indices
        )

        images = images.reshape(
            len(self.continuous_z),
            2,
            64,
            64
        )

        self.images = torch.from_numpy(
            images.astype(np.float32)
        ).unsqueeze(-1)

        # Quantize the model inputs to match the target images.
        self.z = quantize_latents(
            self.continuous_z,
            dataset.FACTOR_SIZES
        ).float()

    def __len__(self):
        return len(self.z)

    def __getitem__(self, index):
        index = int(index)

        return (
            self.images[index],
            self.z[index]
        )


# ============================================================
# Cached composition targets
# ============================================================

class CachedCompositionDataset(Dataset):
    """
    Precompute composition targets once.

    Returns:
        z, target

    z is the quantized model input.

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
