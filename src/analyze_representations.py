
import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F


# ============================================================
# 0. Device and experiment configuration
# ============================================================

FACTOR_NAMES = [
    "shape", "scale", "colour", "x_position", "y_position"
]

SLOT_FACTOR_INDICES = {
    "slot1": [0, 1, 3, 4],
    "slot2": [5, 6, 8, 9],
}

ALL_FACTOR_INDICES = (
    SLOT_FACTOR_INDICES["slot1"]
    + SLOT_FACTOR_INDICES["slot2"]
)

FACTOR_CARDINALITIES = {
    "shape": 3,
    "scale": 6,
    "x_position": 32,
    "y_position": 32,
}

LEVELS = ["initial", "intermediate", "final"]

WHITENING_EPSILON = 1e-9
PROBE_RANK_TOLERANCE = 1e-12
SUBSPACE_RANK_TOLERANCE = 1e-6

# sklearn LogisticRegression default C=1.
LOGISTIC_REGULARISATION_C = 1.0


# ============================================================
# 1. General utilities
# ============================================================

def save_csv(path, rows):

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if not rows:
        print(f"Warning: no rows to save to {path}")
        return

    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def scalar(x):
    return float(x.detach().cpu().item())


def factor_name(global_index):

    slot = 1 if global_index < 5 else 2
    local_index = global_index % 5

    return f"slot{slot}_{FACTOR_NAMES[local_index]}"


def factor_short_name(global_index):
    return FACTOR_NAMES[global_index % 5]


def load_representation(path):

    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(path)

    required = {
        "representations",
        "latents",
        "labels",
        "factor_names",
        "sample_ids",
    }

    with np.load(path, allow_pickle=False) as archive:

        missing = required - set(archive.files)

        if missing:
            raise ValueError(
                f"{path} is missing keys: {sorted(missing)}"
            )

        data = {
            key: archive[key].copy()
            for key in required
        }

    H = np.asarray(data["representations"], dtype=np.float64)
    z = np.asarray(data["latents"], dtype=np.float64)
    labels = np.asarray(data["labels"])
    sample_ids = data["sample_ids"]

    if H.ndim != 2:
        raise ValueError(f"{path}: representations must be [N,D].")

    n = H.shape[0]

    if H.shape[1] == 0 or n < 3:
        raise ValueError(f"{path}: insufficient representation data.")

    if z.shape != (n, 10) or labels.shape != (n, 10):
        raise ValueError(f"{path}: expected latents and labels [N,10].")

    if sample_ids.shape != (n,):
        raise ValueError(f"{path}: sample_ids must be [N].")

    if not all(np.all(np.isfinite(x)) for x in (H, z, labels)):
        raise ValueError(f"{path}: non-finite values detected.")

    if not np.allclose(labels, np.round(labels)):
        raise ValueError(f"{path}: labels must be integer-valued.")

    labels = np.round(labels).astype(np.int64)

    if len(np.unique(sample_ids)) != n:
        raise ValueError(f"{path}: sample IDs are not unique.")

    if np.std(H) <= 1e-12:
        raise ValueError(f"{path}: representation has no variation.")

    sizes = np.array([3, 6, 1, 32, 32] * 2)

    if np.any(labels < 0) or np.any(labels >= sizes):
        raise ValueError(f"{path}: factor labels outside expected ranges.")

    if not np.all(labels[:, [2, 7]] == 0):
        raise ValueError(f"{path}: colour should be fixed at zero.")

    expected_labels = np.rint(
        np.clip(z, 0.0, 1.0) * (sizes - 1)
    ).astype(np.int64)

    if not np.array_equal(labels, expected_labels):
        raise ValueError(f"{path}: labels and latents disagree.")

    if not np.allclose(
        z * (sizes - 1),
        np.round(z * (sizes - 1)),
        atol=1e-5,
    ):
        raise ValueError(f"{path}: latents are not on the expected grid.")

    factor_names = [str(x) for x in data["factor_names"].tolist()]

    expected_names = [
        f"slot{slot}_{factor}"
        for slot in [1, 2]
        for factor in FACTOR_NAMES
    ]

    if factor_names != expected_names:
        raise ValueError(
            f"{path}: unexpected factor order.\n"
            f"Expected: {expected_names}\n"
            f"Found: {factor_names}"
        )

    data["representations"] = H
    data["latents"] = z
    data["labels"] = labels

    return data


def make_train_test_split(n, test_fraction=0.2, seed=42):

    rng = np.random.default_rng(seed)
    indices = rng.permutation(n)

    n_test = int(round(n * test_fraction))

    if n_test < 1 or n_test >= n:
        raise ValueError("Invalid train/test split size.")

    test_indices = np.sort(indices[:n_test])
    train_indices = np.sort(indices[n_test:])

    return train_indices, test_indices


def check_split_coverage(labels, train_indices, test_indices):

    for idx in ALL_FACTOR_INDICES:

        expected = FACTOR_CARDINALITIES[factor_short_name(idx)]

        train_classes = np.unique(labels[train_indices, idx])
        test_classes = np.unique(labels[test_indices, idx])

        expected_classes = np.arange(expected)

        missing_train = np.setdiff1d(expected_classes, train_classes)
        missing_test = np.setdiff1d(expected_classes, test_classes)

        if len(missing_train) or len(missing_test):
            raise ValueError(
                f"{factor_name(idx)} has missing classes. "
                f"Train missing: {missing_train}; "
                f"test missing: {missing_test}."
            )


# ============================================================
# 2. Factor recovery
# ============================================================

def recover_factor_means(H, labels, train_indices, factor_indices):

    H_train = H[train_indices]
    global_mean = H_train.mean(dim=0)

    H_centered = H_train - global_mean
    labels_train = labels[train_indices]

    factor_means = {}

    for idx in factor_indices:

        expected = FACTOR_CARDINALITIES[factor_short_name(idx)]

        means = []

        for value in range(expected):

            mask = labels_train[:, idx] == value

            if not torch.any(mask):
                raise ValueError(
                    f"Missing training values for {factor_name(idx)}."
                )

            means.append(H_centered[mask].mean(dim=0))

        factor_means[idx] = torch.stack(means, dim=0)

    return global_mean, factor_means


# ============================================================
# 3. GPU multinomial logistic regression
# ============================================================

def fit_gpu_logistic_probe(
    H_train,
    y_train,
    max_iter=300,
    tolerance=1e-9,
):
    """
    Multinomial logistic regression implemented in PyTorch.

    Standardisation is fitted on training data only.

    L2 regularisation is applied to weights, not the
    intercept. The objective is mean cross-entropy plus
    the corresponding C=1 regularisation term.

    This replaces sklearn LogisticRegression.
    """

    classes = torch.unique(y_train)

    if len(classes) < 2:
        raise ValueError("Logistic regression requires at least two classes.")

    class_lookup = torch.searchsorted(classes, y_train)
    n_classes = len(classes)

    mean = H_train.mean(dim=0)
    std = H_train.std(dim=0, unbiased=False)

    std = torch.where(std > 1e-12, std, torch.ones_like(std))

    X = (H_train - mean) / std

    n_samples, n_features = X.shape

    model = torch.nn.Linear(
        n_features,
        n_classes,
        bias=True,
        device=H_train.device,
        dtype=H_train.dtype,
    )

    torch.nn.init.zeros_(model.weight)
    torch.nn.init.zeros_(model.bias)

    optimizer = torch.optim.LBFGS(
        model.parameters(),
        lr=1.0,
        max_iter=max_iter,
        tolerance_grad=tolerance,
        tolerance_change=tolerance,
        history_size=10,
        line_search_fn="strong_wolfe",
    )

    # sklearn's summed-loss objective corresponds approximately
    # to this mean-loss regularisation coefficient.
    regularisation = (
        1.0 / (LOGISTIC_REGULARISATION_C * n_samples)
    )

    def closure():

        optimizer.zero_grad()

        logits = model(X)

        loss = F.cross_entropy(logits, class_lookup)

        loss = loss + (
            0.5 * regularisation * model.weight.square().sum()
        )

        loss.backward()

        return loss

    optimizer.step(closure)

    # Convert weights from standardised coordinates
    # back to original representation coordinates.
    original_weights = model.weight.detach() / std.unsqueeze(0)

    return {
        "model": model,
        "mean": mean,
        "std": std,
        "classes": classes,
        "original_weights": original_weights,
    }


def predict_gpu_probe(probe, H):

    X = (H - probe["mean"]) / probe["std"]

    logits = probe["model"](X)

    predictions = probe["classes"][logits.argmax(dim=1)]

    return predictions


def fit_linear_probes(H, labels, train_indices, factor_indices):

    probe_weights = []
    fitted_probes = {}

    for idx in factor_indices:

        probe = fit_gpu_logistic_probe(
            H[train_indices],
            labels[train_indices, idx],
        )

        probe_weights.append(probe["original_weights"])
        fitted_probes[idx] = probe

    W_all = torch.cat(probe_weights, dim=0)

    _, singular_values, Vh = torch.linalg.svd(
        W_all,
        full_matrices=False,
    )

    if len(singular_values) == 0:
        raise ValueError("Probe weight matrix has zero rank.")

    tolerance = (
        max(W_all.shape)
        * torch.finfo(W_all.dtype).eps
        * singular_values[0]
    )

    tolerance = torch.maximum(
        tolerance,
        torch.tensor(
            PROBE_RANK_TOLERANCE,
            device=H.device,
            dtype=H.dtype,
        ),
    )

    rank = int((singular_values > tolerance).sum().item())

    if rank == 0:
        raise ValueError("The joint probe span has zero rank.")

    basis = Vh[:rank].T

    return fitted_probes, basis


def project_representations(H, train_indices, basis):

    train_mean = H[train_indices].mean(dim=0)

    projected = (H - train_mean) @ basis

    return projected, train_mean


# ============================================================
# 4. GPU whitening
# ============================================================

def fit_whitening_transform(
    projected,
    train_indices,
    epsilon=WHITENING_EPSILON,
):

    X_train = projected[train_indices]

    mean = X_train.mean(dim=0)
    X_centered = X_train - mean

    covariance = (
        X_centered.T @ X_centered
    ) / max(len(X_centered) - 1, 1)

    eigenvalues, eigenvectors = torch.linalg.eigh(covariance)

    eigenvalues = torch.clamp(eigenvalues, min=0.0)

    inverse_sqrt = torch.rsqrt(eigenvalues + epsilon)

    whitening_matrix = (
        eigenvectors
        * inverse_sqrt.unsqueeze(0)
    ) @ eigenvectors.T

    return mean, whitening_matrix


def apply_whitening(projected, mean, whitening_matrix):

    return (projected - mean) @ whitening_matrix


# ============================================================
# 5. GPU additive reconstruction and R2
# ============================================================

def calculate_factorisation_r2(
    H,
    labels,
    train_indices,
    test_indices,
    factor_indices,
):

    global_mean, factor_means = recover_factor_means(
        H,
        labels,
        train_indices,
        factor_indices,
    )

    H_test = H[test_indices] - global_mean
    labels_test = labels[test_indices]

    predictions = torch.zeros_like(H_test)

    for idx in factor_indices:
        predictions += factor_means[idx][labels_test[:, idx]]

    residual = H_test - predictions

    sse = residual.square().sum()
    sst = H_test.square().sum()

    if scalar(sst) <= 1e-12:
        r2 = float("nan")
    else:
        r2 = scalar(1.0 - sse / sst)

    rmse = scalar(torch.sqrt(residual.square().mean()))

    return {
        "r2": r2,
        "sse": scalar(sse),
        "sst": scalar(sst),
        "rmse": rmse,
    }


def calculate_all_factorisation_metrics(
    H,
    labels,
    train_indices,
    test_indices,
    factor_indices,
):

    raw = calculate_factorisation_r2(
        H, labels, train_indices, test_indices, factor_indices
    )

    _, basis = fit_linear_probes(
        H, labels, train_indices, factor_indices
    )

    projected, _ = project_representations(
        H, train_indices, basis
    )

    projected_metrics = calculate_factorisation_r2(
        projected, labels, train_indices, test_indices, factor_indices
    )

    whitening_mean, whitening_matrix = fit_whitening_transform(
        projected, train_indices
    )

    projected_whitened = apply_whitening(
        projected, whitening_mean, whitening_matrix
    )

    whitened_metrics = calculate_factorisation_r2(
        projected_whitened,
        labels,
        train_indices,
        test_indices,
        factor_indices,
    )

    return {
        "raw_r2": raw["r2"],
        "projected_r2": projected_metrics["r2"],
        "projected_whitened_r2": whitened_metrics["r2"],
        "raw_rmse": raw["rmse"],
        "projected_rmse": projected_metrics["rmse"],
        "projected_whitened_rmse": whitened_metrics["rmse"],
        "probe_span_rank": int(basis.shape[1]),
        "whitening_epsilon": WHITENING_EPSILON,
    }


# ============================================================
# 6. GPU orthogonality: Paper 2 Appendix H.2
# ============================================================

def calculate_orthogonality(
    H,
    labels,
    train_indices,
    factor_indices,
):

    _, factor_means = recover_factor_means(
        H, labels, train_indices, factor_indices
    )

    normalized_directions = {}

    for idx in factor_indices:

        means = factor_means[idx]

        centered = means - means.mean(dim=0, keepdim=True)

        norms = torch.linalg.vector_norm(
            centered, dim=1, keepdim=True
        )

        valid = norms[:, 0] > 1e-12

        normalized = centered[valid] / norms[valid]

        normalized_directions[idx] = normalized

    pair_results = []
    within_values = []
    across_values = []

    matrix = torch.full(
        (len(factor_indices), len(factor_indices)),
        float("nan"),
        device=H.device,
        dtype=H.dtype,
    )

    for i, factor_a in enumerate(factor_indices):

        A = normalized_directions[factor_a]

        for j in range(i, len(factor_indices)):

            factor_b = factor_indices[j]
            B = normalized_directions[factor_b]

            if len(A) == 0 or len(B) == 0:
                continue

            similarities = torch.abs(A @ B.T)

            if factor_a == factor_b:

                mask = ~torch.eye(
                    len(A), device=H.device, dtype=torch.bool
                )

                values = similarities[mask]
                comparison_type = "within"

            else:

                values = similarities.reshape(-1)
                comparison_type = "across"

            if len(values) == 0:
                continue

            mean_value = values.mean()

            matrix[i, j] = mean_value
            matrix[j, i] = mean_value

            if comparison_type == "within":
                within_values.append(values)
            else:
                across_values.append(values)

            pair_results.append({
                "factor_a": factor_name(factor_a),
                "factor_b": factor_name(factor_b),
                "comparison_type": comparison_type,
                "mean_absolute_cosine": scalar(mean_value),
                "max_absolute_cosine": scalar(values.max()),
                "n_comparisons": len(values),
                "valid_values_a": len(A),
                "valid_values_b": len(B),
            })

    within = (
        torch.cat(within_values)
        if within_values else torch.empty(0, device=H.device)
    )

    across = (
        torch.cat(across_values)
        if across_values else torch.empty(0, device=H.device)
    )

    summary = {
        "mean_within_factor_absolute_cosine": (
            scalar(within.mean()) if len(within) else float("nan")
        ),
        "mean_across_factor_absolute_cosine": (
            scalar(across.mean()) if len(across) else float("nan")
        ),
        "n_within_comparisons": len(within),
        "n_across_comparisons": len(across),
    }

    names = [factor_name(idx) for idx in factor_indices]
    matrix_rows = []

    matrix_cpu = matrix.cpu().numpy()

    for i, name in enumerate(names):

        row = {"factor": name}

        for j, other_name in enumerate(names):
            row[other_name] = float(matrix_cpu[i, j])

        matrix_rows.append(row)

    return summary, pair_results, matrix_rows


# ============================================================
# 7. GPU factor dimensionality: Paper 2 Appendix H.3
# ============================================================

def calculate_factor_dimensionality(
    H,
    labels,
    train_indices,
    factor_indices,
):

    _, factor_means = recover_factor_means(
        H, labels, train_indices, factor_indices
    )

    results = []

    for idx in factor_indices:

        U = factor_means[idx]

        U_centered = U - U.mean(dim=0, keepdim=True)

        singular_values = torch.linalg.svdvals(U_centered)

        variances = singular_values.square()
        total_variance = variances.sum()

        if scalar(total_variance) <= 1e-12:

            k95 = 0
            effective_rank = 0.0
            variance_at_k95 = float("nan")

        else:

            probabilities = variances / total_variance

            cumulative = torch.cumsum(probabilities, dim=0)

            k95 = int(
                torch.searchsorted(
                    cumulative,
                    torch.tensor(
                        0.95, device=H.device, dtype=H.dtype
                    ),
                ).item() + 1
            )

            effective_rank = scalar(
                torch.exp(
                    -torch.sum(
                        probabilities
                        * torch.log(probabilities + 1e-12)
                    )
                )
            )

            variance_at_k95 = scalar(cumulative[k95 - 1])

        results.append({
            "factor": factor_name(idx),
            "n_values": int(U.shape[0]),
            "k95": k95,
            "effective_rank": effective_rank,
            "variance_explained_by_k95": variance_at_k95,
            "total_factor_variance": scalar(total_variance),
        })

    return results


# ============================================================
# 8. GPU overall representation dimensionality
# ============================================================

def calculate_overall_dimensionality(H, train_indices):

    H_train = H[train_indices]

    H_centered = H_train - H_train.mean(dim=0, keepdim=True)

    singular_values = torch.linalg.svdvals(H_centered)

    variances = singular_values.square()
    total = variances.sum()

    if scalar(total) <= 1e-12:
        raise ValueError("Representation has effectively zero variance.")

    probabilities = variances / total

    participation_ratio = scalar(
        1.0 / probabilities.square().sum()
    )

    cumulative = torch.cumsum(probabilities, dim=0)

    k95 = int(
        torch.searchsorted(
            cumulative,
            torch.tensor(0.95, device=H.device, dtype=H.dtype),
        ).item() + 1
    )

    return {
        "effective_dimensionality": participation_ratio,
        "k95": k95,
        "total_components": len(variances),
        "variance_explained_by_k95": scalar(cumulative[k95 - 1]),
    }


# ============================================================
# 9. GPU factor subspace overlap and principal angles
# ============================================================

def get_factor_subspace(
    H,
    labels,
    train_indices,
    factor_idx,
):

    _, factor_means = recover_factor_means(
        H, labels, train_indices, [factor_idx]
    )

    U = factor_means[factor_idx]
    U = U - U.mean(dim=0, keepdim=True)

    _, singular_values, Vh = torch.linalg.svd(
        U, full_matrices=False
    )

    if len(singular_values) == 0 or scalar(singular_values[0]) <= 1e-12:
        return torch.empty((H.shape[1], 0), device=H.device, dtype=H.dtype), 0

    threshold = singular_values[0] * SUBSPACE_RANK_TOLERANCE

    rank = int((singular_values > threshold).sum().item())

    return Vh[:rank].T, rank


def calculate_factor_subspace_overlap(
    H,
    labels,
    train_indices,
    factor_indices,
):

    bases = {}
    ranks = {}

    for idx in factor_indices:

        basis, rank = get_factor_subspace(
            H, labels, train_indices, idx
        )

        bases[idx] = basis
        ranks[idx] = rank

    results = []

    for i, factor_a in enumerate(factor_indices):

        for factor_b in factor_indices[i + 1:]:

            rank_a = ranks[factor_a]
            rank_b = ranks[factor_b]

            if rank_a == 0 or rank_b == 0:

                overlap = float("nan")
                mean_angle = float("nan")
                minimum_angle = float("nan")
                maximum_angle = float("nan")

            else:

                singular_values = torch.linalg.svdvals(
                    bases[factor_a].T @ bases[factor_b]
                )

                overlap = scalar(
                    singular_values.square().sum()
                    / np.sqrt(rank_a * rank_b)
                )

                angles = torch.rad2deg(
                    torch.acos(
                        torch.clamp(singular_values, 0.0, 1.0)
                    )
                )

                mean_angle = scalar(angles.mean())
                minimum_angle = scalar(angles.min())
                maximum_angle = scalar(angles.max())

            results.append({
                "factor_a": factor_name(factor_a),
                "factor_b": factor_name(factor_b),
                "rank_a": rank_a,
                "rank_b": rank_b,
                "subspace_overlap": overlap,
                "mean_principal_angle_degrees": mean_angle,
                "minimum_principal_angle_degrees": minimum_angle,
                "maximum_principal_angle_degrees": maximum_angle,
            })

    return results


# ============================================================
# 10. GPU linear probes and cross-slot leakage
# ============================================================

def run_linear_probes(
    H,
    labels,
    train_indices,
    test_indices,
    factor_indices,
    source_name,
    probe_max_iter,
):

    results = []

    for idx in factor_indices:

        y_train = labels[train_indices, idx]
        y_test = labels[test_indices, idx]

        probe = fit_gpu_logistic_probe(
            H[train_indices],
            y_train,
            max_iter=probe_max_iter,
        )

        predictions = predict_gpu_probe(
            probe, H[test_indices]
        )

        classes = probe["classes"]
        n_classes = len(classes)

        accuracy = (predictions == y_test).to(H.dtype).mean()

        recalls = []

        for cls in classes:

            mask = y_test == cls

            if torch.any(mask):
                recalls.append(
                    (predictions[mask] == cls).to(H.dtype).mean()
                )

        balanced_accuracy = torch.stack(recalls).mean()

        chance = 1.0 / n_classes

        chance_adjusted = (
            (balanced_accuracy - chance) / (1.0 - chance)
        )

        results.append({
            "source": source_name,
            "factor": factor_name(idx),
            "factor_index": idx,
            "accuracy": scalar(accuracy),
            "balanced_accuracy": scalar(balanced_accuracy),
            "chance_balanced_accuracy": chance,
            "chance_adjusted_balanced_accuracy": scalar(chance_adjusted),
            "n_classes": n_classes,
        })

    return results


def calculate_cross_slot_leakage(
    H,
    labels,
    train_indices,
    test_indices,
    source_slot,
    probe_max_iter,
):

    own_indices = SLOT_FACTOR_INDICES[source_slot]

    other_slot = (
        "slot2" if source_slot == "slot1" else "slot1"
    )

    other_indices = SLOT_FACTOR_INDICES[other_slot]

    own_results = run_linear_probes(
        H, labels, train_indices, test_indices,
        own_indices, source_slot, probe_max_iter
    )

    other_results = run_linear_probes(
        H, labels, train_indices, test_indices,
        other_indices, other_slot, probe_max_iter
    )

    own_lookup = {
        factor_short_name(row["factor_index"]):
            row["chance_adjusted_balanced_accuracy"]
        for row in own_results
    }

    other_lookup = {
        factor_short_name(row["factor_index"]):
            row["chance_adjusted_balanced_accuracy"]
        for row in other_results
    }

    results = []

    for factor in ["shape", "scale", "x_position", "y_position"]:

        own_score = own_lookup[factor]
        cross_score = other_lookup[factor]

        ratio = (
            cross_score / own_score
            if np.isfinite(own_score) and own_score > 0
            else float("nan")
        )

        results.append({
            "source_slot": source_slot,
            "target_factor": factor,
            "within_slot_adjusted_balanced_accuracy": own_score,
            "cross_slot_adjusted_balanced_accuracy": cross_score,
            "leakage_ratio": float(ratio),
        })

    return results


# ============================================================
# 11. Automatic condition discovery
# ============================================================

def discover_conditions(root):

    if not root.exists():
        raise FileNotFoundError(root)

    if not root.is_dir():
        raise NotADirectoryError(root)

    conditions = []

    for folder in sorted(root.iterdir()):

        if not folder.is_dir():
            continue

        if folder.name.endswith("_compositional"):
            architecture = "compositional"

        elif folder.name.endswith("_monolithic"):
            architecture = "monolithic"

        else:
            continue

        conditions.append((folder, architecture))

    if not conditions:
        raise FileNotFoundError(
            f"No representation conditions found in {root}."
        )

    print(f"\nDiscovered {len(conditions)} condition(s):")

    for folder, architecture in conditions:
        print(f"  {folder.name}: {architecture}")

    return conditions


# ============================================================
# 12. Analyse one representation
# ============================================================

def analyse_representation(
    data,
    condition_name,
    architecture,
    level,
    slot,
    output_folder,
    train_indices,
    test_indices,
    device,
    dtype,
    probe_max_iter,
):

    H = torch.as_tensor(
        data["representations"],
        device=device,
        dtype=dtype,
    )

    labels = torch.as_tensor(
        data["labels"],
        device=device,
        dtype=torch.long,
    )

    train_indices = torch.as_tensor(
        train_indices, device=device, dtype=torch.long
    )

    test_indices = torch.as_tensor(
        test_indices, device=device, dtype=torch.long
    )

    output_folder.mkdir(parents=True, exist_ok=True)

    factor_indices = (
        SLOT_FACTOR_INDICES[slot]
        if architecture == "compositional"
        else ALL_FACTOR_INDICES
    )

    metadata = {
        "condition": condition_name,
        "architecture": architecture,
        "level": level,
        "slot": slot,
        "evaluation": "IID held-out split",
        "device": str(device),
        "dtype": str(dtype),
        "primary_factorisation_metric": "projected_whitened_r2",
        "orthogonality_space": "original representation space",
        "whitening_epsilon": WHITENING_EPSILON,
    }

    # Linear probes
    probe_results = run_linear_probes(
        H, labels, train_indices, test_indices,
        factor_indices, slot, probe_max_iter
    )

    save_csv(output_folder / "linear_probes.csv", probe_results)

    # Factorisation
    factorisation = calculate_all_factorisation_metrics(
        H, labels, train_indices, test_indices, factor_indices
    )

    save_csv(
        output_folder / "factorisation.csv",
        [{**metadata, **factorisation}],
    )

    # Orthogonality
    orth_summary, orth_pairs, orth_matrix = calculate_orthogonality(
        H, labels, train_indices, factor_indices
    )

    save_csv(
        output_folder / "orthogonality_summary.csv",
        [{**metadata, **orth_summary}],
    )

    save_csv(
        output_folder / "orthogonality_pairs.csv",
        orth_pairs,
    )

    save_csv(
        output_folder / "orthogonality_matrix.csv",
        orth_matrix,
    )

    # Factor-specific K95
    factor_dimensionality = calculate_factor_dimensionality(
        H, labels, train_indices, factor_indices
    )

    save_csv(
        output_folder / "factor_dimensionality.csv",
        [{**metadata, **row} for row in factor_dimensionality],
    )

    # Overall dimensionality
    overall_dimensionality = calculate_overall_dimensionality(
        H, train_indices
    )

    save_csv(
        output_folder / "overall_dimensionality.csv",
        [{**metadata, **overall_dimensionality}],
    )

    # Factor subspace overlap
    overlap_results = calculate_factor_subspace_overlap(
        H, labels, train_indices, factor_indices
    )

    save_csv(
        output_folder / "factor_subspace_overlap.csv",
        overlap_results,
    )

    # Cross-slot leakage
    if architecture == "compositional":

        leakage_results = calculate_cross_slot_leakage(
            H,
            labels,
            train_indices,
            test_indices,
            source_slot=slot,
            probe_max_iter=probe_max_iter,
        )

        save_csv(
            output_folder / "cross_slot_leakage.csv",
            leakage_results,
        )

    print(f"\nCompleted: {condition_name}/{level}/{slot}")

    print(f"  Device: {device}")
    print(f"  Raw R2: {factorisation['raw_r2']:.4f}")
    print(f"  Projected R2: {factorisation['projected_r2']:.4f}")
    print(
        "  Projected-whitened R2: "
        f"{factorisation['projected_whitened_r2']:.4f}"
    )

    print(
        "  Within-factor cosine: "
        f"{orth_summary['mean_within_factor_absolute_cosine']:.4f}"
    )

    print(
        "  Across-factor cosine: "
        f"{orth_summary['mean_across_factor_absolute_cosine']:.4f}"
    )

    print(
        "  Overall K95: "
        f"{overall_dimensionality['k95']}"
    )

    print(
        "  Overall effective dimensionality: "
        f"{overall_dimensionality['effective_dimensionality']:.2f}"
    )

    del H, labels

    if device.type == "cuda":
        torch.cuda.empty_cache()


# ============================================================
# 13. Main
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description="GPU-based Paper 2-style representation analysis."
    )

    parser.add_argument(
        "--root",
        type=str,
        default="representations",
    )

    parser.add_argument(
        "--output",
        type=str,
        default="analysis_results_gpu",
    )

    parser.add_argument(
        "--test-fraction",
        type=float,
        default=0.2,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
    )

    parser.add_argument(
        "--dtype",
        choices=["float32", "float64"],
        default="float64",
    )

    parser.add_argument(
        "--probe-max-iter",
        type=int,
        default=300,
    )

    args = parser.parse_args()

    if not 0 < args.test_fraction < 1:
        raise ValueError("--test-fraction must be between 0 and 1.")

    if args.probe_max_iter < 1:
        raise ValueError("--probe-max-iter must be positive.")

    device = torch.device(args.device)

    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA was requested but is not available. "
            "Check your PyTorch CUDA installation."
        )

    dtype = (
        torch.float64
        if args.dtype == "float64"
        else torch.float32
    )

    if device.type == "cuda":
        print(f"\nUsing GPU: {torch.cuda.get_device_name(device)}")
    else:
        print(f"\nUsing device: {device}")

    root = Path(args.root)
    output_root = Path(args.output)

    conditions = discover_conditions(root)

    # --------------------------------------------------------
    # Common evaluation split
    # --------------------------------------------------------

    first_folder, first_architecture = conditions[0]

    if first_architecture == "compositional":
        reference_path = first_folder / "initial_slot1.npz"
    else:
        reference_path = first_folder / "initial_shared.npz"

    reference = load_representation(reference_path)

    sample_ids = reference["sample_ids"]
    labels = reference["labels"]

    train_indices, test_indices = make_train_test_split(
        len(sample_ids),
        test_fraction=args.test_fraction,
        seed=args.seed,
    )

    check_split_coverage(
        labels,
        train_indices,
        test_indices,
    )

    output_root.mkdir(parents=True, exist_ok=True)

    np.savez_compressed(
        output_root / "common_split.npz",
        train_indices=train_indices,
        test_indices=test_indices,
        sample_ids=sample_ids,
        seed=args.seed,
        test_fraction=args.test_fraction,
    )

    print("\n" + "=" * 65)
    print("GPU REPRESENTATION ANALYSIS")
    print("=" * 65)

    print(f"Conditions: {len(conditions)}")
    print(f"Training samples: {len(train_indices)}")
    print(f"Test samples: {len(test_indices)}")
    print(f"Seed: {args.seed}")
    print(f"Device: {device}")
    print(f"Precision: {dtype}")

    # --------------------------------------------------------
    # Analyse all conditions and representation levels
    # --------------------------------------------------------

    for folder, architecture in conditions:

        condition_name = folder.name

        print("\n" + "=" * 65)
        print(f"CONDITION: {condition_name}")
        print(f"ARCHITECTURE: {architecture}")
        print("=" * 65)

        if architecture == "compositional":

            representations = {
                (level, slot): folder / f"{level}_{slot}.npz"
                for level in LEVELS
                for slot in ["slot1", "slot2"]
            }

        else:

            representations = {
                (level, "shared"): folder / f"{level}_shared.npz"
                for level in LEVELS
            }

        for (level, slot), path in representations.items():

            data = load_representation(path)

            if not np.array_equal(data["sample_ids"], sample_ids):
                raise ValueError(f"Sample IDs differ in {path}.")

            if not np.array_equal(data["labels"], labels):
                raise ValueError(f"Labels differ in {path}.")

            if not np.array_equal(data["latents"], reference["latents"]):
                raise ValueError(f"Latents differ in {path}.")

            analysis_folder = (
                output_root / condition_name / level / slot
            )

            analyse_representation(
                data=data,
                condition_name=condition_name,
                architecture=architecture,
                level=level,
                slot=slot,
                output_folder=analysis_folder,
                train_indices=train_indices,
                test_indices=test_indices,
                device=device,
                dtype=dtype,
                probe_max_iter=args.probe_max_iter,
            )

    print("\n" + "=" * 65)
    print("ALL GPU ANALYSES COMPLETED")
    print(f"Results saved in: {output_root}")
    print("=" * 65)


if __name__ == "__main__":
    main()
