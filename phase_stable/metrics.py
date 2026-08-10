"""Metrics for temporal sampling-origin stability experiments.

The functions in this module deliberately depend only on NumPy.  Inputs are
validated rather than silently dropping NaNs: a missing value in an experiment
artifact is a pipeline error and should not turn into an apparently valid
aggregate.

Array conventions
-----------------
* Origin-wise representations have shape ``(n_origins, ...)``.
* A single representation has shape ``(n_scales, n_times, ...)`` and scale is
  axis 0 unless ``scale_axis`` is supplied.
* MLLM predictions have shape ``(n_items, n_origins)``.
* Timestamp and label arrays are one-dimensional and timestamps must already
  be sorted in non-decreasing order.
"""

from __future__ import annotations

from collections.abc import Callable, Hashable
from typing import Any

import numpy as np

ArrayLike = Any


def _finite_float_array(
    values: ArrayLike,
    *,
    name: str,
    ndim: int | None = None,
    nonempty: bool = True,
) -> np.ndarray:
    """Convert to float64 and enforce shape, non-emptiness, and finiteness."""

    try:
        array = np.asarray(values, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a rectangular numeric array") from exc
    if ndim is not None and array.ndim != ndim:
        raise ValueError(f"{name} must be {ndim}-dimensional, got {array.ndim}")
    if nonempty and array.size == 0:
        raise ValueError(f"{name} must not be empty")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values (no NaN/inf)")
    return array


def _sorted_timestamps(values: ArrayLike, *, name: str) -> np.ndarray:
    timestamps = _finite_float_array(values, name=name, ndim=1, nonempty=False)
    if timestamps.size > 1 and np.any(np.diff(timestamps) < 0):
        raise ValueError(f"{name} must be sorted in non-decreasing order")
    return timestamps


def _origin_matrix(values: ArrayLike, *, name: str) -> np.ndarray:
    array = _finite_float_array(values, name=name)
    if array.ndim < 2:
        raise ValueError(f"{name} must have shape (n_origins, ...)")
    if array.shape[0] < 2:
        raise ValueError(f"{name} must contain at least two origins")
    if int(np.prod(array.shape[1:])) == 0:
        raise ValueError(f"{name} contains an empty per-origin representation")
    return array.reshape(array.shape[0], -1)


def _pair_indices(n: int) -> tuple[np.ndarray, np.ndarray]:
    return np.triu_indices(n, k=1)


def pairwise_cosine_consistency(values: ArrayLike) -> dict[str, float | int]:
    """Return all-origin-pair cosine mean and worst (minimum) similarity.

    Constant non-zero vectors are valid.  A zero vector has no cosine
    direction, so any origin containing one raises ``ValueError`` instead of
    producing NaN or assigning an arbitrary similarity.
    """

    matrix = _origin_matrix(values, name="values")
    norms = np.linalg.norm(matrix, axis=1)
    if np.any(norms == 0):
        bad = np.flatnonzero(norms == 0).tolist()
        raise ValueError(f"cosine similarity is undefined for zero-vector origins: {bad}")
    unit = matrix / norms[:, None]
    left, right = _pair_indices(matrix.shape[0])
    similarities = np.einsum("ij,ij->i", unit[left], unit[right])
    similarities = np.clip(similarities, -1.0, 1.0)
    return {
        "mean": float(np.mean(similarities)),
        "worst": float(np.min(similarities)),
        "n_pairs": int(similarities.size),
    }


def all_pair_cosine(values: ArrayLike) -> dict[str, float | int]:
    """Alias for :func:`pairwise_cosine_consistency`."""

    return pairwise_cosine_consistency(values)


def normalized_l1_distance(first: ArrayLike, second: ArrayLike) -> float:
    """Symmetric normalized L1 distance in ``[0, 1]``.

    The definition is ``sum(|a-b|) / (sum(|a|)+sum(|b|))``.  Two all-zero
    arrays are identical and return 0.  Shapes must match and NaNs are errors.
    """

    first_array = _finite_float_array(first, name="first")
    second_array = _finite_float_array(second, name="second")
    if first_array.shape != second_array.shape:
        raise ValueError(
            f"first and second must have the same shape, got "
            f"{first_array.shape} and {second_array.shape}"
        )
    numerator = float(np.sum(np.abs(first_array - second_array)))
    denominator = float(np.sum(np.abs(first_array)) + np.sum(np.abs(second_array)))
    if denominator == 0:
        return 0.0
    return float(np.clip(numerator / denominator, 0.0, 1.0))


def pairwise_normalized_l1(values: ArrayLike) -> dict[str, float | int]:
    """Return mean and worst (maximum) normalized L1 over all origin pairs."""

    matrix = _origin_matrix(values, name="values")
    left, right = _pair_indices(matrix.shape[0])
    distances = np.asarray(
        [normalized_l1_distance(matrix[a], matrix[b]) for a, b in zip(left, right)],
        dtype=np.float64,
    )
    return {
        "mean": float(np.mean(distances)),
        "worst": float(np.max(distances)),
        "n_pairs": int(distances.size),
    }


def js_divergence(first: ArrayLike, second: ArrayLike, *, base: float = 2.0) -> float:
    """Jensen-Shannon divergence between non-negative, unnormalized vectors.

    Inputs are L1-normalized internally.  Each must have positive mass.  With
    the default base 2 the result lies in ``[0, 1]``.  Zero-valued bins are
    handled by the usual ``0 log 0 = 0`` convention.
    """

    first_array = _finite_float_array(first, name="first", ndim=1)
    second_array = _finite_float_array(second, name="second", ndim=1)
    if first_array.shape != second_array.shape:
        raise ValueError("first and second distributions must have the same shape")
    if np.any(first_array < 0) or np.any(second_array < 0):
        raise ValueError("Jensen-Shannon inputs must be non-negative")
    if not np.isfinite(base) or base <= 0 or base == 1:
        raise ValueError("base must be finite, positive, and different from 1")
    first_mass = float(np.sum(first_array))
    second_mass = float(np.sum(second_array))
    if first_mass <= 0 or second_mass <= 0:
        raise ValueError("each Jensen-Shannon input must have positive total mass")
    p = first_array / first_mass
    q = second_array / second_mass
    midpoint = 0.5 * (p + q)
    log_base = np.log(base)

    def kl_to_midpoint(distribution: np.ndarray) -> float:
        positive = distribution > 0
        return float(
            np.sum(
                distribution[positive]
                * (np.log(distribution[positive] / midpoint[positive]) / log_base)
            )
        )

    result = 0.5 * (kl_to_midpoint(p) + kl_to_midpoint(q))
    return float(max(0.0, result))


def scale_energy_proportions(
    representation: ArrayLike,
    *,
    scale_axis: int = 0,
) -> np.ndarray:
    """Return each wavelet scale's share of total squared L2 energy.

    An all-zero representation is rejected because its scale proportions are
    undefined.  A constant non-zero representation is valid.
    """

    array = _finite_float_array(representation, name="representation")
    if array.ndim < 1:
        raise ValueError("representation must have at least one dimension")
    if not isinstance(scale_axis, (int, np.integer)):
        raise TypeError("scale_axis must be an integer")
    axis = int(scale_axis)
    if axis < 0:
        axis += array.ndim
    if axis < 0 or axis >= array.ndim:
        raise ValueError(f"invalid scale_axis {scale_axis} for shape {array.shape}")
    reduce_axes = tuple(index for index in range(array.ndim) if index != axis)
    energy = np.square(array).sum(axis=reduce_axes, dtype=np.float64)
    total = float(np.sum(energy))
    if total <= 0:
        raise ValueError("scale energy proportions are undefined for an all-zero representation")
    return energy / total


def scale_energy_drift(
    first: ArrayLike,
    second: ArrayLike,
    *,
    scale_axis: int = 0,
    inputs_are_proportions: bool = False,
) -> dict[str, float]:
    """Compare two scale-energy profiles with L1 and Jensen-Shannon drift.

    ``l1`` is the ordinary L1 distance between probability vectors (range
    ``[0, 2]``), not the symmetric signal normalization used by
    :func:`normalized_l1_distance`.  ``js_distance`` is the square root of the
    base-2 Jensen-Shannon divergence.
    """

    if inputs_are_proportions:
        p = _finite_float_array(first, name="first", ndim=1)
        q = _finite_float_array(second, name="second", ndim=1)
        if p.shape != q.shape:
            raise ValueError("energy proportion vectors must have the same shape")
        if np.any(p < 0) or np.any(q < 0) or np.sum(p) <= 0 or np.sum(q) <= 0:
            raise ValueError("energy proportions must be non-negative with positive mass")
        p = p / np.sum(p)
        q = q / np.sum(q)
    else:
        p = scale_energy_proportions(first, scale_axis=scale_axis)
        q = scale_energy_proportions(second, scale_axis=scale_axis)
        if p.shape != q.shape:
            raise ValueError("representations must contain the same number of scales")
    divergence = js_divergence(p, q)
    return {
        "l1": float(np.sum(np.abs(p - q))),
        "js_divergence": divergence,
        "js_distance": float(np.sqrt(divergence)),
    }


def _optimal_timestamp_pairs(
    first: np.ndarray,
    second: np.ndarray,
    tolerance: float | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Max-cardinality, then min-total-distance, non-crossing 1-D matching."""

    n_first, n_second = first.size, second.size
    counts = np.zeros((n_first + 1, n_second + 1), dtype=np.int64)
    costs = np.zeros((n_first + 1, n_second + 1), dtype=np.float64)
    # 1 = skip first, 2 = skip second, 3 = match.
    choice = np.zeros((n_first + 1, n_second + 1), dtype=np.uint8)

    def better(
        candidate_count: int,
        candidate_cost: float,
        best_count: int,
        best_cost: float,
    ) -> bool:
        return candidate_count > best_count or (
            candidate_count == best_count and candidate_cost < best_cost - 1e-15
        )

    for first_index in range(1, n_first + 1):
        for second_index in range(1, n_second + 1):
            best_count = int(counts[first_index - 1, second_index])
            best_cost = float(costs[first_index - 1, second_index])
            best_choice = 1

            candidate_count = int(counts[first_index, second_index - 1])
            candidate_cost = float(costs[first_index, second_index - 1])
            if better(candidate_count, candidate_cost, best_count, best_cost):
                best_count, best_cost, best_choice = candidate_count, candidate_cost, 2

            distance = abs(first[first_index - 1] - second[second_index - 1])
            if tolerance is None or distance <= tolerance:
                candidate_count = int(counts[first_index - 1, second_index - 1]) + 1
                candidate_cost = float(costs[first_index - 1, second_index - 1]) + distance
                if better(candidate_count, candidate_cost, best_count, best_cost):
                    best_count, best_cost, best_choice = candidate_count, candidate_cost, 3

            counts[first_index, second_index] = best_count
            costs[first_index, second_index] = best_cost
            choice[first_index, second_index] = best_choice

    first_indices: list[int] = []
    second_indices: list[int] = []
    first_index, second_index = n_first, n_second
    while first_index > 0 and second_index > 0:
        selected = int(choice[first_index, second_index])
        if selected == 3:
            first_indices.append(first_index - 1)
            second_indices.append(second_index - 1)
            first_index -= 1
            second_index -= 1
        elif selected == 1:
            first_index -= 1
        else:
            second_index -= 1
    first_indices.reverse()
    second_indices.reverse()
    first_result = np.asarray(first_indices, dtype=np.int64)
    second_result = np.asarray(second_indices, dtype=np.int64)
    distances = np.abs(first[first_result] - second[second_result])
    return first_result, second_result, distances


def match_timestamps(
    first: ArrayLike,
    second: ArrayLike,
    *,
    tolerance: float | None = None,
) -> dict[str, Any]:
    """One-to-one timestamp matching with a maximum-cardinality guarantee.

    Among all maximum-cardinality matches, total absolute displacement is
    minimized.  Because timestamps are sorted and absolute distance is a Monge
    cost, an optimal non-crossing dynamic program is sufficient.  ``tolerance``
    is inclusive; ``None`` permits every pair.  Duplicate timestamps are valid.
    """

    first_array = _sorted_timestamps(first, name="first")
    second_array = _sorted_timestamps(second, name="second")
    if tolerance is not None and (not np.isfinite(tolerance) or tolerance < 0):
        raise ValueError("tolerance must be finite and non-negative, or None")
    first_indices, second_indices, distances = _optimal_timestamp_pairs(
        first_array, second_array, tolerance
    )
    return {
        "first_indices": first_indices,
        "second_indices": second_indices,
        "distances": distances,
        "count": int(distances.size),
        "total_distance": float(np.sum(distances)),
        "mean_distance": None if distances.size == 0 else float(np.mean(distances)),
    }


def tolerant_boundary_metrics(
    reference: ArrayLike,
    candidate: ArrayLike,
    *,
    tolerance: float,
) -> dict[str, Any]:
    """One-to-one tolerant boundary precision, recall, and F1.

    Empty-set convention: two empty boundary sets are a perfect match; exactly
    one empty set receives zero precision/recall/F1.  Mean displacement is
    ``None`` when no pair exists, never NaN.
    """

    reference_array = _sorted_timestamps(reference, name="reference")
    candidate_array = _sorted_timestamps(candidate, name="candidate")
    matching = match_timestamps(reference_array, candidate_array, tolerance=tolerance)
    matched = int(matching["count"])
    if reference_array.size == 0 and candidate_array.size == 0:
        precision = recall = f1 = 1.0
    elif reference_array.size == 0 or candidate_array.size == 0:
        precision = recall = f1 = 0.0
    else:
        precision = matched / candidate_array.size
        recall = matched / reference_array.size
        f1 = 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)
    return {
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "matched": matched,
        "reference_count": int(reference_array.size),
        "candidate_count": int(candidate_array.size),
        "mean_distance": matching["mean_distance"],
        "total_distance": matching["total_distance"],
        "reference_indices": matching["first_indices"],
        "candidate_indices": matching["second_indices"],
    }


def boundary_count_cv(counts: ArrayLike, *, ddof: int = 0) -> float:
    """Coefficient of variation of non-negative boundary counts.

    All-zero counts return 0: their count is perfectly invariant, although the
    caller should report the count itself to expose this degenerate solution.
    """

    array = _finite_float_array(counts, name="counts", ndim=1)
    if np.any(array < 0):
        raise ValueError("boundary counts must be non-negative")
    if int(ddof) != ddof or ddof < 0 or ddof >= array.size:
        raise ValueError("ddof must be a non-negative integer smaller than n")
    mean = float(np.mean(array))
    if mean == 0:
        return 0.0
    return float(np.std(array, ddof=int(ddof)) / mean)


def make_time_grid(
    start: float,
    end: float,
    step: float,
    *,
    include_endpoint: bool = True,
) -> np.ndarray:
    """Create a finite increasing absolute-time grid.

    If ``include_endpoint`` is true and ``end`` is not an exact step, the exact
    endpoint is appended.  Thus the final interval can be shorter than ``step``.
    """

    if not np.all(np.isfinite([start, end, step])):
        raise ValueError("start, end, and step must be finite")
    if end < start:
        raise ValueError("end must be greater than or equal to start")
    if step <= 0:
        raise ValueError("step must be positive")
    span = end - start
    count = int(np.floor(span / step + 1e-12))
    grid = start + step * np.arange(count + 1, dtype=np.float64)
    grid = grid[grid <= end + 1e-12 * max(1.0, abs(end))]
    if include_endpoint:
        if grid.size == 0 or not np.isclose(grid[-1], end, rtol=1e-12, atol=1e-12):
            grid = np.append(grid, float(end))
        else:
            grid[-1] = float(end)
    elif grid.size and np.isclose(grid[-1], end, rtol=1e-12, atol=1e-12):
        grid = grid[:-1]
    return grid


def boundaries_to_segment_labels(boundaries: ArrayLike, grid: ArrayLike) -> np.ndarray:
    """Label every grid timestamp by the segment induced by boundaries.

    A timestamp equal to a boundary belongs to the new (right-hand) segment.
    Boundaries outside the grid extent are rejected as an alignment error.
    """

    boundary_array = _sorted_timestamps(boundaries, name="boundaries")
    grid_array = _sorted_timestamps(grid, name="grid")
    if grid_array.size == 0:
        raise ValueError("grid must not be empty")
    if grid_array.size > 1 and np.any(np.diff(grid_array) <= 0):
        raise ValueError("grid must be strictly increasing")
    if boundary_array.size and (
        boundary_array[0] < grid_array[0] or boundary_array[-1] > grid_array[-1]
    ):
        raise ValueError("boundaries must lie within the grid extent")
    return np.searchsorted(boundary_array, grid_array, side="right").astype(np.int64)


def _label_array(labels: ArrayLike, *, name: str) -> np.ndarray:
    array = np.asarray(labels)
    if array.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional")
    if array.size == 0:
        raise ValueError(f"{name} must not be empty")
    if array.dtype.kind in "fc" and not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must not contain NaN/inf")
    if array.dtype.kind == "O":
        for value in array:
            if value is None or (
                isinstance(value, (float, np.floating)) and not np.isfinite(value)
            ):
                raise ValueError(f"{name} must not contain missing values")
    return array


def _contingency(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    try:
        _, first_inverse = np.unique(first, return_inverse=True)
        _, second_inverse = np.unique(second, return_inverse=True)
    except TypeError as exc:
        raise ValueError("labels must contain mutually comparable scalar values") from exc
    table = np.zeros(
        (int(first_inverse.max()) + 1, int(second_inverse.max()) + 1),
        dtype=np.int64,
    )
    np.add.at(table, (first_inverse, second_inverse), 1)
    return table


def adjusted_rand_index(first_labels: ArrayLike, second_labels: ArrayLike) -> float:
    """Adjusted Rand Index, matching sklearn's standard definition."""

    first = _label_array(first_labels, name="first_labels")
    second = _label_array(second_labels, name="second_labels")
    if first.shape != second.shape:
        raise ValueError("label arrays must have the same shape")
    n_samples = first.size
    if n_samples < 2:
        return 1.0
    table = _contingency(first, second)

    def choose_two(values: np.ndarray) -> float:
        values_float = values.astype(np.float64)
        return float(np.sum(values_float * (values_float - 1.0) / 2.0))

    sum_cells = choose_two(table.ravel())
    sum_rows = choose_two(table.sum(axis=1))
    sum_columns = choose_two(table.sum(axis=0))
    total_pairs = n_samples * (n_samples - 1.0) / 2.0
    expected = sum_rows * sum_columns / total_pairs
    maximum = 0.5 * (sum_rows + sum_columns)
    denominator = maximum - expected
    if np.isclose(denominator, 0.0, rtol=0.0, atol=1e-15):
        return 1.0
    return float((sum_cells - expected) / denominator)


def variation_of_information(
    first_labels: ArrayLike,
    second_labels: ArrayLike,
    *,
    base: float = 2.0,
) -> float:
    """Variation of Information between two partitions (bits by default)."""

    first = _label_array(first_labels, name="first_labels")
    second = _label_array(second_labels, name="second_labels")
    if first.shape != second.shape:
        raise ValueError("label arrays must have the same shape")
    if not np.isfinite(base) or base <= 0 or base == 1:
        raise ValueError("base must be finite, positive, and different from 1")
    joint = _contingency(first, second).astype(np.float64) / first.size
    row = joint.sum(axis=1)
    column = joint.sum(axis=0)
    positive = joint > 0
    independent = row[:, None] * column[None, :]
    mutual_information = float(
        np.sum(joint[positive] * np.log(joint[positive] / independent[positive]))
        / np.log(base)
    )

    def entropy(probabilities: np.ndarray) -> float:
        probabilities = probabilities[probabilities > 0]
        return float(-np.sum(probabilities * np.log(probabilities)) / np.log(base))

    result = entropy(row) + entropy(column) - 2.0 * mutual_information
    return float(max(0.0, result))


def segmentation_consistency(
    first_boundaries: ArrayLike,
    second_boundaries: ArrayLike,
    grid: ArrayLike,
) -> dict[str, float]:
    """Compute dense-grid ARI and VI for two temporal segmentations."""

    first_labels = boundaries_to_segment_labels(first_boundaries, grid)
    second_labels = boundaries_to_segment_labels(second_boundaries, grid)
    return {
        "ari": adjusted_rand_index(first_labels, second_labels),
        "vi": variation_of_information(first_labels, second_labels),
    }


def selected_timestamp_metrics(
    first: ArrayLike,
    second: ArrayLike,
    *,
    tolerance: float = 1.0,
    first_embeddings: ArrayLike | None = None,
    second_embeddings: ArrayLike | None = None,
) -> dict[str, Any]:
    """Timestamp F1 and optimal displacement for two selected-frame sets.

    F1 uses the supplied tolerance.  ``optimal_mean_distance`` separately uses
    unrestricted max-cardinality/min-distance matching.  If both embedding
    matrices are provided, mean cosine similarity is computed over that optimal
    matching.  Zero-norm matched embeddings are rejected.
    """

    first_array = _sorted_timestamps(first, name="first")
    second_array = _sorted_timestamps(second, name="second")
    tolerant = tolerant_boundary_metrics(first_array, second_array, tolerance=tolerance)
    optimal = match_timestamps(first_array, second_array, tolerance=None)
    result: dict[str, Any] = {
        "precision": tolerant["precision"],
        "recall": tolerant["recall"],
        "f1": tolerant["f1"],
        "matched_within_tolerance": tolerant["matched"],
        "optimal_matched": optimal["count"],
        "optimal_mean_distance": optimal["mean_distance"],
        "first_indices": optimal["first_indices"],
        "second_indices": optimal["second_indices"],
    }
    if (first_embeddings is None) != (second_embeddings is None):
        raise ValueError("first_embeddings and second_embeddings must be supplied together")
    if first_embeddings is not None:
        first_matrix = _finite_float_array(first_embeddings, name="first_embeddings")
        second_matrix = _finite_float_array(second_embeddings, name="second_embeddings")
        if first_matrix.ndim != 2 or second_matrix.ndim != 2:
            raise ValueError("embedding arrays must be two-dimensional")
        if first_matrix.shape[0] != first_array.size or second_matrix.shape[0] != second_array.size:
            raise ValueError("each selected timestamp must have exactly one embedding")
        if first_matrix.shape[1] != second_matrix.shape[1]:
            raise ValueError("embedding dimensions must match")
        if optimal["count"] == 0:
            result["matched_embedding_cosine"] = None
        else:
            matched_first = first_matrix[optimal["first_indices"]]
            matched_second = second_matrix[optimal["second_indices"]]
            first_norm = np.linalg.norm(matched_first, axis=1)
            second_norm = np.linalg.norm(matched_second, axis=1)
            if np.any(first_norm == 0) or np.any(second_norm == 0):
                raise ValueError("cosine similarity is undefined for zero-norm embeddings")
            cosine = np.einsum("ij,ij->i", matched_first, matched_second) / (
                first_norm * second_norm
            )
            result["matched_embedding_cosine"] = float(
                np.mean(np.clip(cosine, -1.0, 1.0))
            )
    return result


def _prediction_array(predictions: ArrayLike) -> np.ndarray:
    array = np.asarray(predictions)
    if array.ndim != 2 or array.shape[0] == 0 or array.shape[1] == 0:
        raise ValueError("predictions must have non-empty shape (n_items, n_origins)")
    for value in array.ravel():
        if value is None or (
            isinstance(value, (float, np.floating)) and not np.isfinite(value)
        ):
            raise ValueError("predictions must not contain missing/NaN/inf values")
    return array


def mllm_stability_metrics(predictions: ArrayLike, gold: ArrayLike) -> dict[str, Any]:
    """Compute accuracy and answer-stability metrics across sampling origins.

    Constant predictions are allowed: agreement measures stability, while the
    accuracy metrics reveal whether that stable answer is useful.  With a
    single origin, AnswerAgree is defined as 1 and PAD as 0.
    """

    prediction_array = _prediction_array(predictions)
    gold_array = np.asarray(gold)
    if gold_array.ndim != 1 or gold_array.size == 0:
        raise ValueError("gold must be a non-empty one-dimensional array")
    if gold_array.size != prediction_array.shape[0]:
        raise ValueError("gold length must equal the number of prediction items")
    for value in gold_array:
        if value is None or (
            isinstance(value, (float, np.floating)) and not np.isfinite(value)
        ):
            raise ValueError("gold must not contain missing/NaN/inf values")

    correct = prediction_array == gold_array[:, None]
    origin_accuracy = np.mean(correct, axis=0, dtype=np.float64)
    mean_accuracy = float(np.mean(origin_accuracy))
    robust_accuracy = float(np.mean(np.all(correct, axis=1)))
    n_origins = prediction_array.shape[1]
    if n_origins == 1:
        item_agreement = np.ones(prediction_array.shape[0], dtype=np.float64)
    else:
        left, right = _pair_indices(n_origins)
        item_agreement = np.mean(
            prediction_array[:, left] == prediction_array[:, right], axis=1
        )
    answer_agreement = float(np.mean(item_agreement))
    return {
        "mean_accuracy": mean_accuracy,
        "robust_accuracy": robust_accuracy,
        "worst_origin_accuracy": float(np.min(origin_accuracy)),
        "accuracy_std": float(np.std(origin_accuracy, ddof=0)),
        "accuracy_variance": float(np.var(origin_accuracy, ddof=0)),
        "answer_agreement": answer_agreement,
        "pairwise_answer_disagreement": float(1.0 - answer_agreement),
        "origin_accuracy": origin_accuracy,
    }


def _cluster_groups(cluster_ids: ArrayLike, expected_size: int) -> list[np.ndarray]:
    ids = np.asarray(cluster_ids)
    if ids.ndim != 1 or ids.size == 0:
        raise ValueError("cluster_ids must be a non-empty one-dimensional array")
    if ids.size != expected_size:
        raise ValueError("cluster_ids length must match the paired observations")
    positions: dict[Hashable, list[int]] = {}
    for index, raw_id in enumerate(ids.tolist()):
        if raw_id is None or (
            isinstance(raw_id, (float, np.floating)) and not np.isfinite(raw_id)
        ):
            raise ValueError("cluster_ids must not contain missing/NaN/inf values")
        try:
            positions.setdefault(raw_id, []).append(index)
        except TypeError as exc:
            raise ValueError("each cluster id must be hashable") from exc
    return [np.asarray(indices, dtype=np.int64) for indices in positions.values()]


def video_cluster_paired_bootstrap(
    cluster_ids: ArrayLike,
    baseline: ArrayLike,
    treatment: ArrayLike,
    *,
    statistic: Callable[[np.ndarray, np.ndarray], float | np.ndarray] | None = None,
    n_bootstrap: int = 10_000,
    confidence: float = 0.95,
    seed: int | np.random.Generator | None = 0,
    return_samples: bool = False,
) -> dict[str, Any]:
    """Paired bootstrap whose top-level resampling unit is video/cluster.

    ``baseline`` and ``treatment`` stay row-paired.  A sampled video contributes
    all of its rows (questions/origins), and a video drawn twice contributes its
    rows twice.  ``statistic`` receives ``(baseline_sample, treatment_sample)``
    and may return a scalar or numeric array.  The default statistic is the
    element-wise mean paired effect ``treatment - baseline``.

    The percentile confidence interval is two-sided.  No finite bootstrap
    result is silently discarded; NaN/inf from a custom statistic raises.
    """

    baseline_array = np.asarray(baseline)
    treatment_array = np.asarray(treatment)
    if baseline_array.ndim == 0 or treatment_array.ndim == 0:
        raise ValueError("baseline and treatment must have an observation axis")
    if baseline_array.shape != treatment_array.shape:
        raise ValueError("baseline and treatment must have identical shapes")
    if baseline_array.shape[0] == 0:
        raise ValueError("paired observations must not be empty")
    groups = _cluster_groups(cluster_ids, baseline_array.shape[0])
    if int(n_bootstrap) != n_bootstrap or n_bootstrap <= 0:
        raise ValueError("n_bootstrap must be a positive integer")
    if not np.isfinite(confidence) or not 0 < confidence < 1:
        raise ValueError("confidence must lie strictly between 0 and 1")

    if statistic is None:
        try:
            numeric_baseline = baseline_array.astype(np.float64)
            numeric_treatment = treatment_array.astype(np.float64)
        except (TypeError, ValueError) as exc:
            raise ValueError("default statistic requires numeric paired observations") from exc
        if not np.all(np.isfinite(numeric_baseline)) or not np.all(
            np.isfinite(numeric_treatment)
        ):
            raise ValueError("paired observations must not contain NaN/inf")

        def statistic_function(a: np.ndarray, b: np.ndarray) -> float:
            return float(np.mean(b.astype(np.float64) - a.astype(np.float64)))

    else:
        if not callable(statistic):
            raise ValueError("statistic must be callable")
        statistic_function = statistic

    def evaluate(indices: np.ndarray) -> np.ndarray:
        try:
            value = np.asarray(
                statistic_function(baseline_array[indices], treatment_array[indices]),
                dtype=np.float64,
            )
        except Exception as exc:
            raise ValueError("statistic failed on paired bootstrap data") from exc
        if value.ndim > 1:
            raise ValueError("statistic must return a scalar or one-dimensional array")
        if value.size == 0 or not np.all(np.isfinite(value)):
            raise ValueError("statistic must return non-empty finite numeric value(s)")
        return value

    full_indices = np.arange(baseline_array.shape[0], dtype=np.int64)
    estimate = evaluate(full_indices)
    if isinstance(seed, np.random.Generator):
        generator = seed
    else:
        generator = np.random.default_rng(seed)
    bootstrap_shape = (int(n_bootstrap),) + estimate.shape
    bootstrap_samples = np.empty(bootstrap_shape, dtype=np.float64)
    n_clusters = len(groups)
    for bootstrap_index in range(int(n_bootstrap)):
        sampled_groups = generator.integers(0, n_clusters, size=n_clusters)
        indices = np.concatenate([groups[index] for index in sampled_groups])
        value = evaluate(indices)
        if value.shape != estimate.shape:
            raise ValueError("statistic returned inconsistent shapes across resamples")
        bootstrap_samples[bootstrap_index] = value

    alpha = (1.0 - confidence) / 2.0
    lower = np.quantile(bootstrap_samples, alpha, axis=0)
    upper = np.quantile(bootstrap_samples, 1.0 - alpha, axis=0)

    def scalar_or_array(value: np.ndarray) -> float | np.ndarray:
        return float(value) if value.ndim == 0 else value

    result: dict[str, Any] = {
        "estimate": scalar_or_array(estimate),
        "ci_low": scalar_or_array(np.asarray(lower)),
        "ci_high": scalar_or_array(np.asarray(upper)),
        "confidence": float(confidence),
        "n_bootstrap": int(n_bootstrap),
        "n_clusters": int(n_clusters),
    }
    if return_samples:
        result["samples"] = bootstrap_samples
    return result


__all__ = [
    "adjusted_rand_index",
    "all_pair_cosine",
    "boundaries_to_segment_labels",
    "boundary_count_cv",
    "js_divergence",
    "make_time_grid",
    "match_timestamps",
    "mllm_stability_metrics",
    "normalized_l1_distance",
    "pairwise_cosine_consistency",
    "pairwise_normalized_l1",
    "scale_energy_drift",
    "scale_energy_proportions",
    "segmentation_consistency",
    "selected_timestamp_metrics",
    "tolerant_boundary_metrics",
    "variation_of_information",
    "video_cluster_paired_bootstrap",
]
