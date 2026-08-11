import numpy as np
import pytest

from phase_stable.pipeline import PhaseStableWFS, select_top_nms_indices
from phase_stable.transforms import TransformConfig, build_transform
from wfs.core import WFS, WFSConfig


def _piecewise_signal(length: int = 256) -> np.ndarray:
    signal = np.zeros(length, dtype=float)
    signal[length // 4 : length // 2] = 0.8
    signal[length // 2 : 3 * length // 4] = 0.2
    signal[3 * length // 4 :] = 1.0
    signal += 0.03 * np.sin(np.linspace(0, 14 * np.pi, length))
    return signal


@pytest.mark.parametrize("method", ["dwt", "swt", "cycle_spin", "gaussian"])
def test_transform_shapes_and_energy(method: str) -> None:
    signal = _piecewise_signal(256)
    config = TransformConfig(method=method, level=4, cycle_shifts=tuple(range(8)))
    result = build_transform(config).transform(signal)

    expected_levels = 1 if method == "gaussian" else 4
    assert result.representation.shape == (expected_levels, 256)
    assert result.coarse_detail.shape == (256,)
    assert result.saliency.shape == (256,)
    assert np.all(result.saliency >= 0)
    assert np.isclose(np.sum(result.scale_energy_proportions), 1.0)


def test_swt_is_equivariant_away_from_padding() -> None:
    signal = _piecewise_signal(256)
    transform = build_transform(TransformConfig(method="swt", level=4))
    reference = transform.transform(signal).representation
    shifted = transform.transform(np.roll(signal, 5)).representation
    aligned = np.roll(shifted, -5, axis=1)

    np.testing.assert_allclose(reference, aligned, atol=1e-10, rtol=1e-10)


def test_native_dwt_pipeline_matches_original_wfs() -> None:
    signal = _piecewise_signal(256)
    # Match the official pipeline CLI defaults used by PhaseStableWFS.  The
    # upstream WFSConfig programmatic defaults differ from its CLI defaults.
    original = WFS(
        WFSConfig(
            w_duration=0.4,
            w_mean=0.2,
            w_max=0.3,
            w_var=0.1,
            strictness_factor=1.2,
        )
    )
    expected = original.select_keyframes(
        signal,
        num_frames=16,
        dwt_level=4,
        min_peak_distance=5,
        features=None,
    )
    transform = build_transform(
        TransformConfig(method="dwt", level=4, shared_padding=False)
    )
    actual = PhaseStableWFS(transform).run(
        signal,
        num_frames=16,
        min_peak_distance=5,
        features=None,
    )
    assert actual.selected_indices == expected


def test_invalid_transform_input_is_rejected() -> None:
    transform = build_transform(TransformConfig(method="swt", level=2))
    with pytest.raises(ValueError, match="finite"):
        transform.transform([0.0, np.nan, 1.0, 2.0])


@pytest.mark.parametrize("method", ["dwt", "swt"])
def test_short_signal_gets_common_filter_safe_padding(method: str) -> None:
    result = build_transform(TransformConfig(method=method, level=2)).transform(
        np.linspace(0.0, 1.0, 11)
    )
    assert result.representation.shape == (2, 11)
    assert result.metadata["padded_length"] >= 28
    assert result.metadata["padding_left"] + result.metadata["padding_right"] >= 17


def test_trace_is_json_friendly() -> None:
    signal = _piecewise_signal(256)
    transform = build_transform(TransformConfig(method="swt", level=4))
    trace = PhaseStableWFS(transform).run(signal, 16, 5)
    payload = trace.to_dict(include_arrays=True)

    assert payload["transform"]["method"] == "swt"
    assert len(payload["representation"]) == 4
    assert len(payload["selected_indices"]) == 16


def test_top_nms_indices_is_exact_deterministic_and_excludes_endpoints() -> None:
    values = np.array([99.0, 3.0, 3.0, 1.0, 8.0, 2.0, 7.0, 99.0])
    selected = select_top_nms_indices(values, count=3, min_distance=2)
    np.testing.assert_array_equal(selected, [1, 4, 6])


def test_top_nms_indices_finds_feasible_exact_set_that_greedy_misses() -> None:
    # Greedy selection starts at index 2 and cannot add another point at d=3,
    # although the exact feasible pair (1, 4) exists.
    values = np.array([0.0, 4.0, 10.0, 1.0, 4.0, 0.0])
    selected = select_top_nms_indices(values, count=2, min_distance=3)
    np.testing.assert_array_equal(selected, [1, 4])


@pytest.mark.parametrize("method", ["dwt", "swt"])
def test_matched_boundary_count_reuses_unchanged_selection_pipeline(method: str) -> None:
    signal = _piecewise_signal(256)
    transform = build_transform(TransformConfig(method=method, level=4))
    trace = PhaseStableWFS(transform).run(
        signal,
        num_frames=16,
        min_peak_distance=5,
        boundary_count=4,
    )
    assert len(trace.peaks) == 4
    assert len(trace.segments) == 5
    assert len(trace.selected_indices) == 16
    assert trace.peaks[0] > 0
    assert trace.peaks[-1] < signal.size - 1
