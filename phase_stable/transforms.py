"""Comparable DWT, stationary-wavelet, and cycle-spinning transforms.

The experiment represents every wavelet band in the original time coordinate.
This avoids comparing critically sampled coefficient arrays with redundant SWT
arrays that have different shapes.  Rows of ``representation`` are ordered
from fine (level 1) to coarse (level J); the final row is the WFS-SB boundary
signal before taking its absolute value.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Literal, Optional, Protocol, Sequence, Tuple

import numpy as np
import pywt
from scipy.ndimage import gaussian_filter1d


TransformName = Literal["dwt", "swt", "cycle_spin", "gaussian"]


@dataclass(frozen=True)
class TransformConfig:
    """Configuration shared by all temporal wavelet transforms."""

    method: TransformName = "dwt"
    wavelet: str = "db4"
    level: int = 1
    dwt_mode: str = "symmetric"
    shared_padding: bool = True
    padding_mode: str = "reflect"
    swt_norm: bool = True
    cycle_shifts: Tuple[int, ...] = tuple(range(16))
    cycle_aggregation: Literal["mean", "median"] = "mean"
    gaussian_sigma: Optional[float] = None


@dataclass
class TransformResult:
    """Time-aligned representation produced by a configured transform."""

    method: str
    representation: np.ndarray
    coarse_detail: np.ndarray
    saliency: np.ndarray
    scale_energy_proportions: np.ndarray
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.representation = np.asarray(self.representation, dtype=float)
        self.coarse_detail = np.asarray(self.coarse_detail, dtype=float)
        self.saliency = np.asarray(self.saliency, dtype=float)
        self.scale_energy_proportions = np.asarray(
            self.scale_energy_proportions, dtype=float
        )
        if self.representation.ndim != 2:
            raise ValueError("representation must have shape [level, time]")
        if self.coarse_detail.shape != (self.representation.shape[1],):
            raise ValueError("coarse_detail must align with the time dimension")
        if self.saliency.shape != self.coarse_detail.shape:
            raise ValueError("saliency must align with coarse_detail")

    def summary(self) -> Dict[str, Any]:
        """Return JSON-friendly metadata without the large signal arrays."""

        return {
            "method": self.method,
            "num_levels": int(self.representation.shape[0]),
            "num_samples": int(self.representation.shape[1]),
            "scale_energy_proportions": self.scale_energy_proportions.tolist(),
            "metadata": dict(self.metadata),
        }

    @property
    def normalized_saliency(self) -> np.ndarray:
        """Return saliency as a probability mass for cross-origin metrics.

        A zero detail signal remains all-zero and is explicitly diagnosed by
        the analysis layer instead of being hidden behind an epsilon offset.
        """

        total = float(np.sum(self.saliency))
        if total <= np.finfo(float).eps:
            return np.zeros_like(self.saliency, dtype=float)
        return self.saliency / total


class TemporalTransform(Protocol):
    """Protocol implemented by all transform variants."""

    config: TransformConfig

    def transform(self, signal: Sequence[float]) -> TransformResult:
        """Transform one finite one-dimensional signal."""


def _as_signal(signal: Sequence[float]) -> np.ndarray:
    values = np.asarray(signal, dtype=float)
    if values.ndim != 1:
        raise ValueError("signal must be one-dimensional")
    if values.size < 2:
        raise ValueError("signal must contain at least two samples")
    if not np.all(np.isfinite(values)):
        raise ValueError("signal must contain only finite values")
    return values


def _validate_common(config: TransformConfig, signal_length: int) -> None:
    if config.level < 1:
        raise ValueError("level must be at least 1")
    try:
        pywt.Wavelet(config.wavelet)
    except Exception as exc:  # pragma: no cover - exact PyWavelets error varies
        raise ValueError(f"unknown wavelet: {config.wavelet}") from exc
    if 2**config.level > 1_048_576:
        raise ValueError("level is unreasonably large for an experiment signal")
    if config.cycle_aggregation not in {"mean", "median"}:
        raise ValueError("cycle_aggregation must be 'mean' or 'median'")


def _pad_to_level(
    signal: np.ndarray,
    level: int,
    mode: str,
    minimum_length: int = 0,
) -> Tuple[np.ndarray, int, int]:
    multiple = 2**level
    target_length = max(signal.size, int(minimum_length))
    target_length = ((target_length + multiple - 1) // multiple) * multiple
    total = target_length - signal.size
    left = total // 2
    right = total - left
    if total == 0:
        return signal.copy(), 0, 0
    actual_mode = mode
    if signal.size == 1 and mode in {"reflect", "symmetric"}:
        actual_mode = "edge"
    try:
        padded = np.pad(signal, (left, right), mode=actual_mode)
    except ValueError as exc:
        raise ValueError(f"invalid padding_mode={mode!r}") from exc
    return np.asarray(padded, dtype=float), left, right


def _crop(values: np.ndarray, left: int, original_length: int) -> np.ndarray:
    return np.asarray(values[..., left : left + original_length], dtype=float)


def _safe_wavelet_length(config: TransformConfig) -> int:
    wavelet = pywt.Wavelet(config.wavelet)
    return (wavelet.dec_len - 1) * (2**config.level)


def _energy_proportions(representation: np.ndarray) -> np.ndarray:
    energies = np.sum(np.square(representation), axis=1)
    total = float(np.sum(energies))
    if total <= np.finfo(float).eps:
        return np.zeros_like(energies, dtype=float)
    return energies / total


def _make_result(
    method: str,
    representation: np.ndarray,
    metadata: Dict[str, Any],
) -> TransformResult:
    representation = np.asarray(representation, dtype=float)
    coarse = representation[-1].copy()
    return TransformResult(
        method=method,
        representation=representation,
        coarse_detail=coarse,
        saliency=np.abs(coarse),
        scale_energy_proportions=_energy_proportions(representation),
        metadata=metadata,
    )


class DWTTransform:
    """Critically sampled DWT reconstructed onto the input time grid."""

    def __init__(self, config: TransformConfig) -> None:
        self.config = config

    def transform(self, signal: Sequence[float]) -> TransformResult:
        values = _as_signal(signal)
        _validate_common(self.config, values.size)
        if self.config.shared_padding:
            padded, left, right = _pad_to_level(
                values,
                self.config.level,
                self.config.padding_mode,
                _safe_wavelet_length(self.config),
            )
        else:
            padded, left, right = values.copy(), 0, 0

        coeffs = pywt.wavedec(
            padded,
            self.config.wavelet,
            level=self.config.level,
            mode=self.config.dwt_mode,
        )
        details = []
        for detail_level in range(1, self.config.level + 1):
            keep_index = len(coeffs) - detail_level
            isolated = [np.zeros_like(coeff) for coeff in coeffs]
            isolated[keep_index] = coeffs[keep_index]
            reconstructed = pywt.waverec(
                isolated, self.config.wavelet, mode=self.config.dwt_mode
            )[: padded.size]
            details.append(_crop(reconstructed, left, values.size))

        representation = np.stack(details, axis=0)
        return _make_result(
            "dwt",
            representation,
            {
                "wavelet": self.config.wavelet,
                "level": self.config.level,
                "dwt_mode": self.config.dwt_mode,
                "shared_padding": self.config.shared_padding,
                "padding_mode": self.config.padding_mode,
                "padding_left": left,
                "padding_right": right,
                "original_length": int(values.size),
                "padded_length": int(padded.size),
                "pywavelets_version": pywt.__version__,
            },
        )


class SWTTransform:
    """Undecimated stationary wavelet transform (the primary TI-DWT)."""

    def __init__(self, config: TransformConfig) -> None:
        self.config = config

    def transform(self, signal: Sequence[float]) -> TransformResult:
        values = _as_signal(signal)
        _validate_common(self.config, values.size)
        padded, left, right = _pad_to_level(
            values,
            self.config.level,
            self.config.padding_mode,
            _safe_wavelet_length(self.config),
        )
        coeffs = pywt.swt(
            padded,
            self.config.wavelet,
            level=self.config.level,
            trim_approx=True,
            norm=self.config.swt_norm,
        )
        details = []
        for detail_level in range(1, self.config.level + 1):
            keep_index = len(coeffs) - detail_level
            isolated = [np.zeros_like(coeff) for coeff in coeffs]
            isolated[keep_index] = coeffs[keep_index]
            reconstructed = pywt.iswt(
                isolated,
                self.config.wavelet,
                norm=self.config.swt_norm,
            )
            details.append(_crop(reconstructed, left, values.size))

        representation = np.stack(details, axis=0)
        return _make_result(
            "swt",
            representation,
            {
                "wavelet": self.config.wavelet,
                "level": self.config.level,
                "swt_norm": self.config.swt_norm,
                "padding_mode": self.config.padding_mode,
                "padding_left": left,
                "padding_right": right,
                "original_length": int(values.size),
                "padded_length": int(padded.size),
                "pywavelets_version": pywt.__version__,
            },
        )


class CycleSpinTransform:
    """Aggregate aligned DWT reconstructions over a fixed shift set."""

    def __init__(self, config: TransformConfig) -> None:
        self.config = config
        shifts = tuple(dict.fromkeys(int(shift) for shift in config.cycle_shifts))
        if not shifts:
            raise ValueError("cycle_shifts must not be empty")
        self.shifts = shifts

    def transform(self, signal: Sequence[float]) -> TransformResult:
        values = _as_signal(signal)
        _validate_common(self.config, values.size)
        aligned = []
        base_config = TransformConfig(
            method="dwt",
            wavelet=self.config.wavelet,
            level=self.config.level,
            dwt_mode=self.config.dwt_mode,
            shared_padding=self.config.shared_padding,
            padding_mode=self.config.padding_mode,
        )
        base = DWTTransform(base_config)
        for shift in self.shifts:
            shifted = np.roll(values, shift)
            representation = base.transform(shifted).representation
            aligned.append(np.roll(representation, -shift, axis=1))

        stack = np.stack(aligned, axis=0)
        if self.config.cycle_aggregation == "median":
            representation = np.median(stack, axis=0)
        else:
            representation = np.mean(stack, axis=0)
        return _make_result(
            "cycle_spin",
            representation,
            {
                "wavelet": self.config.wavelet,
                "level": self.config.level,
                "dwt_mode": self.config.dwt_mode,
                "shared_padding": self.config.shared_padding,
                "padding_mode": self.config.padding_mode,
                "cycle_shifts": list(self.shifts),
                "cycle_aggregation": self.config.cycle_aggregation,
                "original_length": int(values.size),
                "pywavelets_version": pywt.__version__,
            },
        )


class GaussianDerivativeTransform:
    """Non-wavelet smoothing control using a first Gaussian derivative."""

    def __init__(self, config: TransformConfig) -> None:
        self.config = config

    def transform(self, signal: Sequence[float]) -> TransformResult:
        values = _as_signal(signal)
        sigma = (
            float(self.config.gaussian_sigma)
            if self.config.gaussian_sigma is not None
            else float(2 ** max(0, self.config.level - 1))
        )
        if not np.isfinite(sigma) or sigma <= 0:
            raise ValueError("gaussian_sigma must be finite and positive")
        derivative = gaussian_filter1d(
            values,
            sigma=sigma,
            order=1,
            mode="reflect",
        )
        return _make_result(
            "gaussian",
            derivative[None, :],
            {
                "sigma": sigma,
                "order": 1,
                "mode": "reflect",
                "original_length": int(values.size),
            },
        )


def build_transform(config: TransformConfig) -> TemporalTransform:
    """Construct a temporal transform from a validated configuration."""

    if config.method == "dwt":
        return DWTTransform(config)
    if config.method == "swt":
        return SWTTransform(config)
    if config.method == "cycle_spin":
        return CycleSpinTransform(config)
    if config.method == "gaussian":
        return GaussianDerivativeTransform(config)
    raise ValueError(f"unsupported transform method: {config.method}")
