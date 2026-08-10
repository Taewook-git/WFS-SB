"""Validated YAML configuration loader for phase-stability experiments."""

from __future__ import annotations

import importlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from .analysis import ExperimentConfig
from .pipeline import SelectionConfig


@dataclass(frozen=True)
class PhaseStableConfig:
    experiment: ExperimentConfig
    selection: SelectionConfig
    sampling: Mapping[str, Any]
    metadata: Mapping[str, Any]


def _mapping(payload: Mapping[str, Any], key: str) -> dict[str, Any]:
    value = payload.get(key, {})
    if not isinstance(value, Mapping):
        raise ValueError(f"config section {key!r} must be a mapping")
    return dict(value)


def load_phase_stable_config(path: str | Path) -> PhaseStableConfig:
    try:
        yaml = importlib.import_module("yaml")
    except (ImportError, ModuleNotFoundError) as exc:
        raise RuntimeError("PyYAML is required to read experiment config files") from exc
    source = Path(path)
    with source.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    if not isinstance(payload, Mapping):
        raise ValueError("experiment config root must be a mapping")

    experiment_values = _mapping(payload, "experiment")
    for tuple_field in (
        "methods",
        "cycle_shifts",
        "report_boundary_tolerances_sec",
    ):
        if tuple_field in experiment_values:
            value = experiment_values[tuple_field]
            if not isinstance(value, (list, tuple)):
                raise ValueError(f"experiment.{tuple_field} must be a sequence")
            experiment_values[tuple_field] = tuple(value)
    try:
        experiment = ExperimentConfig(**experiment_values)
    except TypeError as exc:
        raise ValueError(f"unknown or invalid experiment config field: {exc}") from exc

    selection_values = _mapping(payload, "selection")
    try:
        selection = SelectionConfig(**selection_values)
    except TypeError as exc:
        raise ValueError(f"unknown or invalid selection config field: {exc}") from exc
    return PhaseStableConfig(
        experiment=experiment,
        selection=selection,
        sampling=_mapping(payload, "sampling"),
        metadata=_mapping(payload, "metadata"),
    )


__all__ = ["PhaseStableConfig", "load_phase_stable_config"]
