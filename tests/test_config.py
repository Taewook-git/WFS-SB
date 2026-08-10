from pathlib import Path

from phase_stable.config import load_phase_stable_config


def test_checked_in_config_is_loadable() -> None:
    root = Path(__file__).resolve().parents[1]
    config = load_phase_stable_config(root / "configs" / "phase_stable_icassp.yaml")
    assert config.experiment.methods == ("dwt", "swt")
    assert config.experiment.wavelet == "db4"
    assert config.selection.w_duration == 0.4
    assert config.sampling["num_origins"] == 5
