import json
from pathlib import Path

from phase_stable.repro import sha256_file, write_reproducibility_manifests


def test_reproducibility_manifests_hash_inputs(tmp_path: Path) -> None:
    source = tmp_path / "input.jsonl"
    source.write_text('{"x": 1}\n', encoding="utf-8")
    run_path, environment_path = write_reproducibility_manifests(
        tmp_path / "run",
        command="test",
        config={"seed": 7},
        input_paths=[source],
        repo_root=tmp_path,
    )
    payload = json.loads(run_path.read_text(encoding="utf-8"))
    assert payload["inputs"][0]["sha256"] == sha256_file(source)
    assert payload["config"] == {"seed": 7}
    assert environment_path.is_file()
