from pathlib import Path

from symbolic_kan import load_experiment_config


def test_smoke_config_loads() -> None:
    path = Path(__file__).parents[1] / "experiments/reaction_diffusion/configs/smoke.yaml"
    config = load_experiment_config(path)
    assert config.profile == "smoke"
    assert config.model.readout == "fixed_sum"
    assert config.provenance["upstream_commit"].startswith("9481a82")
