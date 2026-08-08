from symbolic_kan import ModelConfig, SymbolicKAN, TrainingConfig
from symbolic_kan.training import build_adamw


def test_configured_weight_decay_reaches_optimizer() -> None:
    model = SymbolicKAN(ModelConfig(hidden_units=2, edges_per_unit=2, num_blocks=1))
    optimizer = build_adamw(
        model,
        TrainingConfig(adam_epochs=1, weight_decay=0.123, gate_weight_decay=0.456),
    )
    decays = {group["weight_decay"] for group in optimizer.param_groups}
    assert 0.123 in decays
    assert 0.456 in decays
