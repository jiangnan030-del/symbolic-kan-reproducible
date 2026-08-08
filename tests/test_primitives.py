import torch

from symbolic_kan.primitives import apply_primitive


def test_inverse_is_finite_at_zero() -> None:
    value = apply_primitive("inv", torch.tensor([0.0], dtype=torch.float64))
    assert torch.isfinite(value).all()


def test_supported_primitives_keep_shape() -> None:
    x = torch.linspace(-0.5, 0.5, 7, dtype=torch.float64)
    for name in ("x", "x2", "sin", "cos", "exp", "log1p", "lorentz"):
        assert apply_primitive(name, x).shape == x.shape
