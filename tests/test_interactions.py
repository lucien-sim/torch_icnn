try:
    import torch
except Exception:  # pragma: no cover - tests will skip if torch missing
    torch = None

from torch_icnn.networks import PartiallyMixedNetwork, ConstraintSpec
from torch_icnn.validation import (
    validate_monotonicity_randomized,
    validate_convexity_randomized,
)


def test_partially_mixed_interaction_small():
    if torch is None:
        return
    # dims: 0 convex increasing, 1 concave decreasing, 2 free increasing
    constraints = [
        ConstraintSpec(convexity="convex", monotonicity="increasing"),
        ConstraintSpec(convexity="concave", monotonicity="decreasing"),
        ConstraintSpec(convexity="free", monotonicity="increasing"),
    ]

    torch.manual_seed(0)
    net = PartiallyMixedNetwork(input_dim=3, hidden_sizes=(8, 8), constraints=constraints)
    net.enable_interaction(interaction_hidden_sizes=(4, 4))

    mres = validate_monotonicity_randomized(net, n_samples=256, eps=1e-4, tol_abs=1e-6, tol_rel=1e-6, seed=0)
    assert mres["ok"], f"monotonicity failed after enabling interactions: {mres['failures']}"

    cres = validate_convexity_randomized(net, n_jensen=128, tol_abs=1e-6, tol_rel=1e-6, seed=0)
    assert cres["ok"], f"convexity failed after enabling interactions: {cres['failures']}"


def test_partially_mixed_interaction_gradients():
    if torch is None:
        return
    constraints = [
        ConstraintSpec(convexity="convex", monotonicity="increasing"),
        ConstraintSpec(convexity="concave", monotonicity="decreasing"),
        ConstraintSpec(convexity="free", monotonicity="increasing"),
    ]

    torch.manual_seed(0)
    net = PartiallyMixedNetwork(input_dim=3, hidden_sizes=(8, 8), constraints=constraints)
    net.enable_interaction(interaction_hidden_sizes=(4, 4))

    x = torch.randn(10, 3, requires_grad=True)
    y = net(x)
    grads = torch.autograd.grad(y.sum(), x)[0]

    # convex increasing ==> derivative >= -tol
    assert grads[:, 0].min().item() >= -1e-5
    # concave decreasing ==> derivative <= tol
    assert grads[:, 1].max().item() <= 1e-5
    # free increasing ==> derivative >= -tol
    assert grads[:, 2].min().item() >= -1e-5
