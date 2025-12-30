try:
    import torch
except Exception:  # pragma: no cover - tests will skip if torch missing
    torch = None

from torch_icnn.networks import (
    PartiallyMixedNetwork,
    PartiallyConvexNetwork,
    PartiallyConcaveNetwork,
    ConstraintSpec,
)
from torch_icnn.validation import (
    validate_monotonicity_randomized,
    validate_convexity_randomized,
)


def test_all_free_inputs():
    if torch is None:
        return
    constraints = [ConstraintSpec(convexity="free", monotonicity="free") for _ in range(3)]
    net = PartiallyMixedNetwork(input_dim=3, hidden_sizes=(8, 8), constraints=constraints)

    # No convex/concave subsets -> validators should return ok
    mres = validate_monotonicity_randomized(net, n_samples=256, seed=0)
    assert mres["ok"], f"monotonicity failed for all-free: {mres['failures']}"

    cres = validate_convexity_randomized(net, n_jensen=128, seed=0)
    assert cres["ok"], f"convexity failed for all-free: {cres['failures']}"


def test_all_concave_inputs():
    if torch is None:
        return
    constraints = [ConstraintSpec(convexity="concave", monotonicity="increasing") for _ in range(2)]
    net = PartiallyMixedNetwork(input_dim=2, hidden_sizes=(8,), constraints=constraints)

    # Concave tests should run (Jensen with sign -1)
    cres = validate_convexity_randomized(net, n_jensen=128, seed=0)
    assert cres["ok"], f"convexity (concave) failed: {cres['failures']}"

    # Gradients signs: increasing on concave means derivative should be >= 0
    x = torch.randn(8, 2, requires_grad=True)
    y = net(x)
    grads = torch.autograd.grad(y.sum(), x)[0]
    assert grads.min().item() >= -1e-5


def test_no_free_inputs_mixed():
    if torch is None:
        return
    # convex on idx0, concave on idx1, no free inputs
    constraints = [
        ConstraintSpec(convexity="convex", monotonicity="increasing"),
        ConstraintSpec(convexity="concave", monotonicity="decreasing"),
    ]
    net = PartiallyMixedNetwork(input_dim=2, hidden_sizes=(8,), constraints=constraints)

    mres = validate_monotonicity_randomized(net, n_samples=256, seed=0)
    assert mres["ok"], f"monotonicity failed for no-free: {mres['failures']}"

    cres = validate_convexity_randomized(net, n_jensen=128, seed=0)
    assert cres["ok"], f"convexity failed for no-free: {cres['failures']}"


def test_single_dim_convex_and_concave():
    if torch is None:
        return
    # single convex dimension
    net_c = PartiallyMixedNetwork(input_dim=1, hidden_sizes=(8,), constraints=[ConstraintSpec(convexity="convex")])
    cres = validate_convexity_randomized(net_c, n_jensen=128, seed=0)
    assert cres["ok"], "single-dim convex failed"

    # single concave dimension
    net_cc = PartiallyMixedNetwork(input_dim=1, hidden_sizes=(8,), constraints=[ConstraintSpec(convexity="concave")])
    cres2 = validate_convexity_randomized(net_cc, n_jensen=128, seed=0)
    assert cres2["ok"], "single-dim concave failed"