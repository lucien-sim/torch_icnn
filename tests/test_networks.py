try:
    import torch
except Exception:  # pragma: no cover - tests will skip if torch missing
    torch = None

from torch_icnn.networks import (
    PartiallyConvexNetwork,
    PartiallyConcaveNetwork,
    PartiallyMixedNetwork,
    ConstraintSpec,
)
from torch_icnn.validation import (
    validate_monotonicity_randomized,
    validate_convexity_randomized,
)


def test_monotone_convex_small():
    if torch is None:
        return
    constraints = [
        ConstraintSpec(convexity="convex", monotonicity="increasing"),
        ConstraintSpec(convexity="free", monotonicity="increasing"),
    ]
    net = PartiallyConvexNetwork(input_dim=2, hidden_sizes=(8, 8), constraints=constraints)

    mres = validate_monotonicity_randomized(net, n_samples=512, eps=1e-4, tol_abs=1e-6, tol_rel=1e-6, seed=0)
    assert mres["ok"], f"monotonicity failed: {mres['failures']}"

    cres = validate_convexity_randomized(net, n_jensen=256, tol_abs=1e-6, tol_rel=1e-6, seed=0)
    assert cres["ok"], f"convexity failed: {cres['failures']}"


def test_uses_concave_subnetwork():
    if torch is None:
        return
    constraints = [
        ConstraintSpec(convexity="concave", monotonicity="decreasing"),
        ConstraintSpec(convexity="free", monotonicity="increasing"),
    ]
    net = PartiallyConcaveNetwork(input_dim=2, hidden_sizes=(8,), constraints=constraints)
    from torch_icnn.networks import ConcaveSubnetwork, opp_monotonicity
    assert isinstance(net._conc, ConcaveSubnetwork)

    # verify the concave subnetwork flipped monotonicities internally
    expected_convex = [opp_monotonicity(constraints[0].monotonicity)]
    expected_free = [opp_monotonicity(constraints[1].monotonicity)]
    assert net._conc._conv.block.mono_list_convex == expected_convex
    assert net._conc._conv.block.mono_list_free == expected_free


def test_mixed_uses_concave_subnetwork():
    if torch is None:
        return
    constraints = [
        ConstraintSpec(convexity="convex", monotonicity="increasing"),
        ConstraintSpec(convexity="concave", monotonicity="decreasing"),
        ConstraintSpec(convexity="free", monotonicity="increasing"),
    ]

    net = PartiallyMixedNetwork(input_dim=3, hidden_sizes=(8, 8), constraints=constraints)
    from torch_icnn.networks import ConcaveSubnetwork, opp_monotonicity
    assert isinstance(net._h, ConcaveSubnetwork)

    # verify monotonicities were flipped inside the concave subnetwork
    expected_h_convex = [opp_monotonicity(constraints[1].monotonicity)]
    expected_h_free = [opp_monotonicity(constraints[2].monotonicity)]
    assert net._h._conv.block.mono_list_convex == expected_h_convex
    assert net._h._conv.block.mono_list_free == expected_h_free


def test_gradients_signs():
    if torch is None:
        return
    constraints = [
        ConstraintSpec(convexity="convex", monotonicity="increasing"),
        ConstraintSpec(convexity="free", monotonicity="increasing"),
    ]
    net = PartiallyConvexNetwork(input_dim=2, hidden_sizes=(8, 8), constraints=constraints)
    x = torch.randn(10, 2, requires_grad=True)
    y = net(x)
    grads = torch.autograd.grad(y.sum(), x)[0]
    # both partial derivatives should be >= -small_tol
    assert grads[:, 0].min().item() >= -1e-5
    assert grads[:, 1].min().item() >= -1e-5


def test_partially_concave_wrapper_consistency():
    if torch is None:
        return
    # concave (decreasing) on first input, free increasing on second
    constraints = [
        ConstraintSpec(convexity="concave", monotonicity="decreasing"),
        ConstraintSpec(convexity="free", monotonicity="increasing"),
    ]
    torch.manual_seed(0)
    conc = PartiallyConcaveNetwork(input_dim=2, hidden_sizes=(8,), constraints=constraints)

    # Construct equivalent convex base with flipped monotonicity
    base_constraints = [
        ConstraintSpec(convexity="convex", monotonicity="increasing"),  # flipped dec->inc and negated
        ConstraintSpec(convexity="free", monotonicity="decreasing"),
    ]
    torch.manual_seed(0)
    base = PartiallyConvexNetwork(input_dim=2, hidden_sizes=(8,), constraints=base_constraints)

    x = torch.randn(5, 2)
    out_conc = conc(x)
    out_base = base(x)
    # concave wrapper should be negation of the convex base
    assert torch.allclose(out_conc, -out_base, atol=1e-5)


def test_partially_mixed_small():
    if torch is None:
        return
    # dims: 0 convex increasing, 1 concave decreasing, 2 free increasing
    constraints = [
        ConstraintSpec(convexity="convex", monotonicity="increasing"),
        ConstraintSpec(convexity="concave", monotonicity="decreasing"),
        ConstraintSpec(convexity="free", monotonicity="increasing"),
    ]

    net = PartiallyMixedNetwork(input_dim=3, hidden_sizes=(8, 8), constraints=constraints)

    mres = validate_monotonicity_randomized(net, n_samples=512, eps=1e-4, tol_abs=1e-6, tol_rel=1e-6, seed=0)
    assert mres["ok"], f"monotonicity failed: {mres['failures']}"

    cres = validate_convexity_randomized(net, n_jensen=256, tol_abs=1e-6, tol_rel=1e-6, seed=0)
    assert cres["ok"], f"convexity failed: {cres['failures']}"