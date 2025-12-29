try:
    import torch
except Exception:  # pragma: no cover - tests will skip if torch missing
    torch = None

from icnn.networks import (
    PartiallyConvexNetwork,
    PartiallyConcaveNetwork,
    validate_monotonicity_randomized,
    validate_convexity_randomized,
    ConstraintSpec,
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
    conc = PartiallyConcaveNetwork(input_dim=2, hidden_sizes=(8,), constraints=constraints)

    # Construct equivalent convex base with flipped monotonicity
    base_constraints = [
        ConstraintSpec(convexity="convex", monotonicity="increasing"),  # flipped dec->inc and negated
        ConstraintSpec(convexity="free", monotonicity="decreasing"),
    ]
    base = PartiallyConvexNetwork(input_dim=2, hidden_sizes=(8,), constraints=base_constraints)

    x = torch.randn(5, 2)
    out_conc = conc(x)
    out_base = base(x)
    # concave wrapper should be negation of the convex base
    assert torch.allclose(out_conc, -out_base, atol=1e-5)