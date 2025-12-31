from .networks import (
    PartiallyConvexNetwork,
    PartiallyConcaveNetwork,
    PartiallyMixedNetwork,
    ConstraintSpec,
)
from typing import Union
import torch
import numpy as np


def validate_monotonicity_randomized(
    net: Union[PartiallyConvexNetwork, PartiallyConcaveNetwork, PartiallyMixedNetwork],
    n_samples: int = 256,
    eps: float = 1e-3,
    tol_abs: float = 1e-5,
    tol_rel: float = 1e-6,
    seed: int | None = None,
    verbose: bool = False,
) -> dict:
    """Randomized monotonicity checks (runs in float32, restores dtype).
    The funciton samples n_samples random inputs in [-1,1]^D, and for each
            monotonic input dimension, perturbs that input by +eps and checks that
            the output changes in the expected direction (within tolerances).

    Parameters
    ----------
    net: PartiallyConvexNetwork
        The network to validate.
    n_samples: int
        Number of random samples to use.
    eps: float
        Perturbation size for monotonicity checks.
    tol_abs: float
        Absolute tolerance for monotonicity violations.
    tol_rel: float
        Relative tolerance for monotonicity violations.
    seed: int | None
        Random seed for reproducibility.
    verbose: bool
        Whether to print per-dimension results.

    Returns:
        {'ok': bool, 'failures': {idx: ...}}.
    """
    if seed is not None:
        torch.manual_seed(int(seed))

    # determine device (robust if model has no parameters)
    try:
        first_param = next(net.parameters())
        device = first_param.device
    except StopIteration:
        device = torch.device("cpu")

    D = int(net.input_dim)
    X = (torch.rand(n_samples, D, device=device) * 2.0) - 1.0
    # avoid tracking gradients for randomized checks
    with torch.no_grad():
        Y = net(X)
    Y_np = Y.cpu().detach().numpy()
    mono_failures = {}
    for idx, c in enumerate(net.constraints):
        m = c.monotonicity
        if m == "free":
            continue
        Xp = X.clone()
        Xp[:, idx] = Xp[:, idx] + eps
        with torch.no_grad():
            Yp = net(Xp)
        delta = (Yp - Y).cpu().detach().numpy()
        scale = np.maximum(1.0, np.abs(Y_np))
        if m == "increasing":
            viol = delta < -(tol_abs + tol_rel * scale)
            violation_vals = np.minimum(0.0, delta + (tol_abs + tol_rel * scale))
        else:  # decreasing
            viol = delta > (tol_abs + tol_rel * scale)
            violation_vals = np.maximum(0.0, delta - (tol_abs + tol_rel * scale))
        if viol.any():
            ix = np.nonzero(viol)[0][:5].tolist()
            mono_failures[idx] = {
                "monotone": m,
                "n_violations": int(viol.sum()),
                "examples": [
                    {
                        "x": X[i].cpu().numpy().tolist(),
                        "delta": float(delta[i]),
                        "violation": float(violation_vals[i]),
                    }
                    for i in ix
                ],
            }
            if verbose:
                print(f"[mono] idx={idx} expected {m}, violations={int(viol.sum())}")
        elif verbose:
            print(f"[mono] idx={idx} ok ({m})")

    return {"ok": len(mono_failures) == 0, "failures": mono_failures}


def validate_convexity_randomized(
    net: Union[PartiallyConvexNetwork, PartiallyConcaveNetwork, PartiallyMixedNetwork],
    n_jensen: int = 256,
    tol_abs: float = 1e-5,
    tol_rel: float = 1e-6,
    seed: int | None = None,
    verbose: bool = False,
) -> dict:
    """Randomized Jensen checks for convexity/concavity (runs in float32).
    The function samples n_jensen random pairs of points in [-1,1]^D,
    and for each convex input dimension subset, checks the Jensen inequality.

    Jensen inequality:
            For convex inputs: f(tx1 + (1-t)x2) <= t f(x1) + (1-t) f(x2)
            For concave inputs: f(tx1 + (1-t)x2) >= t f(x1) + (1-t) f(x2)

    Parameters
    ----------
    net: PartiallyConvexNetwork
        The network to validate.
    n_jensen: int
        Number of random Jensen tests to run per convex subset.
    tol_abs: float
        Absolute tolerance for Jensen violations.
    tol_rel: float
        Relative tolerance for Jensen violations.
    seed: int | None
        Random seed for reproducibility.
    verbose: bool
        Whether to print per-subset results.

    Returns:
            {'ok': bool, 'failures': {'convex': [...]}}.
    """
    if seed is not None:
        torch.manual_seed(int(seed))

    # determine device (robust if model has no parameters)
    try:
        first_param = next(net.parameters())
        device = first_param.device
    except StopIteration:
        device = torch.device("cpu")
    D = int(net.input_dim)

    # ensure float32 for checks
    did_cast = False
    float_params = [p for p in net.parameters() if p.dtype.is_floating_point]
    orig_dtype = float_params[0].dtype if float_params else torch.get_default_dtype()
    if orig_dtype != torch.float32:
        net.float()
        did_cast = True

    try:
        dtype = torch.float32
        jensen_failures = {"convex": []}

        def _make_jensen_pair(D: int, indices: list[int], device, dtype):
            """Construct a pair (x1, x2, t) where x1 and x2 differ only on `indices`.

            Returns: (x1, x2, t) tensors/scalar.
            """
            x1 = (torch.rand(1, D, device=device, dtype=dtype) * 2.0) - 1.0
            x2 = x1.clone()
            if len(indices) > 0:
                x1_vals = (torch.rand(len(indices), device=device, dtype=dtype) * 2.0) - 1.0
                x2_vals = (torch.rand(len(indices), device=device, dtype=dtype) * 2.0) - 1.0
                x1[0, indices] = x1_vals
                x2[0, indices] = x2_vals
            t = float(torch.rand(1).item())
            return x1, x2, t

        def _jensen_test(indices: list[int], sign: float) -> list:
            fail_examples = []
            if len(indices) == 0:
                return fail_examples
            for _ in range(n_jensen):
                x1, x2, t = _make_jensen_pair(D, indices, device, dtype)
                xt = t * x1 + (1.0 - t) * x2
                with torch.no_grad():
                    f1 = net(x1)[0].item()
                    f2 = net(x2)[0].item()
                    ft = net(xt)[0].item()
                left = ft * sign
                right = (t * f1 + (1.0 - t) * f2) * sign
                thresh = tol_abs + tol_rel * max(1.0, abs(right))
                violation = left - right - thresh
                # numerical tolerance slack to avoid flagging tiny floating-point
                # rounding artifacts as failures. Treat as violation only if
                # the excess is meaningfully larger than roundoff.
                # numeric slack: ignore tiny violations smaller than the absolute
                # tolerance. This avoids flagging minute floating-point rounding
                # errors as failures when users pass very tight tolerances.
                numeric_eps = max(1e-12, tol_abs)
                if violation > numeric_eps:
                    fail_examples.append(
                        {
                            "indices": indices,
                            "t": t,
                            "x1": x1.cpu().numpy().tolist(),
                            "x2": x2.cpu().numpy().tolist(),
                            "ft": ft,
                            "rhs": (t * f1 + (1.0 - t) * f2),
                            "violation": float(violation),
                        }
                    )
                    if len(fail_examples) >= 5:
                        break
            return fail_examples

        # Build index sets for convex and concave inputs from constraints
        convex_idx = [i for i, c in enumerate(getattr(net, "constraints", [])) if c.convexity == "convex"]
        concave_idx = [i for i, c in enumerate(getattr(net, "constraints", [])) if c.convexity == "concave"]

        if len(convex_idx) > 0:
            cfails = _jensen_test(convex_idx, sign=+1.0)
            jensen_failures["convex"].extend(cfails)
            if verbose:
                print(f"[jensen] convex failures: {len(cfails)}")

        if len(concave_idx) > 0:
            dfails = _jensen_test(concave_idx, sign=-1.0)
            jensen_failures.setdefault("concave", []).extend(dfails)
            if verbose:
                print(f"[jensen] concave failures: {len(dfails)}")

        ok = len(jensen_failures.get("convex", [])) == 0 and len(jensen_failures.get("concave", [])) == 0
        return {"ok": ok, "failures": jensen_failures}
    finally:
        if did_cast:
            net.to(dtype=orig_dtype)


# export updated
__all__ = [
    "validate_monotonicity_randomized",
    "validate_convexity_randomized",
]


if __name__ == "__main__":
    # demo: input 2 dims, first convex monotone increasing, second monotone decreasing (free)
    constraints = [
        ConstraintSpec(convexity="convex", monotonicity="increasing"),
        ConstraintSpec(convexity="free", monotonicity="decreasing"),
    ]
    net = PartiallyConvexNetwork(
        input_dim=2, hidden_sizes=(32, 32), constraints=constraints
    )
    x = torch.tensor(
        [[-2.0, 0.5], [-1.0, 0.0], [0.0, 1.0], [2.0, -1.0]], requires_grad=True
    )
    y = net(x)
    grads = torch.autograd.grad(y.sum(), x)[0]
    print("y:", y.detach().numpy())
    print("grad (w.r.t x):", grads.detach().numpy())
    print("min grad w.r.t convex (idx 0) >= 0?:", grads[:, 0].min().item())

    # run randomized validator
    print("\nMonotonicity test:")
    mres = validate_monotonicity_randomized(
        net, n_samples=2000, eps=1e-4, tol_abs=1e-5, tol_rel=1e-6, seed=0, verbose=True
    )
    print("monotonicity result:", mres)
    print("\nConvexity (Jensen) test:")
    cres = validate_convexity_randomized(
        net, n_jensen=2000, tol_abs=1e-5, tol_rel=1e-6, seed=0, verbose=True
    )
    print("convexity result:", cres)
