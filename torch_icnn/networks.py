# TODO: understand the interaction module. 
# TODO: get pytest working.
# TODO: add some partial dependence visualization utilities.
# TODO: test this properly
# TODO: add compatibility with periodic inputs through an encoding
# (the variables created will be "free").


from typing import Sequence, Literal
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass


@dataclass
class ConstraintSpec:
    convexity: Literal["convex", "concave", "free"] = "free"
    monotonicity: Literal["increasing", "decreasing", "free"] = "free"


def opp_monotonicity(m: str) -> Literal["increasing", "decreasing", "free"]:
    """Return the opposite monotonicity: 'increasing' <-> 'decreasing', 'free' -> 'free'."""
    if m == "increasing":
        return "decreasing"
    if m == "decreasing":
        return "increasing"
    return "free"


def apply_col_monotone(
    raw_weight: torch.Tensor, mono_list: list | None, softplus: nn.Softplus
) -> torch.Tensor:
    """Apply per-column monotonicity constraints to a weight matrix.

    This mirrors the previous behavior but is a standalone helper so that
    multiple modules can reuse it.
    """
    if mono_list is None:
        return raw_weight
    _, n_in = raw_weight.shape
    if n_in == 0:
        return raw_weight
    raw_pos = softplus(raw_weight)
    W = raw_weight.clone()
    for j, m in enumerate(mono_list):
        if j >= n_in:
            break
        if m == "increasing":
            W[:, j] = raw_pos[:, j]
        elif m == "decreasing":
            W[:, j] = -raw_pos[:, j]
        # else keep raw
    return W


class ICNNBlock(nn.Module):
    """ICNN style layer stack that computes the hidden activations z.

    This encapsulates per-layer parameters (Wc, Wu, Wu_tilde), U recurrence
    matrices (raw_U), activations, and per-column monotonicity application.
    """

    def __init__(
        self,
        n_convex: int,
        n_free: int,
        hidden_sizes: Sequence[int],
        mono_list_c: list | None = None,
        mono_list_f: list | None = None,
        activation: nn.Module = nn.ReLU,
    ) -> None:
        super().__init__()
        self.n_convex = int(n_convex)
        self.n_free = int(n_free)
        self.hidden_sizes = list(hidden_sizes)
        assert len(self.hidden_sizes) > 0
        self.activation = activation()
        self.softplus = nn.Softplus()
        self.mono_list_c = None if mono_list_c is None else list(mono_list_c)
        self.mono_list_f = None if mono_list_f is None else list(mono_list_f)

        # helper to create per-layer param lists
        def _make_params(n_in: int):
            if n_in == 0:
                return None, None
            w_list = nn.ParameterList()
            b_list = nn.ParameterList()
            for h in self.hidden_sizes:
                w_list.append(nn.Parameter(torch.randn(h, n_in) * 0.1))
                b_list.append(nn.Parameter(torch.zeros(h)))
            return w_list, b_list

        # U matrices
        raw_U = nn.ParameterList()
        for i in range(len(self.hidden_sizes)):
            prev = self.hidden_sizes[i - 1] if i > 0 else 0
            if i == 0:
                raw_U.append(nn.Parameter(torch.empty(0)))
            else:
                raw_U.append(
                    nn.Parameter(torch.randn(self.hidden_sizes[i], prev) * 0.1)
                )
        self.raw_U = raw_U

        # per-layer input mappings
        self.Wc_w, self.Wc_b = _make_params(self.n_convex)
        self.Wu_w, self.Wu_b = _make_params(self.n_free)
        self.Wu_tilde_w, self.Wu_tilde_b = _make_params(self.n_free)

    def forward(self, xc: torch.Tensor, xf: torch.Tensor) -> torch.Tensor:
        z = None
        for i in range(len(self.hidden_sizes)):
            # u contribution from full free input vector xf
            if self.Wu_tilde_w is not None:
                W_u_raw = self.Wu_tilde_w[i]
                b_u = self.Wu_tilde_b[i]
                W_u = apply_col_monotone(W_u_raw, self.mono_list_f, self.softplus)
                u_ip1 = self.activation(F.linear(xf, W_u, b_u))
            else:
                u_ip1 = 0.0

            parts = []
            if self.Wc_w is not None:
                Wc_raw = self.Wc_w[i]
                b_c = self.Wc_b[i]
                Wc_w = apply_col_monotone(Wc_raw, self.mono_list_c, self.softplus)
                parts.append(F.linear(xc, Wc_w, b_c))

            if self.Wu_w is not None:
                Wu_raw = self.Wu_w[i]
                b_u = self.Wu_b[i]
                Wu_w = apply_col_monotone(Wu_raw, self.mono_list_f, self.softplus)
                parts.append(F.linear(xf, Wu_w, b_u))

            if i > 0:
                Upos = self.softplus(self.raw_U[i])
                parts.append(z @ Upos.T)

            parts.append(u_ip1 if isinstance(u_ip1, torch.Tensor) else 0.0)

            agg = sum(parts) if parts else 0.0
            z = self.activation(agg)
        return z


class ReadoutModule(nn.Module):
    """Final readout: w^T z + lin_xc(y) + lin_xf(f) + b with monotonic constraints."""

    def __init__(
        self,
        z_dim: int,
        n_convex: int,
        n_free: int,
        mono_list_c: list | None = None,
        mono_list_f: list | None = None,
    ) -> None:
        super().__init__()
        self.z_dim = int(z_dim)
        self.n_convex = int(n_convex)
        self.n_free = int(n_free)
        self.softplus = nn.Softplus()
        self.mono_list_c = None if mono_list_c is None else list(mono_list_c)
        self.mono_list_f = None if mono_list_f is None else list(mono_list_f)

        self.w_readout = nn.Parameter(torch.randn(1, self.z_dim) * 0.1)
        self.lin_xc_w = (
            nn.Parameter(torch.randn(1, self.n_convex) * 0.1)
            if self.n_convex > 0
            else None
        )
        self.lin_xf_w = (
            nn.Parameter(torch.randn(1, self.n_free) * 0.1) if self.n_free > 0 else None
        )
        self.b = nn.Parameter(torch.zeros(1))

    def forward(
        self, z: torch.Tensor, xc: torch.Tensor, xf: torch.Tensor
    ) -> torch.Tensor:
        if z is not None:
            rw = apply_col_monotone(
                self.w_readout, ["increasing"] * self.z_dim, self.softplus
            )
            out_c = F.linear(z, rw, None).view(-1)
        else:
            out_c = torch.zeros(
                z.shape[0] if z is not None else xc.shape[0], device=xc.device
            )

        readout = out_c
        if self.n_convex > 0:
            W_xc = apply_col_monotone(self.lin_xc_w, self.mono_list_c, self.softplus)
            readout = readout + F.linear(xc, W_xc, None).view(-1)
        if self.n_free > 0:
            W_xf = apply_col_monotone(self.lin_xf_w, self.mono_list_f, self.softplus)
            readout = readout + F.linear(xf, W_xf, None).view(-1)
        readout = readout + self.b.view(1)
        return readout.view(-1)


class ConvexSubnetwork(nn.Module):
    """Thin wrapper that combines an ICNNBlock and ReadoutModule for convex modeling."""

    def __init__(
        self,
        n_convex: int,
        n_free: int,
        hidden_sizes=(64, 64),
        activation: nn.Module = nn.ReLU,
        mono_list_c: list | None = None,
        mono_list_f: list | None = None,
    ) -> None:
        super().__init__()
        self.block = ICNNBlock(
            n_convex, n_free, hidden_sizes, mono_list_c, mono_list_f, activation
        )
        self.readout = ReadoutModule(
            hidden_sizes[-1], n_convex, n_free, mono_list_c, mono_list_f
        )

    def forward(self, xc: torch.Tensor, xf: torch.Tensor) -> torch.Tensor:
        z = self.block(xc, xf)
        return self.readout(z, xc, xf)


class ConcaveSubnetwork(nn.Module):
    """Wrapper that produces a concave contribution by negating a ConvexSubnetwork."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__()
        self._conv = ConvexSubnetwork(*args, **kwargs)

    def forward(self, xc: torch.Tensor, xf: torch.Tensor) -> torch.Tensor:
        return -self._conv(xc, xf)





class PartiallyConvexNetwork(nn.Module):
    """Partially convex scalar network with per-input constraints.

    Inputs can be declared *convex* (y) or *free* (f) per-dimension. The
    architecture follows an ICNN-style construction with non-negative
    recurrence and non-negative readout to guarantee convexity in the
    declared convex inputs. Per-input monotonicity (increasing/decreasing)
    is enforced by constraining columns of input-weight matrices via
    a Softplus sign parametrization. A lot of the principles come from
    the ICNN paper: "Input Convex Neural Networks" (Amos et al., ICML 2017).

    Compact architecture (per-layer i):
        xc: convex inputs y
        xf: free inputs f

        u^{(i+1)}(f) = W_tilde_u_i f + b_u_i     # (computed from free inputs only)
        z^{(0)} = 0, u^{(0)} = 0

        z^{(i+1)} = σ( Wc_i y + Wu_i f + z^{(i)} U_i^T + u^{(i+1)}(f) + b_c_i )

    Final scalar output:
        f(x) = w^T z^{(L)} + lin_xc(y) + lin_xf(f) + b

    Convexity guarantee (sketch)
    ---------------------------
    - Activation σ is assumed convex and non-decreasing (e.g., ReLU).
    - U_i are enforced elementwise non-negative via Softplus.
    - Wc_i (weights from convex inputs) can have any sign (convexity does not require sign constraints); convexity is instead enforced by non-negative recurrence U_i and non-negative readout weights w.
    By induction: z^{(0)} = 0 is convex in y. If z^{(i)} is convex in y,
    then a^{(i+1)}(y) = Wc_i y + z^{(i)} U_i^T + ... is a sum of convex
    or linear terms (all non-negative combinations), so a^{(i+1)} is
    convex; σ(·) preserves convexity when σ is convex and non-decreasing.
    Thus z^{(L)} is convex in y and so is f(x) (sum of convex and linear).

    Monotonicity guarantee (sketch)
    ------------------------------
    - For a coordinate declared "increasing", all columns that consume
      that input (Wc, Wu, linear_i_free, lin_xc, lin_xf, and the readout
      column) are constrained to be non-negative via Softplus.
    - Since σ is non-decreasing, each path derivative ∂f/∂x_j is a sum of
      products of non-negative factors → ∂f/∂x_j ≥ 0 (and ≤ 0 for
      "decreasing" using negative Softplus).

    Notes
    -----
    - Convexity holds only with respect to inputs labelled `convex`.
    - Monotonicity constraints are applied at forward time using Softplus
      parametrization (smooth, differentiable).

    Parameters
    ----------
    input_dim: int
        Total input dimensionality D.
    hidden_sizes: tuple[int,...]
        Hidden layer widths (ICNN-style).
    activation: nn.Module
        Activation constructor (default: nn.ReLU) — should be convex & non-decreasing.
    constraints: Sequence[ConstraintSpec] | None
        Per-input constraint specs. If None, no constraints.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_sizes=(64, 64),
        activation: nn.Module = nn.ReLU,
        constraints: Sequence[ConstraintSpec] | None = None,
    ) -> None:
        super().__init__()
        self.input_dim = int(input_dim)
        self.hidden_sizes = list(hidden_sizes)
        assert len(self.hidden_sizes) > 0, "hidden_sizes must be non-empty"
        self.activation = activation()
        self.softplus = nn.Softplus()

        # parse constraints
        if constraints is None:
            # If no constraints, everything is free/free
            constraints = [
                ConstraintSpec(convexity="free", monotonicity="free")
                for _ in range(self.input_dim)
            ]
        # Any None constraints are free/free
        constraints = [
            ConstraintSpec(convexity="free", monotonicity="free") if c is None else c
            for c in constraints
        ]

        if len(constraints) != self.input_dim:
            raise ValueError("constraints length must equal input_dim")

        # Require users to use PartiallyConcaveNetwork for concave behavior.
        if [i for i, c in enumerate(constraints) if c.convexity == "concave"]:
            raise ValueError(
                "PartiallyConvexNetwork does not accept 'concave...' constraints; "
                "use PartiallyConcaveNetwork for concave behavior"
            )

        self.constraints = constraints

        # simplified index sets: free inputs and convex inputs
        self.free_idx = tuple(
            i for i, c in enumerate(constraints) if c.convexity == "free"
        )
        self.convex_idx = tuple(
            i for i, c in enumerate(constraints) if c.convexity == "convex"
        )

        # counts
        self.n_convex = len(self.convex_idx)
        self.n_free = len(self.free_idx)

        # Determine which inputs in xc and xf are monotonic (per-column lists)
        self.mono_list_c = [constraints[idx].monotonicity for idx in self.convex_idx]
        self.mono_list_f = [constraints[idx].monotonicity for idx in self.free_idx]

        # Build a modular ConvexSubnetwork (ICNN block + readout)
        self._conv = ConvexSubnetwork(
            n_convex=self.n_convex,
            n_free=self.n_free,
            hidden_sizes=self.hidden_sizes,
            activation=activation,
            mono_list_c=self.mono_list_c,
            mono_list_f=self.mono_list_f,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.view(-1, self.input_dim).float()
        batch = x.shape[0]

        # split inputs into convex (y) and free (x) groups
        xc = (
            x[:, list(self.convex_idx)]
            if self.n_convex > 0
            else torch.empty(batch, 0, dtype=x.dtype, device=x.device)
        )
        xf = (
            x[:, list(self.free_idx)]
            if self.n_free > 0
            else torch.empty(batch, 0, dtype=x.dtype, device=x.device)
        )

        return self._conv(xc, xf)


class PartiallyConcaveNetwork(nn.Module):
    """Concave network implemented directly from modular components.

    This builds a convex subnetwork on selected inputs and negates its output to
    produce concave behavior. Mapping of constraints follows the previous
    semantics: inputs declared `concave` are mapped to convex inputs for the
    internal subnetwork, and monotonicities are flipped appropriately so that
    negating the internal convex subnetwork yields the requested concave
    monotonicities.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_sizes=(64, 64),
        activation: nn.Module = nn.ReLU,
        constraints: Sequence[ConstraintSpec] | None = None,
    ) -> None:
        super().__init__()
        self.input_dim = int(input_dim)

        # normalize constraints
        if constraints is None:
            constraints = [ConstraintSpec() for _ in range(self.input_dim)]
        constraints = [ConstraintSpec() if c is None else c for c in constraints]
        if len(constraints) != self.input_dim:
            raise ValueError("constraints length must equal input_dim")
        self.constraints = constraints

        # Disallow explicit 'convex' constraints in the concave wrapper (use other classes)
        if any(c.convexity == "convex" for c in self.constraints):
            raise ValueError(
                "PartiallyConcaveNetwork does not accept 'convex' constraints; "
                "use PartiallyConvexNetwork or PartiallyMixedNetwork instead"
            )

        # Determine which indices map to internal convex inputs and frees
        # Concave inputs become internal convex inputs with flipped monotonicity
        self.convex_idx_internal = tuple(i for i, c in enumerate(self.constraints) if c.convexity == "concave")
        self.free_idx_internal = tuple(i for i, c in enumerate(self.constraints) if c.convexity == "free")

        mono_list_c = [opp_monotonicity(self.constraints[i].monotonicity) for i in self.convex_idx_internal]
        mono_list_f = [opp_monotonicity(self.constraints[i].monotonicity) for i in self.free_idx_internal]

        # build internal convex subnetwork that we will negate
        self._conv = ConvexSubnetwork(
            n_convex=len(self.convex_idx_internal),
            n_free=len(self.free_idx_internal),
            hidden_sizes=hidden_sizes,
            activation=activation,
            mono_list_c=mono_list_c,
            mono_list_f=mono_list_f,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.view(-1, self.input_dim).float()
        batch = x.shape[0]
        xc = x[:, list(self.convex_idx_internal)] if len(self.convex_idx_internal) > 0 else torch.empty(batch, 0, dtype=x.dtype, device=x.device)
        xf = x[:, list(self.free_idx_internal)] if len(self.free_idx_internal) > 0 else torch.empty(batch, 0, dtype=x.dtype, device=x.device)
        return -self._conv(xc, xf)


class PartiallyMixedNetwork(nn.Module):
    """Partially mixed convex/concave network.

    Implements f(x) = g(x_convex, x_free) - h(x_concave, x_free) where g and h
    are `ConvexSubnetwork` instances. This enforces that nonlinear dependence on
    convex inputs only appears in g and nonlinear dependence on concave inputs
    only appears in h. Monotonicity constraints are mapped so that the combined
    partial derivative has the requested sign:

        - For a requested monotonicity 'increasing' on input j: g uses
          'increasing' and h uses 'decreasing' so that ∂f/∂x_j = g' - h' ≥ 0.

    Interactions
    ------------
    Nonlinear interactions between convex and concave variables occur indirectly
    through shared free inputs (`xf`). Explicit interaction modules were removed
    to keep curvature guarantees simple.

    Related tests
    -------------
    - Basic mixed behavior: `tests/test_networks.py::test_partially_mixed_small`
    - Edge cases: `tests/test_edge_cases.py` (all-free, all-concave, no-free, single-dim)

    Parameters
    ----------
    input_dim: int
        Total input dimensionality D.
    hidden_sizes, activation: passed through to subnets.
    constraints: Sequence[ConstraintSpec]
        Per-input desired constraints. Items may be 'convex', 'concave', or 'free'.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_sizes=(64, 64),
        activation: nn.Module = nn.ReLU,
        constraints: Sequence[ConstraintSpec] | None = None,
    ) -> None:
        super().__init__()
        self.input_dim = int(input_dim)
        if constraints is None:
            constraints = [ConstraintSpec() for _ in range(self.input_dim)]
        if len(constraints) != self.input_dim:
            raise ValueError("constraints length must equal input_dim")
        self.constraints = [ConstraintSpec() if c is None else c for c in constraints]

        # index sets
        self.convex_idx = tuple(
            i for i, c in enumerate(self.constraints) if c.convexity == "convex"
        )
        self.concave_idx = tuple(
            i for i, c in enumerate(self.constraints) if c.convexity == "concave"
        )
        self.free_idx = tuple(
            i for i, c in enumerate(self.constraints) if c.convexity == "free"
        )

        # build mono lists for g and h
        mono_list_g_c = [self.constraints[i].monotonicity for i in self.convex_idx]
        mono_list_g_f = [self.constraints[i].monotonicity for i in self.free_idx]

        mono_list_h_c = [
            opp_monotonicity(self.constraints[i].monotonicity) for i in self.concave_idx
        ]
        mono_list_h_f = [opp_monotonicity(self.constraints[i].monotonicity) for i in self.free_idx]

        # build ConvexSubnetwork instances for g and h (h will be negated at combination)
        self._g = ConvexSubnetwork(
            n_convex=len(self.convex_idx),
            n_free=len(self.free_idx),
            hidden_sizes=hidden_sizes,
            activation=activation,
            mono_list_c=mono_list_g_c,
            mono_list_f=mono_list_g_f,
        )
        self._h = ConvexSubnetwork(
            n_convex=len(self.concave_idx),
            n_free=len(self.free_idx),
            hidden_sizes=hidden_sizes,
            activation=activation,
            mono_list_c=mono_list_h_c,
            mono_list_f=mono_list_h_f,
        )



    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.view(-1, self.input_dim).float()
        batch = x.shape[0]
        # slice components
        xc = (
            x[:, list(self.convex_idx)]
            if len(self.convex_idx) > 0
            else torch.empty(batch, 0, dtype=x.dtype, device=x.device)
        )
        xn = (
            x[:, list(self.concave_idx)]
            if len(self.concave_idx) > 0
            else torch.empty(batch, 0, dtype=x.dtype, device=x.device)
        )
        xf = (
            x[:, list(self.free_idx)]
            if len(self.free_idx) > 0
            else torch.empty(batch, 0, dtype=x.dtype, device=x.device)
        )

        xg = (
            torch.cat([xc, xf], dim=1)
            if (xc.numel() or xf.numel())
            else torch.empty(batch, 0, dtype=x.dtype, device=x.device)
        )
        xh = (
            torch.cat([xn, xf], dim=1)
            if (xn.numel() or xf.numel())
            else torch.empty(batch, 0, dtype=x.dtype, device=x.device)
        )

        out_g = self._g(xg)
        out_h = self._h(xh)

        return out_g - out_h


def viz() -> None:
    from torchviz import make_dot, make_dot_from_trace

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
    make_dot(
        y.mean(), params=dict(net.named_parameters()), show_attrs=True, show_saved=True
    )


# export updated
__all__ = [
    "PartiallyConvexNetwork",
    "PartiallyConcaveNetwork",
    "PartiallyMixedNetwork",
    # modular components
    "ICNNBlock",
    "ReadoutModule",
    "ConvexSubnetwork",
    "ConcaveSubnetwork",
]
