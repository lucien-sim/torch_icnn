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

        # Helper to create per-layer weight and bias ParameterLists for a fixed input dimension
        def _make_linear_params(n_in: int):
            if n_in == 0:
                return None, None
            w_list = nn.ParameterList()
            b_list = nn.ParameterList()
            for h in self.hidden_sizes:
                w_list.append(nn.Parameter(torch.randn(h, n_in) * 0.1))
                b_list.append(nn.Parameter(torch.zeros(h)))
            return w_list, b_list

        # U matrices (z -> z) between layers (enforced non-negative via softplus in forward)
        def _make_raw_Us():
            raw_Us = nn.ParameterList()
            for i in range(len(self.hidden_sizes)):
                prev = self.hidden_sizes[i - 1] if i > 0 else 0
                if i == 0:
                    raw_Us.append(nn.Parameter(torch.empty(0)))
                else:
                    raw_Us.append(nn.Parameter(torch.randn(self.hidden_sizes[i], prev) * 0.1))
            return raw_Us

        # U_i parameters (z -> z recurrence), raw stored and made positive in forward as Upos = softplus(raw_U[i])
        self.raw_U = _make_raw_Us()

        # per-layer mappings: convex inputs -> hidden (Wc_i) and free inputs -> hidden (Wu_i)
        self.Wc_w, self.Wc_b = _make_linear_params(self.n_convex)
        self.Wu_w, self.Wu_b = _make_linear_params(self.n_free)

        # per-layer u maps (compute u from the full free input vector): Wu_tilde_i f + b_u_i
        self.Wu_tilde_w, self.Wu_tilde_b = _make_linear_params(self.n_free)

        # readout and per-input linear terms (explicit parameters)
        self.w_readout = nn.Parameter(torch.randn(1, self.hidden_sizes[-1]) * 0.1)
        self.lin_xc_w = nn.Parameter(torch.randn(1, self.n_convex) * 0.1) if self.n_convex > 0 else None
        self.lin_xf_w = nn.Parameter(torch.randn(1, self.n_free) * 0.1) if self.n_free > 0 else None

        # scalar bias term
        self.b = nn.Parameter(torch.zeros(1))

    # apply per-input monotone constraints to columns of a weight matrix (out, in)
    def _apply_col_monotone(
        self, raw_weight: torch.Tensor, mono_list: list | None
    ) -> torch.Tensor:
        """Apply per-column monotonicity constraints to a weight matrix.
        Achieved by applying softplus to enforce positivity for 'increasing' columns,
        negative softplus for 'decreasing' columns, and leaving 'free' columns unchanged.

        Parameters
        ----------
        raw_weight : torch.Tensor
            Weight matrix of shape (out, in)
        mono_list : list | None
            Per-input monotonicity list with values 'increasing'|'decreasing'|None

        Returns
        -------
        torch.Tensor
            New weight matrix with monotone columns enforced.
        """
        if mono_list is None:
            return raw_weight
        _, n_in = raw_weight.shape
        if n_in == 0:
            return raw_weight
        raw_pos = self.softplus(raw_weight)
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

        # compute per-layer z iteratively
        z = None
        for i in range(len(self.hidden_sizes)):
            # u contribution from full free input vector xf (Wu_tilde_i)
            if self.Wu_tilde_w is not None:
                W_u_raw = self.Wu_tilde_w[i]
                b_u = self.Wu_tilde_b[i]
                W_u = self._apply_col_monotone(W_u_raw, self.mono_list_f)
                u_ip1 = self.activation(F.linear(xf, W_u, b_u))
            else:
                u_ip1 = 0.0

            # compute z_{i+1}
            parts = []

            # contribution from convex inputs y (Wc_i)
            if self.Wc_w is not None:
                Wc_raw = self.Wc_w[i]
                b_c = self.Wc_b[i]
                Wc_w = self._apply_col_monotone(Wc_raw, self.mono_list_c)
                parts.append(F.linear(xc, Wc_w, b_c))

            # contribution from free inputs (direct path Wu_i)
            if self.Wu_w is not None:
                Wu_raw = self.Wu_w[i]
                b_u = self.Wu_b[i]
                Wu_w = self._apply_col_monotone(Wu_raw, self.mono_list_f)
                parts.append(F.linear(xf, Wu_w, b_u))

            # contribution from previous z via U (enforce non-negativity via softplus)
            if i > 0:
                Upos = self.softplus(self.raw_U[i])
                parts.append(z @ Upos.T)

            # include u contribution
            parts.append(u_ip1 if isinstance(u_ip1, torch.Tensor) else 0.0)

            agg = sum(parts) if parts else 0.0
            z = self.activation(agg)

        # final readout from z path (enforce non-negative readout weights for convexity)
        if z is not None:
            rw = self._apply_col_monotone(self.w_readout, ["increasing"] * self.hidden_sizes[-1])
            out_c = F.linear(z, rw, None).view(-1)
        else:
            out_c = torch.zeros(batch, device=x.device)

        # combine and add linear per-input terms with per-index monotonicity
        readout = out_c

        if self.n_convex > 0:
            W_xc = self._apply_col_monotone(self.lin_xc_w, self.mono_list_c)
            readout = readout + F.linear(xc, W_xc, None).view(-1)

        if self.n_free > 0:
            W_xf = self._apply_col_monotone(self.lin_xf_w, self.mono_list_f)
            readout = readout + F.linear(xf, W_xf, None).view(-1)
        # add scalar bias and return
        readout = readout + self.b.view(1)
        return readout.view(-1)


class PartiallyConcaveNetwork(nn.Module):
    """Wrapper implementing partially concave networks by negating a convex base.

    Accepts the same constraint literals as `PartiallyConvexNetwork`, including
    'concave...' variants. Internally these are mapped to equivalent convex
    constraints for the base and monotonicities are flipped so that negating the
    base yields the intended concave + monotonic behaviour.
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

        # map original (concave...) constraints to convex ones suitable for base
        if constraints is None:
            base_constraints = None
        else:
            base_constraints = []
            for c in constraints:
                # support both ConstraintSpec dataclass and legacy string literals
                if c is None:
                    base_constraints.append(
                        ConstraintSpec(convexity="free", monotonicity="free")
                    )
                elif c.convexity == "concave":
                    # flip monotonicity when converting to convex base
                    if c.monotonicity == "increasing":
                        base_constraints.append(
                            ConstraintSpec(
                                convexity="convex", monotonicity="decreasing"
                            )
                        )
                    elif c.monotonicity == "decreasing":
                        base_constraints.append(
                            ConstraintSpec(
                                convexity="convex", monotonicity="increasing"
                            )
                        )
                    else:
                        base_constraints.append(ConstraintSpec(convexity="convex"))
                elif (
                    c.monotonicity in ("increasing", "decreasing")
                    and c.convexity == "free"
                ):
                    # flip direction when negating the base
                    base_constraints.append(
                        ConstraintSpec(
                            convexity="free",
                            monotonicity=(
                                "decreasing"
                                if c.monotonicity == "increasing"
                                else "increasing"
                            ),
                        )
                    )
                else:
                    base_constraints.append(c)

        self._base = PartiallyConvexNetwork(
            input_dim=input_dim,
            hidden_sizes=hidden_sizes,
            activation=activation,
            constraints=base_constraints,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return -self._base(x)


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
]
