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
    """Partially convex scalar network with per-input constraints. Input/output
    relationships can be constrained such that they're convex or free, and monotonic
    increasing, decreasing, or free, in each input dimension.

    The architecture is based on Input Convex Neural Networks (ICNNs)
        from Amos et al., "Input Convex Neural Networks", ICML 2017.

    Monotonicity is additionally enforced by constraining weights to
    be non-negative or non-positive as appropriate.

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
        self.constraints = constraints

        if len(constraints) != self.input_dim:
            raise ValueError("constraints length must equal input_dim")

        # simplified index sets: free inputs and convex inputs
        self.free_idx = tuple(
            i for i, c in enumerate(constraints) if c is None or c.convexity == "free"
        )
        self.convex_idx = tuple(
            i
            for i, c in enumerate(constraints)
            if c is not None and c.convexity == "convex"
        )

        # Require users to use PartiallyConcaveNetwork for concave behavior.
        if [i for i, c in enumerate(constraints) if c and c.convexity == "concave"]:
            raise ValueError(
                "PartiallyConvexNetwork does not accept 'concave...' constraints; "
                "use PartiallyConcaveNetwork for concave behavior"
            )

        # Simplified index sets: free inputs (x) vs convex inputs (y)
        self.free_idx = tuple(
            i for i, c in enumerate(constraints) if c.convexity == "free"
        )
        self.convex_idx = tuple(
            i for i, c in enumerate(constraints) if c and c.convexity == "convex"
        )

        # per-input monotonicity map: idx -> 'increasing'|'decreasing'|None
        self.mono_map = {}
        for i, c in enumerate(constraints):
            if c.monotonicity == "increasing":
                self.mono_map[i] = "increasing"
            elif c.monotonicity == "decreasing":
                self.mono_map[i] = "decreasing"
            else:
                self.mono_map[i] = "free"

        # counts
        self.n_convex = len(self.convex_idx)
        self.n_free = len(self.free_idx)

        # Helper to create per-layer linear maps from a fixed input dimension
        def _make_per_layer(n_in: int) -> nn.ModuleList | None:
            if n_in == 0:
                return None
            return nn.ModuleList(
                [nn.Linear(n_in, h, bias=True) for h in self.hidden_sizes]
            )
        
        def _make_raw_Us():
            raw_Us = nn.ParameterList()
            for i in range(len(self.hidden_sizes)):
                prev = self.hidden_sizes[i - 1] if i > 0 else 0
                if i == 0:
                    raw_Us.append(nn.Parameter(torch.empty(0)))
                else:
                    raw_Us.append(
                        nn.Parameter(torch.randn(self.hidden_sizes[i], prev) * 0.1)
                    )
            return raw_Us

        # Initialize all matrices for weights and biases

        def _make_W_tilda_i():
            raw_W_tildas = nn.ParameterList()
            for i in range(len(self.hidden_sizes)):
                prev = self.hidden_sizes[i - 1] if i > 0 else self.n_free
                raw_W_tildas.append(
                    nn.Parameter(torch.randn(self.hidden_sizes[i], prev) * 0.1)
                )
            return raw_W_tildas
        
        def _make_W_i_z(): 
            W_i_zs = nn.ParameterList()
            for i in range(len(self.hidden_sizes)):
                prev = self.hidden_sizes[i - 1] if i > 0 else self.n_convex
                W_i_zs.append(
                    nn.Parameter(torch.randn(self.hidden_sizes[i], prev) * 0.1)
                )
            return W_i_zs
        
        def _make_W_i_zu(): 
            W_i_zus = nn.ParameterList()
            for i in range(len(self.hidden_sizes)):
                size_u = self.hidden_sizes[i - 1] if i > 0 else self.n_free
                size_z = self.hidden_sizes[i - 1] if i > 0 else self.n_convex
                W_i_zus.append(
                    nn.Parameter(torch.randn(size_z, size_u) * 0.1)  # Could be wrong! 
                )
            return W_i_zus
        
        def _make_W_i_y(): 
            W_i_ys = nn.ParameterList()
            for i in range(len(self.hidden_sizes)):
                W_i_ys.append(
                    nn.Parameter(torch.randn(self.hidden_sizes[i], self.n_convex) * 0.1)
                )
            return W_i_ys
        
        def _make_W_i_yu(): 
            W_i_yus = nn.ParameterList()
            for i in range(len(self.hidden_sizes)):
                size_u = self.hidden_sizes[i - 1] if i > 0 else self.n_free
                W_i_yus.append(
                    nn.Parameter(torch.randn(self.n_convex, size_u) * 0.1)  # Could be wrong!
                )
            return W_i_yus
        
        def _make_W_i_u(): 
            W_i_us = nn.ParameterList()
            for i in range(len(self.hidden_sizes)):
                size_u = self.hidden_sizes[i - 1] if i > 0 else self.n_free
                W_i_us.append(
                    nn.Parameter(torch.randn(self.hidden_sizes[i], size_u) * 0.1)
                )
            return W_i_us
        
        def _make_b_tilda_i(): 
            b_tilda_i = nn.ParameterList()
            for i in range(len(self.hidden_sizes)):
                b_tilda_i.append(
                    nn.Parameter(torch.randn(self.hidden_sizes[i]) * 0.1)
                )
            return b_tilda_i
        
        def _make_b_i_z(): 
            b_i_z = nn.ParameterList()
            for i in range(len(self.hidden_sizes)):
                b_i_z.append(
                    nn.Parameter(torch.randn(self.hidden_sizes[i]) * 0.1)
                )
            return b_i_z
        
        def _make_b_i_y(): 
            b_i_y = nn.ParameterList()
            for i in range(len(self.hidden_sizes)):
                b_i_y.append(
                    nn.Parameter(torch.randn(self.n_convex) * 0.1)
                )
            return b_i_y
        
        def _make_b_i(): 
            b_i = nn.ParameterList()
            for i in range(len(self.hidden_sizes)):
                b_i.append(
                    nn.Parameter(torch.randn(self.n_convex) * 0.1)
                )
            return b_i
        
        self.W_tilda_i = _make_W_tilda_i() if self.n_free > 0 else None
        self.b_tilda_i = _make_b_tilda_i() if self.n_free > 0 else None
        self.W_i_z = _make_W_i_z()
        self.W_i_zu = _make_W_i_zu()
        self.W_i_y = _make_W_i_y()
        self.W_i_yu = _make_W_i_yu()
        self.W_i_u = _make_W_i_u()
        self.b_i_z = _make_b_i_z()
        self.b_i_y = _make_b_i_y()
        self.b_i = _make_b_i()

        # Weight and bias for read-out
        self.W_k = nn.Parameter(torch.randn(1, self.hidden_sizes[-1]) * 0.1)
        self.b_k = nn.Parameter(torch.zeros(1))

    # apply per-input monotone constraints to columns of a weight matrix (out, in)
    def _apply_col_monotone(
        self, raw_weight: torch.Tensor, mono_list: list[str]
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
            # if free, don't change weights
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
        u = xf  # initial u_0 = xf
        for i in range(len(self.hidden_sizes)):
            # u contribution from full free input vector xf
            if self.W_tilda_i is not None:
                mono_list_f = [i for i, c in enumerate(self.constraints) if c.convexity == "free"]
                W_u = self._apply_col_monotone(
                    self.linear_i_free[i].weight, mono_list_f
                )
                u_ip1 = self.activation(F.linear(xf, W_u, self.linear_i_free[i].bias))
            else:
                u_ip1 = 0.0

            # compute z_{i+1}
            parts = []

            # contribution from convex inputs y
            if self.Wc_conv is not None:
                Wc = self.Wc_conv[i]
                Wc_w = self._apply_col_monotone(
                    Wc.weight, [self.mono_map[idx] for idx in self.convex_idx]
                )
                parts.append(F.linear(xc, Wc_w, Wc.bias))

            # contribution from free inputs (direct path)
            if self.Wu_free is not None:
                Wu_w = self._apply_col_monotone(
                    self.Wu_free[i].weight,
                    [self.mono_map[idx] for idx in self.free_idx],
                )
                parts.append(F.linear(xf, Wu_w, self.Wu_free[i].bias))

            # contribution from previous z via U (enforce non-negativity via softplus)
            if i > 0:
                Upos = self.softplus(self.raw_Us[i])
                parts.append(z @ Upos.T)

            # include u contribution
            parts.append(u_ip1 if isinstance(u_ip1, torch.Tensor) else 0.0)

            agg = sum(parts) if parts else 0.0
            z = self.activation(agg)

        # final readout from z path (enforce non-negative readout weights for convexity)
        if z is not None:
            rw = self._apply_col_monotone(
                self.readout_layer.weight, ["increasing"] * self.hidden_sizes[-1]
            )
            out_c = F.linear(z, rw, None).view(-1)
        else:
            out_c = torch.zeros(batch, device=x.device)

        # combine and add linear per-input terms with per-index monotonicity
        readout = out_c

        if self.n_convex > 0:
            mono_list_c = [self.mono_map[idx] for idx in self.convex_idx]
            W_xc = self.lin_xc.weight.clone()
            W_xc = self._apply_col_monotone(W_xc, mono_list_c)
            readout = readout + F.linear(xc, W_xc, None).view(-1)

        if self.n_free > 0:
            mono_list_f = [self.mono_map[idx] for idx in self.free_idx]
            W_xf = self.lin_xf.weight.clone()
            W_xf = self._apply_col_monotone(W_xf, mono_list_f)
            readout = readout + F.linear(xf, W_xf, None).view(-1)
        # add scalar bias and return
        readout = readout + self.b_out.view(1)
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


# export updated
__all__ = [
    "PartiallyConvexNetwork",
    "PartiallyConcaveNetwork",
]

