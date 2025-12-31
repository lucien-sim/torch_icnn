try:
    import torch
except Exception:
    torch = None


def test_opp_monotonicity():
    from torch_icnn.networks import opp_monotonicity

    assert opp_monotonicity("increasing") == "decreasing"
    assert opp_monotonicity("decreasing") == "increasing"
    assert opp_monotonicity("free") == "free"


def test_make_jensen_pair_differs_only_on_indices():
    if torch is None:
        return
    from torch_icnn.validation import _make_jensen_pair
    D = 5
    indices = [1, 3]
    x1, x2, t = _make_jensen_pair(D, indices, device=torch.device("cpu"), dtype=torch.float32)
    x1 = x1.cpu().numpy()[0]
    x2 = x2.cpu().numpy()[0]
    # check only indices differ
    for i in range(D):
        if i in indices:
            assert x1[i] != x2[i]
        else:
            assert abs(x1[i] - x2[i]) < 1e-6
