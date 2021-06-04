import torch
from torch import Tensor
import numpy as np


# Original code from https://github.com/lyft/l5kit/blob/20ab033c01610d711c3d36e1963ecec86e8b85b6/l5kit/l5kit/evaluation/metrics.py
def nll_batch(gt: Tensor, pred: Tensor, confidences: Tensor, avails: Tensor) -> Tensor:
    """
    Compute a negative log-likelihood for the multi-modal scenario.
    log-sum-exp trick is used here to avoid underflow and overflow, For more information about it see:
    https://en.wikipedia.org/wiki/LogSumExp#log-sum-exp_trick_for_log-domain_calculations
    https://timvieira.github.io/blog/post/2014/02/11/exp-normalize-trick/
    https://leimao.github.io/blog/LogSumExp/
    Args:
        gt (Tensor): array of shape (bs)x(time)x(2D coords)
        pred (Tensor): array of shape (bs)x(modes)x(time)x(2D coords)
        confidences (Tensor): array of shape (bs)x(modes) with a confidence for each mode in each sample
        avails (Tensor): array of shape (bs)x(time) with the availability for each gt timestep
    Returns:
        Tensor: negative log-likelihood for this example, a single float number
    """
    assert len(pred.shape) == 4, f"expected 3D (MxTxC) array for pred, got {pred.shape}"
    
    batch_size, num_modes, future_len, num_coords = pred.shape

    assert gt.shape == (batch_size, future_len,
                        num_coords), f"expected 2D (Time x Coords) array for gt, got {gt.shape}"
    assert confidences.shape == (
        batch_size, num_modes), f"expected 1D (Modes) array for gt, got {confidences.shape}"

    # assert torch.allclose(torch.sum(confidences, dim=1) ,
    #                         confidences.new_ones((batch_size,)) ), "confidences should sum to 1"
    if not (torch.allclose(torch.sum(confidences, dim=1), confidences.new_ones((batch_size,)))):
        print('confidences:', confidences)
        print('torch.sum:', torch.sum(confidences, dim=1))
        # print('confidences:', confidences.new_ones((batch_size,)))
        raise "confidences should sum to 1"

    assert avails.shape == (batch_size, future_len), f"expected 1D (Time) array for gt, got {avails.shape}"
    # assert all data are valid
    assert torch.isfinite(pred).all(), "invalid value found in pred"
    assert torch.isfinite(gt).all(), "invalid value found in gt"
    assert torch.isfinite(confidences).all(), "invalid value found in confidences"
    assert torch.isfinite(avails).all(), "invalid value found in avails"

    # convert to (batch_size, num_modes, future_len, num_coords)
    gt = torch.unsqueeze(gt, 1)  # add modes
    avails = avails[:, None, :, None]  # add modes and cords

    # error (batch_size, num_modes, future_len)
    # reduce coords and use availability
    error = torch.sum(((gt - pred) * avails) ** 2, dim=-1)

    # when confidence is 0 log goes to -inf, but we're fine with it
    with np.errstate(divide="ignore"):
        # error (batch_size, num_modes)
        error = torch.log(confidences) - 0.5 * \
            torch.sum(error, dim=-1)  # reduce time

    # error are negative at this point, so max() gives the minimum one
    max_value, _ = error.max(dim=1, keepdim=True)

    error = -torch.log(torch.sum(torch.exp(error - max_value),
                                 dim=-1, keepdim=True)) - max_value  # reduce modes

    return torch.mean(error)