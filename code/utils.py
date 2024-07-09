import torch
import torch.nn.functional as F

import numpy as np

def do_mixup(x, mixup_lambda):
    """Mixup x of even indexes (0, 2, 4, ...) with x of odd indexes 
    (1, 3, 5, ...).

    Args:
      x: (batch_size * 2, ...)
      mixup_lambda: (batch_size * 2,)

    Returns:
      out: (batch_size, ...)
    """
    out = (x[0 :: 2].transpose(0, -1) * mixup_lambda[0 :: 2] + \
        x[1 :: 2].transpose(0, -1) * mixup_lambda[1 :: 2]).transpose(0, -1)
    return out

def print_args(args):
    log_str = ("==========================================\n")
    log_str += ("==========       config      =============\n")
    log_str += ("==========================================\n")
    for arg, content in args.__dict__.items():
        log_str += ("{}:{}\n".format(arg, content))
    log_str += ("\n==========================================\n")
    print(log_str)
    args.out_file.write(log_str+'\n')
    args.out_file.flush()

def linear_rampup(current, rampup_length):
    if rampup_length == 0:
        return 1.0
    else:
        current = np.clip(current / rampup_length, 0.0, 1.0)
        return float(current)

class SemiLoss(object):
    def __call__(self, outputs_x, targets_x, outputs_u, targets_u, epoch, max_epochs=30, lambda_u=75):
        probs_u = torch.softmax(outputs_u, dim=1)

        Lx = -torch.mean(torch.sum(F.log_softmax(outputs_x, dim=1) * targets_x, dim=1))
        Lu = torch.mean((probs_u - targets_u)**2)

        return Lx, Lu, lambda_u * linear_rampup(epoch, max_epochs)


def interleave_offsets(batch, nu):
    groups = [batch // (nu + 1)] * (nu + 1)
    for x in range(batch - sum(groups)):
        groups[-x - 1] += 1
    offsets = [0]
    for g in groups:
        offsets.append(offsets[-1] + g)
    assert offsets[-1] == batch
    return offsets

def interleave(xy, batch):
    nu = len(xy) - 1
    offsets = interleave_offsets(batch, nu)
    xy = [[v[offsets[p]:offsets[p + 1]] for p in range(nu + 1)] for v in xy]
    for i in range(1, nu + 1):
        xy[0][i], xy[i][i] = xy[i][i], xy[0][i]
    return [torch.cat(v, dim=0) for v in xy]