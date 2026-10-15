"""Canonical Algorithm 1 applied solely to ADD; server-only source candidate.

Algorithm 1 differentiates the query objective with respect to the base theta.
The paper's Eq. 16b instead writes phi; this implementation interpretation is
recorded explicitly in the source contract. No first-order approximation.
"""
from collections import OrderedDict
import torch
from torch.utils.checkpoint import checkpoint
from .model import complex_mse


def add_meta_step(model, support_x, support_h, query_x, query_h,
                  alpha=1e-3, beta=1e-3, microbatch=1, deadline_check=lambda: None):
    if support_x.shape[0] != 16 or query_x.shape[0] != 16:
        raise ValueError("Equal-budget candidate requires 16 support + 16 query frames")
    theta = OrderedDict(model.add.named_parameters())
    if microbatch not in (1, 2, 4, 8):
        raise ValueError("Unsupported exact-episode microbatch")
    names = tuple(theta)

    def objective(inputs, truth, parameters):
        terms = []
        for start in range(0, 16, microbatch):
            deadline_check()
            x = inputs[start:start + microbatch]
            h = truth[start:start + microbatch]
            # Explicit parameter arguments are essential: a recomputation must
            # re-enter functional_call with phi, never read restored base theta.
            def loss_function(*values, x=x, h=h):
                return complex_mse(model.add_prediction(
                    x, OrderedDict(zip(names, values))), h)
            loss = checkpoint(loss_function, *tuple(parameters.values()),
                              use_reentrant=False, preserve_rng_state=True)
            terms.append(loss * (x.shape[0] / 16))
        return sum(terms)

    support_loss = objective(support_x, support_h, theta)
    if not bool(torch.isfinite(support_loss)):
        raise FloatingPointError("Nonfinite support loss")
    support_grad = torch.autograd.grad(support_loss, tuple(theta.values()),
                                       create_graph=True, allow_unused=False)
    adapted = OrderedDict((name, value - alpha * grad)
                          for (name, value), grad in zip(theta.items(), support_grad))
    query_loss = objective(query_x, query_h, adapted)
    if not bool(torch.isfinite(query_loss)):
        raise FloatingPointError("Nonfinite query loss")
    meta_grad = torch.autograd.grad(query_loss, tuple(theta.values()),
                                   create_graph=False, allow_unused=False)
    if not all(torch.isfinite(grad).all().item() for grad in meta_grad):
        raise FloatingPointError("Nonfinite second-order meta gradient")
    # K=1 here. Thus Algorithm 1's task-sum equals this single query loss.
    deadline_check()
    with torch.no_grad():
        for value, grad in zip(theta.values(), meta_grad):
            value.add_(grad, alpha=-beta)
    return {"support_loss": float(support_loss.detach()),
            "query_loss": float(query_loss.detach()),
            "support_frames": 16, "query_frames": 16,
            "inner_steps": 1, "outer_updates": 1, "second_order": True}
