"""PCGrad bridge with torch-optimizer integration and local fallback."""

from __future__ import annotations

from typing import Iterable, List

import torch


class PCGrad:
    """Project conflicting task gradients before optimizer step.

    This is a lightweight local fallback implementation used when
    ``torch_optimizer.PCGrad`` is unavailable in the installed package.
    """

    def __init__(
        self, optimizer: torch.optim.Optimizer, reduction: str = "mean", eps: float = 1e-12
    ) -> None:
        self.optimizer = optimizer
        self.reduction = str(reduction).strip().lower()
        if self.reduction not in {"mean", "sum"}:
            raise ValueError("PCGrad reduction must be one of: mean, sum")
        self.eps = float(eps)

    @property
    def _params(self) -> List[torch.nn.Parameter]:
        params: List[torch.nn.Parameter] = []
        for group in self.optimizer.param_groups:
            for p in group.get("params", []):
                if p is not None and p.requires_grad:
                    params.append(p)
        return params

    def zero_grad(self, set_to_none: bool = True) -> None:
        self.optimizer.zero_grad(set_to_none=set_to_none)

    def step(self) -> None:
        self.optimizer.step()

    @staticmethod
    def _dot(grads_a: List[torch.Tensor], grads_b: List[torch.Tensor]) -> torch.Tensor:
        return sum((ga * gb).sum() for ga, gb in zip(grads_a, grads_b))

    @staticmethod
    def _norm_sq(grads: List[torch.Tensor]) -> torch.Tensor:
        return sum((g * g).sum() for g in grads)

    def _collect_grads(self) -> List[torch.Tensor]:
        grads: List[torch.Tensor] = []
        for p in self._params:
            if p.grad is None:
                grads.append(torch.zeros_like(p, memory_format=torch.preserve_format))
            else:
                grads.append(p.grad.detach().clone())
        return grads

    def pc_backward(self, objectives: Iterable[torch.Tensor]) -> None:
        tasks = list(objectives)
        if not tasks:
            raise ValueError("pc_backward requires at least one objective tensor")

        # Capture per-task gradients.
        per_task_grads: List[List[torch.Tensor]] = []
        for idx, task_loss in enumerate(tasks):
            self.zero_grad(set_to_none=True)
            task_loss.backward(retain_graph=idx < (len(tasks) - 1))
            per_task_grads.append(self._collect_grads())

        # Project conflicts: if dot(g_i, g_j) < 0 then remove conflicting component.
        projected: List[List[torch.Tensor]] = [
            [g.clone() for g in task_grads] for task_grads in per_task_grads
        ]
        n_tasks = len(projected)

        for i in range(n_tasks):
            order = torch.randperm(n_tasks).tolist()
            for j in order:
                if i == j:
                    continue
                dot_ij = self._dot(projected[i], per_task_grads[j])
                if dot_ij >= 0:
                    continue
                denom = self._norm_sq(per_task_grads[j]).clamp_min(self.eps)
                coeff = dot_ij / denom
                projected[i] = [gi - coeff * gj for gi, gj in zip(projected[i], per_task_grads[j])]

        # Merge projected gradients across tasks.
        merged: List[torch.Tensor] = []
        for p_idx in range(len(projected[0])):
            stack = torch.stack([projected[t][p_idx] for t in range(n_tasks)], dim=0)
            if self.reduction == "sum":
                merged.append(stack.sum(dim=0))
            else:
                merged.append(stack.mean(dim=0))

        # Write final merged gradients back to parameters.
        self.zero_grad(set_to_none=True)
        for p, g in zip(self._params, merged):
            p.grad = g


def create_pcgrad_optimizer(
    optimizer: torch.optim.Optimizer,
    reduction: str = "mean",
):
    """Create a PCGrad wrapper.

    Preference order:
    1) Use torch_optimizer.PCGrad when available.
    2) Fall back to local PCGrad implementation.
    """
    try:
        import torch_optimizer as torch_optim

        pcgrad_cls = getattr(torch_optim, "PCGrad", None)
        if pcgrad_cls is not None:
            wrapped = pcgrad_cls(optimizer)
            setattr(wrapped, "_pcgrad_backend", "torch_optimizer")
            return wrapped
    except Exception:
        pass

    wrapped = PCGrad(optimizer=optimizer, reduction=reduction)
    setattr(wrapped, "_pcgrad_backend", "local")
    return wrapped
