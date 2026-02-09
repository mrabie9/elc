"""Evidential (Dempster-Shafer) ResNet wrapper for ``main.py``."""

from __future__ import annotations

from typing import Callable, Dict, List, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from dst import Dempster_Shafer_module, DM
from models.masknet import BasicBlock, Bottleneck, MaskConv1d


def _compute_task_splits(num_classes: int, n_tasks: int) -> List[int]:
    if n_tasks <= 0:
        raise ValueError("Number of tasks must be positive.")
    base = num_classes // n_tasks
    remainder = num_classes % n_tasks
    splits = [base] * n_tasks
    for idx in range(remainder):
        splits[-(idx + 1)] += 1  # favour later tasks when classes do not divide evenly
    return splits


def _normalise_depth(depth_hint: Union[str, int, None]) -> int:
    if depth_hint is None:
        return 18
    if isinstance(depth_hint, str):
        if depth_hint.lower().startswith("resnet"):
            depth_hint = depth_hint[len("resnet") :]
        try:
            depth_hint = int(depth_hint)
        except ValueError as exc:
            raise ValueError(f"Unsupported backbone spec '{depth_hint}'.") from exc
    if depth_hint in (18, 34, 50):
        return int(depth_hint)
    raise ValueError(f"Unsupported ResNet depth '{depth_hint}'. Expected one of {{18, 34, 50}}.")


class EvidentialResNet(nn.Module):
    def __init__(
        self,
        block: nn.Module,
        num_blocks: Sequence[int],
        slice_size: int,
        num_classes: int,
        classes_per_task: Optional[List[int]] = None,
        nu: Union[float, Sequence[float]] = 0.9,
    ) -> None:
        super().__init__()
        self.in_planes = 64
        self.slice_size = slice_size
        self.classes_per_task = classes_per_task

        self.conv1 = MaskConv1d(2, 64, kernel_size=7, stride=2, padding=1, bias=False)
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.do = nn.Dropout(p=0.2)

        if slice_size == 512:
            self.avg_pool_out_dim = int(self.slice_size / 64)
        else:
            self.avg_pool_out_dim = int(self.slice_size / 16)

        self.bn = nn.LayerNorm(512 * block.expansion)

        if self.classes_per_task is not None:
            self.ds_module = nn.ModuleList(
                [
                    Dempster_Shafer_module(
                        n_feature_maps=512 * block.expansion,
                        n_classes=task_classes,
                        n_prototypes=task_classes * 20,
                    )
                    for task_classes in self.classes_per_task
                ]
            )
            self.dm_layer = nn.ModuleList(
                [
                    DM(
                        num_class=task_classes,
                        nu=float(nu[idx]) if isinstance(nu, (list, tuple)) else float(nu),
                        device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
                    )
                    for idx, task_classes in enumerate(self.classes_per_task)
                ]
            )
        else:
            self.ds_module = Dempster_Shafer_module(
                n_feature_maps=512 * block.expansion,
                n_classes=num_classes,
                n_prototypes=num_classes * 20,
            )
            self.dm_layer = DM(
                num_class=num_classes,
                nu=float(nu) if not isinstance(nu, (list, tuple)) else float(nu[0]),
                device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
            )

    def _make_layer(self, block: nn.Module, planes: int, num_blocks: int, stride: int) -> nn.Sequential:
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(
        self,
        x: torch.Tensor,
        task_id: Optional[int] = None,
        return_features: bool = False,
    ):
        out = F.relu(self.conv1(x))
        out = self.layer1(out)
        out = self.do(out)
        out = self.layer2(out)
        out = self.do(out)
        out = self.layer3(out)
        out = self.do(out)
        out = self.layer4(out)
        out = self.do(out)
        out = F.adaptive_avg_pool1d(out, 1)
        out = out.view(out.size(0), -1)
        features = self.do(out)

        if torch.isnan(features).any():
            raise ValueError("Model features contains NaNs")

        features = self.bn(features)

        if torch.isnan(features).any():
            raise ValueError("Model norm features contains NaNs")

        if task_id is not None and isinstance(self.ds_module, nn.ModuleList):
            mass_normalized = self.ds_module[task_id](features)
            out = self.dm_layer[task_id](mass_normalized)
        else:
            mass_normalized = self.ds_module(features)
            out = self.dm_layer(mass_normalized)

        if torch.isnan(mass_normalized).any():
            raise ValueError("Model mass_normalized contains NaNs")

        omegas = mass_normalized[:, -1]
        beliefs = mass_normalized[:, :-1]

        if torch.isnan(out).any():
            raise ValueError("Model dm_layer contains NaNs")

        if return_features:
            return out, features, omegas, beliefs
        return out


def ResNet18_1d(
    slice_size: int,
    num_classes: int,
    classes_per_task: Optional[List[int]] = None,
    nu: Union[float, Sequence[float]] = 0.9,
) -> EvidentialResNet:
    return EvidentialResNet(BasicBlock, [2, 2, 2, 2], slice_size, num_classes, classes_per_task, nu=nu)


def ResNet34_1d(
    slice_size: int,
    num_classes: int,
    classes_per_task: Optional[List[int]] = None,
    nu: Union[float, Sequence[float]] = 0.9,
) -> EvidentialResNet:
    return EvidentialResNet(BasicBlock, [3, 4, 6, 3], slice_size, num_classes, classes_per_task, nu=nu)


def ResNet50_1d(
    slice_size: int,
    num_classes: int,
    classes_per_task: Optional[List[int]] = None,
    nu: Union[float, Sequence[float]] = 0.9,
) -> EvidentialResNet:
    return EvidentialResNet(Bottleneck, [3, 4, 6, 3], slice_size, num_classes, classes_per_task, nu=nu)


def _get_backbone(
    input_size: int,
    num_classes: int,
    n_tasks: int,
    args: object,
    classes_per_task: Optional[List[int]],
    nu: Union[float, Sequence[float]],
) -> nn.Module:
    depth_hint = getattr(args, "depth", 18)
    try:
        depth = _normalise_depth(depth_hint)
    except ValueError:
        depth = 18

    factory_map: Dict[int, Callable[..., nn.Module]] = {
        18: ResNet18_1d,
        34: ResNet34_1d,
        50: ResNet50_1d,
    }
    factory = factory_map.get(depth, ResNet18_1d)

    try:
        backbone = factory(input_size, num_classes, classes_per_task=classes_per_task, nu=nu)
    except TypeError:
        backbone = factory(input_size, num_classes)

    return backbone


class Net(nn.Module):
    """Evidential ResNet wrapper compatible with ``main.py``."""

    def __init__(self, input_size: int, num_classes: int, args: object) -> None:
        super().__init__()

        n_tasks = getattr(args, "tasks", 1)
        classes_per_task = None
        if getattr(args, "multi_head", False) and n_tasks > 1:
            classes_per_task = _compute_task_splits(num_classes, n_tasks)

        nu = getattr(args, "nu", 0.9)
        self.model = _get_backbone(
            input_size,
            num_classes,
            n_tasks,
            args,
            classes_per_task=classes_per_task,
            nu=nu,
        )
        self.classes_per_task = classes_per_task or [num_classes]
        self.split = classes_per_task is not None

    def forward(self, x: torch.Tensor, t: Optional[int] = None, return_features: bool = False):
        if self.split:
            return self.model(x, task_id=t, return_features=return_features)
        return self.model(x, return_features=return_features)


__all__ = ["Net", "ResNet18_1d", "ResNet34_1d", "ResNet50_1d"]
