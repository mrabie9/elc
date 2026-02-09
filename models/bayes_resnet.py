"""Uncertainty-guided Continual Learning (BLC) implementation for ``main.py``.

This rewrite adapts the original reference code – which relied on an external
training loop – so that it exposes the standard ``Net`` interface used across
this repository.  The core ingredients of BLC remain intact: a Bayesian linear
head with learnable mean and log-variance parameters, the KL-style penalties on
``mu``/``sigma`` drift, and the per-task snapshot that anchors the posterior of
all subsequent tasks.

For practicality we reuse the existing ``ResNet1D`` backbone as a deterministic
feature extractor while keeping the Bayesian machinery on the classifier
weights.  This mirrors the original setting where low-level filters are
deterministic and only the task-specific classifier carries uncertainty.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Callable, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from contextlib import contextmanager

from models.masknet import ResNet18_1d, ResNet34_1d, ResNet50_1d


def _calculate_fan_in_and_fan_out(tensor: torch.Tensor) -> Tuple[int, int]:
    if tensor.dim() < 2:
        raise ValueError("Tensor needs at least 2 dims to compute fan in/out")
    if tensor.dim() == 2:  # Linear layer
        fan_in = tensor.size(1)
        fan_out = tensor.size(0)
    else:
        num_input_fmaps = tensor.size(1)
        num_output_fmaps = tensor.size(0)
        receptive_field = tensor[0][0].numel()
        fan_in = num_input_fmaps * receptive_field
        fan_out = num_output_fmaps * receptive_field
    return fan_in, fan_out


class Gaussian:
    """Reparameterised Gaussian for Bayesian layers."""

    def __init__(self, mu: torch.Tensor, rho: torch.Tensor) -> None:
        self.mu = mu
        self.rho = rho
        self._normal = torch.distributions.Normal(0, 1)

    @property
    def sigma(self) -> torch.Tensor:
        return torch.log1p(torch.exp(self.rho))

    def sample(self) -> torch.Tensor:
        eps = self._normal.sample(self.mu.size()).to(self.mu.device)
        return self.mu + self.sigma * eps


class BayesianLinear(nn.Module):
    """Factorised Gaussian linear layer mirroring the BLC implementation."""

    def __init__(self, in_features: int, out_features: int, ratio: float = 0.5) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.weight_mu = nn.Parameter(torch.Tensor(out_features, in_features))
        fan_in, _ = _calculate_fan_in_and_fan_out(self.weight_mu)
        total_var = 2.0 / fan_in
        noise_var = total_var * ratio # spread of posterior over mu of weights (epistemic uncertainty)
        mu_var = total_var - noise_var # init variance of mu (to enable learning)

        noise_std = noise_var**0.5
        mu_std = mu_var**0.5
        bound = (3.0**0.5) * mu_std 
        nn.init.uniform_(self.weight_mu, -bound, bound) # init uniform distr. for mu in [-bound, bound]

        rho_init = float(torch.log(torch.expm1(torch.tensor(noise_std)))) # std = log(1 + exp(rho)) => rho = log(exp(std) - 1)
        self.weight_rho = nn.Parameter(torch.full((out_features, in_features), rho_init)) # each weight has its own rho
        self.bias = nn.Parameter(torch.zeros(out_features))
        self.weight = Gaussian(self.weight_mu, self.weight_rho)

    def forward(self, x: torch.Tensor, sample: bool = False) -> torch.Tensor:
        weight = self.weight.sample() if sample else self.weight.mu
        return F.linear(x, weight, self.bias)


@dataclass
class BLCConfig:
    lr: float = 1e-3
    lr_rho: float = 1e-2
    beta: float = 0.0002
    alpha: float = 0.3
    ratio: float = 0.125
    clipgrad: float = 10.0
    split: bool = True
    eval_samples: int = 20

    @staticmethod
    def from_args(args: object) -> "BLCConfig":
        cfg = BLCConfig()
        if hasattr(args, "clipgrad"):
            cfg.clipgrad = getattr(args, "clipgrad")
        if hasattr(args, "split"):
            cfg.split = getattr(args, "split")
        if hasattr(args, "ratio"):
            cfg.ratio = getattr(args, "ratio")
        if hasattr(args, "eval_samples"):
            cfg.eval_samples = max(1, int(getattr(args, "eval_samples")))
        return cfg


def _normalise_depth(depth_hint: Union[str, int, None]) -> int:
    if depth_hint is None:
        return 18
    if isinstance(depth_hint, str):
        if depth_hint.lower().startswith("resnet"):
            depth_hint = depth_hint[len("resnet") :]
        try:
            depth_hint = int(depth_hint)
        except ValueError as exc:  # pragma: no cover - defensive
            raise ValueError(f"Unsupported backbone spec '{depth_hint}'.") from exc
    if depth_hint in (18, 34, 50):
        return int(depth_hint)
    raise ValueError(f"Unsupported ResNet depth '{depth_hint}'. Expected one of {{18, 34, 50}}.")


def _get_backbone(
    input_size: int,
    num_classes: int,
    n_tasks: int,
    args: object,
) -> nn.Module:
    depth_hint = getattr(args, "bayes_backbone", None)
    if depth_hint is None:
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

    classes_per_task = None
    if getattr(args, "multi_head", False) and n_tasks > 1:
        classes_per_task = _compute_task_splits(num_classes, n_tasks)

    try:
        backbone = factory(input_size, num_classes, classes_per_task=classes_per_task)
    except TypeError:
        backbone = factory(input_size, num_classes)

    return backbone


def _compute_task_splits(num_classes: int, n_tasks: int) -> List[int]:
    if n_tasks <= 0:
        raise ValueError("Number of tasks must be positive.")
    base = num_classes // n_tasks
    remainder = num_classes % n_tasks
    splits = [base] * n_tasks
    for idx in range(remainder):
        splits[-(idx + 1)] += 1  # favour later tasks when classes do not divide evenly
    return splits


class BayesianClassifier(nn.Module):
    """ResNet1D feature extractor followed by per-task Bayesian heads."""

    def __init__(self, input_size: int, num_classes: int, n_tasks: int,
                 cfg: BLCConfig, args: object | None) -> None:
        super().__init__()
        self.cfg = cfg
        self.n_tasks = n_tasks
        self.num_classes = num_classes

        # Feature extractor (deterministic)
        self.feature_net = _get_backbone(input_size, num_classes, n_tasks, args)
        # for name, param in self.feature_net.named_parameters():
        #     if "w_mask" in name:
        #         param.requires_grad_(False)
        self.feature_dim = 512 #self.feature_net.avg_pool_out_dim
        # self._infer_feature_dim()
        self.classes_per_task = _compute_task_splits(num_classes, n_tasks)
        self.heads = nn.ModuleList(
            [
                BayesianLinear(self.feature_dim, task_classes, ratio=cfg.ratio)
                for task_classes in self.classes_per_task
            ]
        )

        self.split = not args.disjoint_classifier #cfg.split

    def forward(self, x: torch.Tensor, sample: bool = False) -> List[torch.Tensor] | torch.Tensor:
        feats = self._extract_features(x)
        outputs = [head(feats, sample=sample) for head in self.heads]
        # outputs = [head(feats) for head in self.heads]
        if self.split:
            return outputs
        return torch.cat(outputs, dim=1)

    def _infer_feature_dim(self) -> int:
        norm = getattr(self.feature_net, "bn", None)
        if norm is None or not hasattr(norm, "normalized_shape"):
            raise AttributeError("Feature backbone must expose a LayerNorm with 'normalized_shape'.")
        normalized_shape = norm.normalized_shape
        if isinstance(normalized_shape, int):
            return int(normalized_shape)
        feature_dim = 1
        for dim in normalized_shape:
            feature_dim *= int(dim)
        return feature_dim

    def _extract_features(self, x: torch.Tensor) -> torch.Tensor:
        net = self.feature_net
        # out = F.relu(net.bn1(net.conv1(x)))
        out = F.relu(net.conv1(x))
        out = net.layer1(out)
        out = net.layer2(out)
        out = net.layer3(out)
        out = net.layer4(out)
        pool_size = getattr(net, "avg_pool_out_dim", None)
        if pool_size is None:
            out = F.adaptive_avg_pool1d(out, 1)
        else:
            out = F.avg_pool1d(out, pool_size)
        out = out.view(out.size(0), -1)
        features = out
        # features = net.do(out)
        # features = net.bn(features)
        return features


class DisjointBayesianClassifier(nn.Module):
    """ResNet1D feature extractor with a single Bayesian head sliced per task."""

    def __init__(self, input_size: int, num_classes: int, n_tasks: int,
                 cfg: BLCConfig, args: object | None) -> None:
        super().__init__()
        self.cfg = cfg
        self.n_tasks = n_tasks
        self.num_classes = num_classes
        self.classes_per_task = _compute_task_splits(num_classes, n_tasks)

        self.feature_net = _get_backbone(input_size, num_classes, n_tasks, args)
        for name, param in self.feature_net.named_parameters():
            if "w_mask" in name:
                param.requires_grad_(False)
        self.feature_dim = 512#self._infer_feature_dim()

        self.head = BayesianLinear(self.feature_dim, num_classes, ratio=cfg.ratio)
        self.split = not args.disjoint_classifier
        self.task_slices: List[Tuple[int, int]] = []
        start = 0
        for task_classes in self.classes_per_task:
            end = start + task_classes
            self.task_slices.append((start, end))
            start = end

    def forward(self, x: torch.Tensor, sample: bool = False) -> List[torch.Tensor] | torch.Tensor:
        feats = self._extract_features(x)
        logits = self.head(feats, sample=sample)
        if not self.split:
            return logits
        return [logits[:, start:end] for start, end in self.task_slices]

    def _infer_feature_dim(self) -> int:
        norm = getattr(self.feature_net, "bn", None)
        if norm is None or not hasattr(norm, "normalized_shape"):
            raise AttributeError("Feature backbone must expose a LayerNorm with 'normalized_shape'.")
        normalized_shape = norm.normalized_shape
        if isinstance(normalized_shape, int):
            return int(normalized_shape)
        feature_dim = 1
        for dim in normalized_shape:
            feature_dim *= int(dim)
        return feature_dim

    def _extract_features(self, x: torch.Tensor) -> torch.Tensor:
        net = self.feature_net
        out = F.relu(net.conv1(x))
        # out = F.relu(net.bn1(net.conv1(x)))
        out = net.layer1(out)
        out = net.do(out)
        out = net.layer2(out)
        out = net.do(out)
        out = net.layer3(out)
        out = net.do(out)
        out = net.layer4(out)
        out = net.do(out)
        pool_size = getattr(net, "avg_pool_out_dim", None)
        if pool_size is None:
            out = F.adaptive_avg_pool1d(out, 1)
        else:
            out = F.avg_pool1d(out, pool_size)
        out = out.view(out.size(0), -1)
        # features = net.do(out)
        # features = net.bn(features)
        features = out
        return features


class Net(nn.Module):
    """Bayes learner compatible with ``main.py``."""

    def __init__(self, input_size: int, num_classes: int, args: object) -> None:
        super().__init__()

        self.cfg = BLCConfig.from_args(args)
        n_tasks = args.tasks if hasattr(args, "tasks") else NotImplementedError
        assert n_tasks > 0, "Number of tasks must be positive for BLC"
        self.args = args
        self.input_size = input_size
        self.num_classes = num_classes
        self.n_tasks = n_tasks
        self.classes_per_task = _compute_task_splits(num_classes, n_tasks)

        self.use_disjoint: bool = bool(getattr(args, "disjoint_classifier", False))
        self.classifier_cls = DisjointBayesianClassifier if self.use_disjoint else BayesianClassifier

        self.model = self.classifier_cls(input_size, num_classes, n_tasks, self.cfg, args)
        self.split = not args.disjoint_classifier

        # Optimiser: deterministic layers + Bayesian mu share lr, rho parameters get lr_rho
        mu_params: List[nn.Parameter] = []
        rho_params: List[nn.Parameter] = []
        if self.use_disjoint:
            head = self.model.head
            mu_params.extend([head.weight_mu, head.bias])
            rho_params.append(head.weight_rho)
        else:
            for head in self.model.heads:
                mu_params.extend([head.weight_mu, head.bias])
                rho_params.append(head.weight_rho)
        backbone_params = [
            param for name, param in self.model.feature_net.named_parameters()
            if not name.startswith(("ds_module", "dm_layer")) and "w_mask" not in name
        ]
        mu_params.extend(backbone_params)

        # self.optimizer = torch.optim.Adam(
        #     [
        #         {"params": mu_params, "lr": self.cfg.lr},
        #         {"params": rho_params, "lr": self.cfg.lr_rho},
        #     ],
        #     lr=self.cfg.lr,
        # )

        self.current_task: Optional[int] = None
        self.model_old: Optional[nn.Module] = None
        self.saved = False
        # self.ce = nn.CrossEntropyLoss(reduction="mean")
        self.is_task_incremental: bool = True

    # ------------------------------------------------------------------
    def compute_offsets(self, task):
        if self.is_task_incremental:
            start = sum(self.classes_per_task[:task])
            end = start + self.classes_per_task[task]
        else:
            start = 0
            end = self.num_classes
        return int(start), int(end)

    def _device(self) -> torch.device:
        return next(self.parameters()).device
    
    @contextmanager
    def _temporarily_enable_bn_training(self):
        bn_modules: List[nn.BatchNorm1d] = []
        states: List[bool] = []
        for module in self.model.modules():
            if isinstance(module, nn.BatchNorm1d):
                bn_modules.append(module)
                states.append(module.training)
                module.train(True)
        try:
            yield
        finally:
            for module, state in zip(bn_modules, states):
                module.train(state)

    def forward(self, x: torch.Tensor, t: int, s: Optional[float] = None) -> torch.Tensor:
        if not self.training:
            num_samples = max(1, self.cfg.eval_samples)
            if num_samples == 1:
                outputs = self.model(x, sample=False)
                return outputs[t] if self.split else outputs

            probs_acc: Optional[torch.Tensor] = None
            with torch.no_grad():
                with self._temporarily_enable_bn_training():
                    for _ in range(num_samples):
                        sampled = self.model(x, sample=True)
                        head_logits = sampled[t] if self.split else sampled
                        head_probs = F.softmax(head_logits, dim=-1)
                        probs_acc = head_probs if probs_acc is None else probs_acc + head_probs

            assert probs_acc is not None
            probs_mean = probs_acc / float(num_samples)
            return torch.log(probs_mean.clamp_min(1e-8))

        outputs = self.model(x, sample=False)
        return outputs[t] if self.split else outputs

    def observe(self, x: torch.Tensor, y: torch.Tensor, t: int, back_prop: bool = True) -> Tuple[float, float]:
        if (self.current_task is None) or (t != self.current_task):
            if self.current_task is not None:
                self.model_old = self._snapshot_model()
                self.saved = True
            self.current_task = t

        device = self._device()

        # if self.split:
        #     offset1, offset2 = self.compute_offsets(t)
        #     y_local = y.clone()
        #     y_local = y_local - offset1
        #     task_width = self.classes_per_task[t]
        #     if (y_local.min() < 0 ) or (y_local.max() >= task_width):
        #         raise ValueError(
        #             f"Labels out of range for task {t}: expected in [0, {task_width-1}] after offset, "
        #             f"got [{int(y_local.min())}, {int(y_local.max())}]"
        #         )
        #     y = y_local

        x = x.to(device)
        y = y.to(device)

        self.train()
        for module in self.model.modules():
            if isinstance(module, nn.BatchNorm1d):
                module.track_running_stats = False
        
        # offset1, offset2 = self.compute_offsets(t)
        outputs = self.model(x, sample=True)
        logits = outputs[t] if self.split else outputs

        preds = torch.argmax(logits, dim=1)
        tr_acc = (preds == y).float().mean().item()
        # loss = self._apply_regularisation(loss, y.size(0))

        if back_prop:
            loss = self.ce(logits, y)
            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if self.cfg.clipgrad > 0:
                torch.nn.utils.clip_grad_norm_(self.parameters(), self.cfg.clipgrad)
            self.optimizer.step()
            return float(loss.detach().cpu()), tr_acc
        else:
            return logits, tr_acc

    def on_epoch_end(self) -> None:  # pragma: no cover - hook for symmetry
        pass

    # ------------------------------------------------------------------
    def _snapshot_model(self) -> nn.Module:
        clone = self.classifier_cls(
            self.input_size,
            self.num_classes,
            self.n_tasks,
            self.cfg,
            self.args,
        )
        clone.load_state_dict(self.model.state_dict())
        clone.to(self._device())
        clone.eval()
        for param in clone.parameters():
            param.requires_grad_(False)
        return clone

    def _apply_regularisation(self, base_loss: torch.Tensor, batch_size: int) -> torch.Tensor:
        if not self.saved or self.model_old is None:
            return base_loss
        if self.use_disjoint:
            return self._apply_regularisation_disjoint(base_loss, batch_size)
        return self._apply_regularisation_multihead(base_loss, batch_size)

    def _apply_regularisation_multihead(self, base_loss: torch.Tensor, batch_size: int) -> torch.Tensor:
        sigma_weight_reg = torch.zeros_like(base_loss)
        sigma_weight_normal_reg = torch.zeros_like(base_loss)
        mu_weight_reg = torch.zeros_like(base_loss)
        mu_bias_reg = torch.zeros_like(base_loss)
        l1_mu_weight_reg = torch.zeros_like(base_loss)
        l1_mu_bias_reg = torch.zeros_like(base_loss)

        eps = 1e-8
        assert self.model_old is not None
        for old_head, new_head in zip(self.model_old.heads, self.model.heads):
            trainer_weight_mu = new_head.weight_mu
            saver_weight_mu = old_head.weight_mu
            trainer_bias = new_head.bias
            saver_bias = old_head.bias

            trainer_weight_sigma = torch.log1p(torch.exp(new_head.weight_rho))
            saver_weight_sigma = torch.log1p(torch.exp(old_head.weight_rho))

            fan_in, _ = _calculate_fan_in_and_fan_out(trainer_weight_mu)
            std_init = math.sqrt((2.0 / fan_in) * self.cfg.ratio)

            saver_strength = std_init / (saver_weight_sigma + eps)
            bias_strength = saver_strength.mean(dim=1)

            mu_weight_reg = mu_weight_reg + ((saver_strength * (trainer_weight_mu - saver_weight_mu)) ** 2).sum()
            mu_bias_reg = mu_bias_reg + ((bias_strength * (trainer_bias - saver_bias)) ** 2).sum()

            l1_mu_weight_reg = l1_mu_weight_reg + (
                (saver_weight_mu.pow(2) / saver_weight_sigma.pow(2))
                * (trainer_weight_mu - saver_weight_mu).abs()
            ).sum()
            l1_mu_bias_reg = l1_mu_bias_reg + (
                (saver_bias.pow(2) / (saver_weight_sigma.mean(dim=1).pow(2) + eps))
                * (trainer_bias - saver_bias).abs()
            ).sum()

            weight_sigma_ratio = trainer_weight_sigma.pow(2) / (saver_weight_sigma.pow(2) + eps)
            sigma_weight_reg = sigma_weight_reg + (weight_sigma_ratio - torch.log(weight_sigma_ratio + eps)).sum()
            sigma_weight_normal_reg = sigma_weight_normal_reg + (
                trainer_weight_sigma.pow(2) - torch.log(trainer_weight_sigma.pow(2) + eps)
            ).sum()

        loss = base_loss
        loss = loss + self.cfg.alpha * (mu_weight_reg + mu_bias_reg) / (2 * batch_size)
        loss = loss + self.saved * (l1_mu_weight_reg + l1_mu_bias_reg) / batch_size
        loss = loss + self.cfg.beta * (sigma_weight_reg + sigma_weight_normal_reg) / (2 * batch_size)
        return loss

    def _apply_regularisation_disjoint(self, base_loss: torch.Tensor, batch_size: int) -> torch.Tensor:
        sigma_weight_reg = torch.zeros_like(base_loss)
        sigma_weight_normal_reg = torch.zeros_like(base_loss)
        mu_weight_reg = torch.zeros_like(base_loss)
        mu_bias_reg = torch.zeros_like(base_loss)
        l1_mu_weight_reg = torch.zeros_like(base_loss)
        l1_mu_bias_reg = torch.zeros_like(base_loss)

        eps = 1e-8
        assert self.model_old is not None
        current_head = self.model.head
        previous_head = self.model_old.head
        for start, end in self.model.task_slices:
            trainer_weight_mu = current_head.weight_mu[start:end]
            saver_weight_mu = previous_head.weight_mu[start:end]
            trainer_bias = current_head.bias[start:end]
            saver_bias = previous_head.bias[start:end]

            trainer_weight_sigma = torch.log1p(torch.exp(current_head.weight_rho[start:end]))
            saver_weight_sigma = torch.log1p(torch.exp(previous_head.weight_rho[start:end]))

            fan_in, _ = _calculate_fan_in_and_fan_out(trainer_weight_mu)
            std_init = math.sqrt((2.0 / fan_in) * self.cfg.ratio)

            saver_strength = std_init / (saver_weight_sigma + eps)
            bias_strength = saver_strength.mean(dim=1)

            mu_weight_reg = mu_weight_reg + ((saver_strength * (trainer_weight_mu - saver_weight_mu)) ** 2).sum()
            mu_bias_reg = mu_bias_reg + ((bias_strength * (trainer_bias - saver_bias)) ** 2).sum()

            l1_mu_weight_reg = l1_mu_weight_reg + (
                (saver_weight_mu.pow(2) / saver_weight_sigma.pow(2))
                * (trainer_weight_mu - saver_weight_mu).abs()
            ).sum()
            l1_mu_bias_reg = l1_mu_bias_reg + (
                (saver_bias.pow(2) / (saver_weight_sigma.mean(dim=1).pow(2) + eps))
                * (trainer_bias - saver_bias).abs()
            ).sum()

            weight_sigma_ratio = trainer_weight_sigma.pow(2) / (saver_weight_sigma.pow(2) + eps)
            sigma_weight_reg = sigma_weight_reg + (weight_sigma_ratio - torch.log(weight_sigma_ratio + eps)).sum()
            sigma_weight_normal_reg = sigma_weight_normal_reg + (
                trainer_weight_sigma.pow(2) - torch.log(trainer_weight_sigma.pow(2) + eps)
            ).sum()

        loss = base_loss
        loss = loss + self.cfg.alpha * (mu_weight_reg + mu_bias_reg) / (2 * batch_size)
        loss = loss + self.saved * (l1_mu_weight_reg + l1_mu_bias_reg) / batch_size
        loss = loss + self.cfg.beta * (sigma_weight_reg + sigma_weight_normal_reg) / (2 * batch_size)
        return loss
    
    @torch.no_grad()
    def mc_epistemic_classification(self, x, t, S=20, temperature=1.0, clamp_eps=1e-16):
        """
        Monte-Carlo epistemic uncertainty for classification.
        Assumes model(x, sample=True/False) returns logits.
        Returns:
            p_mean: (B, C) predictive probabilities
            H_pred: (B,) predictive entropy H[p_mean]
            EH:     (B,) expected entropy E_s[ H[p_s] ]
            MI:     (B,) mutual information H[p_mean] - E_s[H[p_s]]  (epistemic)
        """
        model = self.model
        model.eval()
        logits_accum = []
        probs_accum  = []

        for _ in range(S):
            logits = model(x, sample=True)[t] if self.split else model(x, sample=True) #/ temperature                  # (B, C)
            probs  = F.softmax(logits, dim=-1)                            # (B, C)
            # logits_accum.append(logits)
            probs_accum.append(probs)

        # Stack over samples
        probs_stack = torch.stack(probs_accum, dim=0)                     # (S, B, C)

        # Predictive mean probability
        p_mean = probs_stack.mean(dim=0)                                  # (B, C)

        # Entropy of the mean H[p_mean]
        p_mean_clamped = p_mean.clamp(min=clamp_eps, max=1.0)
        H_pred = -(p_mean_clamped * p_mean_clamped.log()).sum(dim=-1)     # (B,)

        # Expected entropy E_s[ H[p_s] ]
        probs_clamped = probs_stack.clamp(min=clamp_eps, max=1.0)
        entropies = -(probs_clamped * probs_clamped.log()).sum(dim=-1)    # (S, B)
        EH = entropies.mean(dim=0)                                        # (B,)

        # Mutual information (epistemic)
        MI = H_pred - EH                                                   # (B,)

        return p_mean, H_pred, EH, MI

class DisjointNet(Net):
    """Convenience wrapper that forces the shared-head Bayes ResNet variant."""

    def __init__(self, input_size: int, num_classes: int, args: object) -> None:
        setattr(args, "disjoint_classifier", True)
        super().__init__(input_size, num_classes, args)


__all__ = ["Net", "DisjointNet"]
