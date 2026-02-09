import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Callable, Dict, List, Optional, Tuple, Union
from models.masknet import ResNet, BasicBlock

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
    """Factorised Gaussian linear layer mirroring the UCL implementation."""

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
    

class BayesianClassifier(ResNet):
    def __init__(self, input_size: int, num_classes: int,args) -> None:
        super().__init__(BasicBlock, [2,2,2,2], input_size, num_classes)
        self.args = args
        n_tasks = args.tasks
        # Replace final linear layer with BayesianLinear
        classes_per_task = num_classes // n_tasks

        if args.disjoint_classifier:
            self.fc = BayesianLinear(512, num_classes, ratio=0.125)
        else:

            self.fc = nn.ModuleList(
                [BayesianLinear(512, classes_per_task, ratio=0.125)
                # [nn.Linear(512, classes_per_task)
                 for _ in range(n_tasks)]
            )

        self.split = not args.disjoint_classifier
        self.eval_samples = 20

    def forward(self, x, sample: bool = False, t: int = None, back_prop=None):
        out = F.relu(self.conv1(x))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        feats = F.avg_pool1d(out, self.avg_pool_out_dim)
        feats = feats.view(out.size(0), -1)
        if self.split:
            outputs = [head(feats, sample=True) for head in self.fc]
        else:
            outputs = self.fc(feats, sample=True)
        return outputs
    
    def fwd_eval(self, x, t: int = None):
        if not self.training:
            num_samples = max(1, self.eval_samples)
            if num_samples == 1:
                outputs = self.forward(x, sample=False)
                return outputs

            probs_acc: Optional[torch.Tensor] = None
            with torch.no_grad():
                for _ in range(num_samples):
                    sampled = self.forward(x, sample=True)
                    head_logits = sampled[t] if self.split else sampled
                    head_probs = F.softmax(head_logits, dim=-1)
                    probs_acc = head_probs if probs_acc is None else probs_acc + head_probs

            assert probs_acc is not None
            probs_mean = probs_acc / float(num_samples)
            return torch.log(probs_mean.clamp_min(1e-8))

        outputs = self.forward(x, sample=False)
        return outputs