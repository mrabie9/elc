import torch
import numpy as np
import torch.nn.functional as F

class Distance_layer(torch.nn.Module):
    '''
    verified
    '''
    def __init__(self, n_prototypes, n_feature_maps):
        super(Distance_layer, self).__init__()
        self.w = torch.nn.Linear(in_features=n_feature_maps, out_features=n_prototypes, bias=False).weight
        self.n_prototypes = n_prototypes

    def forward(self, inputs):
        for i in range(self.n_prototypes):
            if i == 0:
                un_mass_i = (self.w[i, :] - inputs) ** 2
                un_mass_i = torch.sum(un_mass_i, dim=-1, keepdim=True)
                un_mass = un_mass_i

            if i >= 1:
                un_mass_i = (self.w[i, :] - inputs) ** 2
                un_mass_i = torch.sum(un_mass_i, dim=-1, keepdim=True)
                un_mass = torch.cat([un_mass, un_mass_i], -1)
        return un_mass

class DistanceActivation_layer(torch.nn.Module):
    '''
    verified
    '''
    def __init__(self, n_prototypes,init_alpha=0,init_gamma=0.1):
        super(DistanceActivation_layer, self).__init__()
        self.eta = torch.nn.Linear(in_features=n_prototypes, out_features=1, bias=False)#.weight.data.fill_(torch.from_numpy(np.array(init_gamma)).to(device))
        self.xi = torch.nn.Linear(in_features=n_prototypes, out_features=1, bias=False)#.weight.data.fill_(torch.from_numpy(np.array(init_alpha)).to(device))
        #torch.nn.init.kaiming_uniform_(self.eta.weight)
        #torch.nn.init.kaiming_uniform_(self.xi.weight)
        torch.nn.init.constant_(self.eta.weight,init_gamma)
        torch.nn.init.constant_(self.xi.weight,init_alpha)
        #self.alpha_test = 1/(torch.exp(-self.xi.weight)+1)
        self.n_prototypes = n_prototypes
        self.alpha = None

    def forward(self, inputs):
        with torch.autograd.set_detect_anomaly(True):  # Enable backward NaN/Inf tracing

            gamma = torch.square(self.eta.weight)
            if torch.isnan(gamma).any() or torch.isinf(gamma).any():
                raise ValueError("NaN/Inf in gamma (eta.weight squared)")

            alpha = torch.neg(self.xi.weight)
            if torch.isnan(alpha).any() or torch.isinf(alpha).any():
                raise ValueError("NaN/Inf in neg(xi.weight)")

            alpha = torch.exp(alpha) + 1
            if torch.isnan(alpha).any() or torch.isinf(alpha).any():
                raise ValueError("NaN/Inf in exp(alpha)+1")

            alpha = torch.div(1, alpha)
            if torch.isnan(alpha).any() or torch.isinf(alpha).any():
                raise ValueError("NaN/Inf in 1/alpha")

            self.alpha = alpha

            si = torch.mul(gamma, inputs)
            if torch.isnan(si).any() or torch.isinf(si).any():
                raise ValueError("NaN/Inf in gamma * inputs")

            si = torch.neg(si)
            if torch.isnan(si).any() or torch.isinf(si).any():
                raise ValueError("NaN/Inf in -si")

            si = torch.exp(si)
            if torch.isnan(si).any() or torch.isinf(si).any():
                raise ValueError("NaN/Inf in exp(si)")

            si = torch.mul(si, alpha)
            if torch.isnan(si).any() or torch.isinf(si).any():
                raise ValueError("NaN/Inf in si * alpha")

            max_val, max_idx = torch.max(si, dim=-1, keepdim=True)
            if torch.isnan(max_val).any() or torch.isinf(max_val).any():
                raise ValueError("NaN/Inf in max(si)")

            si = si / (max_val + 1e-4)  # avoid divide-by-zero
            if torch.isnan(si).any() or torch.isinf(si).any():
                raise ValueError("NaN/Inf in si normalization")

        return si



'''class Belief_layer(torch.nn.Module):
    def __init__(self, prototypes, num_class):
        super(DS2, self).__init__()
        self.beta = torch.nn.Linear(in_features=prototypes, out_features=num_class, bias=False).weight
        self.prototypes = prototypes
        self.num_class = num_class

    def forward(self, inputs):
        beta = torch.square(self.beta)
        beta_sum = torch.sum(beta, dim=0, keepdim=True)
        self.u = torch.div(beta, beta_sum)
        inputs_new = torch.unsqueeze(inputs, dim=-2)
        for i in range(self.prototypes):
            if i == 0:
                mass_prototype_i = torch.mul(self.u[:, i], inputs_new[..., i])  #batch_size * n_class
                mass_prototype = torch.unsqueeze(mass_prototype_i, -2)
            if i > 0:
                mass_prototype_i = torch.unsqueeze(torch.mul(self.u[:, i], inputs_new[..., i]), -2)
                mass_prototype = torch.cat([mass_prototype, mass_prototype_i], -2)
        return mass_prototype'''

class Belief_layer(torch.nn.Module):
    '''
    verified
    '''
    def __init__(self, n_prototypes, num_class):
        super(Belief_layer, self).__init__()
        self.beta = torch.nn.Linear(in_features=n_prototypes, out_features=num_class, bias=False).weight
        self.num_class = num_class
    def forward(self, inputs):
        beta = torch.square(self.beta)
        beta_sum = torch.sum(beta, dim=0, keepdim=True)
        u = torch.div(beta, beta_sum)
        mass_prototype = torch.einsum('cp,b...p->b...pc',u, inputs)
        return mass_prototype

class Omega_layer(torch.nn.Module):
    '''
    verified, give same results

    '''
    def __init__(self, n_prototypes, num_class):
        super(Omega_layer, self).__init__()
        self.n_prototypes = n_prototypes
        self.num_class = num_class

    def forward(self, inputs):
        mass_omega_sum = 1 - torch.sum(inputs, -1, keepdim=True)
        #mass_omega_sum = 1. - mass_omega_sum[..., 0]
        #mass_omega_sum = torch.unsqueeze(mass_omega_sum, -1)
        mass_with_omega = torch.cat([inputs, mass_omega_sum], -1)
        return mass_with_omega


class Dempster_layer(torch.nn.Module):
    '''
    verified give same results

    '''
    def __init__(self, n_prototypes, num_class):
        super(Dempster_layer, self).__init__()
        self.n_prototypes = n_prototypes
        self.num_class = num_class

    def forward(self, inputs):
        m1 = inputs[..., 0, :]
        omega1 = torch.unsqueeze(inputs[..., 0, -1], -1)
        for i in range(self.n_prototypes - 1):
            m2 = inputs[..., (i + 1), :]
            omega2 = torch.unsqueeze(inputs[..., (i + 1), -1], -1)
            combine1 = torch.mul(m1, m2)
            combine2 = torch.mul(m1, omega2)
            combine3 = torch.mul(omega1, m2)
            combine1_2 = combine1 + combine2
            combine2_3 = combine1_2 + combine3
            combine2_3 = combine2_3 / torch.sum(combine2_3, dim=-1, keepdim=True) # potential div 0
            m1 = combine2_3
            omega1 = torch.unsqueeze(combine2_3[..., -1], -1)
        return m1


class DempsterNormalize_layer(torch.nn.Module):
    '''
    verified

    '''
    def __init__(self):
        super(DempsterNormalize_layer, self).__init__()
    def forward(self, inputs):
        mass_combine_normalize = inputs / torch.sum(inputs, dim=-1, keepdim=True) # potential 0 div if dempster layer gives 0
        # print(torch.sum(mass_combine_normalize[0,:]))
        return mass_combine_normalize


class Dempster_Shafer_module(torch.nn.Module):
    def __init__(self, n_feature_maps, n_classes, n_prototypes):
        super(Dempster_Shafer_module, self).__init__()
        self.n_prototypes = n_prototypes
        self.n_classes = n_classes
        self.n_feature_maps = n_feature_maps
        self.ds1 = Distance_layer(n_prototypes=self.n_prototypes, n_feature_maps=self.n_feature_maps)
        self.ds1_activate = DistanceActivation_layer(n_prototypes = self.n_prototypes)
        self.ds2 = Belief_layer(n_prototypes= self.n_prototypes, num_class=self.n_classes)
        self.ds2_omega = Omega_layer(n_prototypes= self.n_prototypes,num_class= self.n_classes)
        self.ds3_dempster = Dempster_layer(n_prototypes= self.n_prototypes,num_class= self.n_classes)
        self.ds3_normalize = DempsterNormalize_layer()

    def forward(self, inputs):
        '''
        '''
        ED = self.ds1(inputs)
        ED_ac = self.ds1_activate(ED)
        mass_prototypes = self.ds2(ED_ac)
        mass_prototypes_omega = self.ds2_omega(mass_prototypes)
        mass_Dempster = self.ds3_dempster(mass_prototypes_omega)
        mass_Dempster_normalize = self.ds3_normalize(mass_Dempster)
        return mass_Dempster_normalize

def tile(a, dim, n_tile, device):
    init_dim = a.size(dim)
    repeat_idx = [1] * a.dim()
    repeat_idx[dim] = n_tile
    a = a.repeat(*(repeat_idx))
    order_index = torch.LongTensor(np.concatenate([init_dim * np.arange(n_tile) + i for i in range(init_dim)])).to(
        device)
    return torch.index_select(a, dim, order_index)


class DM(torch.nn.Module):
    def __init__(self, num_class, nu=0.9, device=torch.device('cpu')):
        super(DM, self).__init__()
        self.nu = nu
        self.num_class = num_class
        self.device = device

    def forward(self, inputs):
        # upper = torch.unsqueeze((1 - self.nu) * inputs[..., -1], -1)  # here 0.1 = 1 - \nu
        # upper = tile(upper, dim=-1, n_tile=self.num_class + 1, device=self.device)
        # outputs = (inputs + upper)#[..., :-1]
        
        upper = torch.unsqueeze((1 - self.nu) * inputs[..., -1], -1)  # shape [B, 1]
        upper_tiled = tile(upper, dim=-1, n_tile=self.num_class, device=self.device)
        beliefs = inputs[..., :-1] + upper_tiled
        omega = self.nu * inputs[..., -1:]  # keep omega as last column

        outputs = torch.cat([beliefs, omega], dim=-1)

        return outputs

import torch

def _owa_first_weight(set_size: int, gamma: float) -> float:
    """
    Compute g1(s, gamma) by max-entropy OWA under the TDI constraint:
        sum_k w_k g_k = gamma,  sum_k g_k = 1, g_k>=0,
    where w_k = (s - k) / (s - 1) (descending ranks), s = |A|.
    Returns the first weight g1 (applied to the largest element).
    For s=1, define g1=1.
    """
    if set_size <= 1:
        return 1.0
    s = set_size
    w = torch.linspace(1.0, 0.0, steps=s)  # w_k = (s-k)/(s-1), k=1..s
    w = (torch.arange(s, dtype=torch.double) * 0.0 + 0)  # placeholder to keep dtype
    w = torch.tensor([(s - (k+1)) / (s - 1) for k in range(s)], dtype=torch.double)

    # Max-entropy with linear constraints -> exponential family: g_k ∝ exp(λ * w_k)
    # Solve for λ by bisection so that sum g_k w_k = gamma.
    def mean_w(lambda_):
        g = torch.exp(lambda_ * w)
        g = g / g.sum()
        return (g * w).sum(), g[0]  # also returns g1 for convenience

    # Bisection bounds: lambda-> -inf pushes mass to small w (≈0), -> +inf to large w (≈1)
    lo, hi = -100.0, 100.0
    for _ in range(60):
        mid = (lo + hi) / 2.0
        mw, _ = mean_w(mid)
        if mw.item() < gamma:
            lo = mid
        else:
            hi = mid
    _, g1 = mean_w((lo + hi) / 2.0)
    return float(g1)

def set_valued_predict_from_masses(
    beliefs: torch.Tensor,  # [B,K] = m({ω_j})
    omegas: torch.Tensor,   # [B]   = m(Ω)
    gamma: float = 0.8,
    nu: float = 1.0,
    acts: str = "singletons+pairs+Omega",  # which sets to allow
    top_pairs_per_sample: int = 1          # if pairs, pick top-2 singletons per sample
):
    """
    Returns:
      chosen_sets: list of length B, each is a tuple(sorted class indices) or ('Omega',)
      E: [B, num_acts] tensor of expected utilities per candidate set in order returned
      act_list: list of acts corresponding to columns in E
    """
    B, K = beliefs.shape
    mOmega = omegas
    # Precompute g1 for sizes we may use
    g1_single = _owa_first_weight(1, gamma)           # =1.0
    g1_pair   = _owa_first_weight(2, gamma)           # e.g., ≈0.8 when gamma≈0.8
    g1_full   = _owa_first_weight(K, gamma)           # utility for Ω

    # Assemble candidate acts
    act_list = []
    # singletons
    if "singletons" in acts:
        act_list.extend([ (i,) for i in range(K) ])
    # pairs (global: all pairs) — large; or per-sample top-2 below
    all_pairs = []
    if "pairs" in acts and top_pairs_per_sample is None:
        for i in range(K):
            for j in range(i+1, K):
                all_pairs.append((i,j))
        act_list.extend(all_pairs)
    # Omega
    if "Omega" in acts:
        act_list.append(("Omega",))

    # Compute EU for all acts that don't depend on sample-specific selection
    E_cols = []
    # Singletons: EU = g1(1,γ) * (m_j + (1-ν) mΩ) = (m_j + (1-ν) mΩ)
    if "singletons" in acts:
        base = (1.0 - nu) * mOmega[:, None] + 0.0  # [B,1]
        E_single = g1_single * (beliefs + base)    # [B,K]
        E_cols.append(E_single)

    # Global all-pairs
    if "pairs" in acts and top_pairs_per_sample is None:
        # sum over j in A is just beliefs[:,i]+beliefs[:,j]
        E_pairs = []
        base = (1.0 - nu) * mOmega  # [B]
        for (i,j) in all_pairs:
            e = g1_pair * (beliefs[:, i] + beliefs[:, j] + base)
            E_pairs.append(e[:, None])
        if E_pairs:
            E_cols.append(torch.cat(E_pairs, dim=1))

    # Omega: constant across samples
    if "Omega" in acts:
        E_Omega = torch.full((B,1), g1_full, dtype=beliefs.dtype, device=beliefs.device)
        E_cols.append(E_Omega)

    if top_pairs_per_sample is None:
        # Stack all columns now
        E = torch.cat(E_cols, dim=1)
        best_idx = torch.argmax(E, dim=1)
        # Map argmax to acts
        chosen_sets = []
        for b in range(B):
            idx = best_idx[b].item()
            chosen_sets.append(act_list[idx])
        return chosen_sets, E, act_list

    # Per-sample top-pair option (lighter): build pairs per sample from top-2 belief classes
    # and evaluate only those pairs.
    # Build dynamic act_list per sample, but also produce E with consistent ordering:
    base_cols = 0
    if "singletons" in acts:
        base_cols += K
    if "Omega" in acts:
        base_cols += 1

    # Pre-allocate E for singletons + (top_pairs_per_sample) + Omega
    num_acts = (K if "singletons" in acts else 0) + (top_pairs_per_sample if "pairs" in acts else 0) + (1 if "Omega" in acts else 0)
    E = torch.empty((B, num_acts), dtype=beliefs.dtype, device=beliefs.device)
    chosen_pair_list = []

    col = 0
    if "singletons" in acts:
        base = (1.0 - nu) * mOmega[:, None]
        E[:, :K] = g1_single * (beliefs + base)
        col += K

    if "pairs" in acts:
        base = (1.0 - nu) * mOmega
        topk = torch.topk(beliefs, k=min(2*top_pairs_per_sample, K), dim=1).indices  # rough
        # form up to 'top_pairs_per_sample' disjoint pairs greedily
        for b in range(B):
            idxs = topk[b].tolist()
            pairs_b = []
            used = set()
            for t in idxs:
                if t in used: 
                    continue
                # find a partner not used yet
                partner = None
                for u in idxs:
                    if u != t and u not in used:
                        partner = u; break
                if partner is None:
                    continue
                i, j = sorted([t, partner])
                if (i,j) not in pairs_b:
                    pairs_b.append((i,j))
                    used.add(i); used.add(j)
                if len(pairs_b) == top_pairs_per_sample:
                    break
            chosen_pair_list.append(pairs_b)

        # fill pair columns
        for k_pair in range(top_pairs_per_sample):
            # default if not enough pairs: very small EU
            e_col = torch.full((B,), -1e9, dtype=beliefs.dtype, device=beliefs.device)
            for b, pairs_b in enumerate(chosen_pair_list):
                if k_pair < len(pairs_b):
                    i, j = pairs_b[k_pair]
                    e_col[b] = g1_pair * (beliefs[b, i] + beliefs[b, j] + base[b])
            E[:, col + k_pair] = e_col
        col += top_pairs_per_sample

    if "Omega" in acts:
        E[:, col] = g1_full

    # Choose best act per sample
    best_idx = torch.argmax(E, dim=1)

    # Build human-readable chosen sets
    chosen_sets = []
    for b in range(B):
        idx = best_idx[b].item()
        if "singletons" in acts and idx < K:
            chosen_sets.append( (idx,) )
        else:
            j = idx - (K if "singletons" in acts else 0)
            if "pairs" in acts and 0 <= j < top_pairs_per_sample and j < len(chosen_pair_list[b]):
                chosen_sets.append( tuple(sorted(chosen_pair_list[b][j])) )
            else:
                chosen_sets.append( ("Omega",) )
    # Return acts used (for column meaning)
    act_list = ["(i,)" for _ in range(K)] if "singletons" in acts else []
    if "pairs" in acts:
        act_list += ["(pair_k)" for _ in range(top_pairs_per_sample)]
    if "Omega" in acts:
        act_list += ["Omega"]
    return chosen_sets, E, act_list
