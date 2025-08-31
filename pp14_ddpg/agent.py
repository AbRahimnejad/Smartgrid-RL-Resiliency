import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import deque, namedtuple
from dataclasses import dataclass

@dataclass
class DDPGConfig:
    actor_lr: float = 1e-4
    critic_lr: float = 1e-4
    gamma: float = 0.98
    tau: float = 1e-3
    buffer_size: int = 100_000
    batch_size: int = 128
    warmup_steps: int = 2000
    updates_per_step: int = 1
    rms_alpha: float = 0.95
    rms_eps: float = 1e-8
    noise_sigma: float = 0.2   # exploration noise

class Replay:
    def __init__(self, capacity=100_000):
        self.buf = deque(maxlen=capacity)
        self.T = namedtuple("T", "s a r sp done")
    def push(self, s, a, r, sp, done):
        # sanitize NaNs/Infs
        s = np.nan_to_num(s, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
        sp = np.nan_to_num(sp, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
        a = np.nan_to_num(a, nan=0.0).astype(np.float32)
        r = float(np.nan_to_num(r, nan=0.0))
        self.buf.append(self.T(s,a,r,sp,done))
    def sample(self, batch):
        idx = np.random.choice(len(self.buf), size=batch, replace=False)
        batch = [self.buf[i] for i in idx]
        s  = torch.from_numpy(np.stack([b.s  for b in batch])).float()
        a  = torch.from_numpy(np.stack([b.a  for b in batch])).float()
        r  = torch.tensor([b.r for b in batch], dtype=torch.float32).unsqueeze(1)
        sp = torch.from_numpy(np.stack([b.sp for b in batch])).float()
        d  = torch.tensor([b.done for b in batch], dtype=torch.float32).unsqueeze(1)
        return s, a, r, sp, d
    def __len__(self): return len(self.buf)

def mlp(sizes, act=nn.ReLU, last_act=None):
    layers=[]
    for i in range(len(sizes)-1):
        layers += [nn.Linear(sizes[i], sizes[i+1])]
        if i < len(sizes)-2:
            layers += [act()]
        elif last_act is not None:
            layers += [last_act()]
    return nn.Sequential(*layers)

class Actor(nn.Module):
    def __init__(self, obs_dim, act_dim):
        super().__init__()
        self.net = mlp([obs_dim, 128, 128, act_dim], act=nn.ReLU, last_act=nn.Tanh)
    def forward(self, x):
        return self.net(x)  # in [-1,1]

class Critic(nn.Module):
    def __init__(self, obs_dim, act_dim):
        super().__init__()
        self.net = mlp([obs_dim + act_dim, 256, 256, 1], act=nn.ReLU)
    def forward(self, x, a):
        return self.net(torch.cat([x,a], dim=-1))

class DDPGAgent:
    def __init__(self, obs_dim, act_dim, act_lo, act_hi, cfg):
        self.obs_dim = obs_dim; self.act_dim = act_dim
        self.act_lo = torch.tensor(act_lo, dtype=torch.float32)
        self.act_hi = torch.tensor(act_hi, dtype=torch.float32)
        self.cfg = cfg

        self.actor = Actor(obs_dim, act_dim)
        self.critic = Critic(obs_dim, act_dim)
        self.actor_tgt = Actor(obs_dim, act_dim)
        self.critic_tgt = Critic(obs_dim, act_dim)
        self.actor_tgt.load_state_dict(self.actor.state_dict())
        self.critic_tgt.load_state_dict(self.critic.state_dict())

        self.pi_opt = torch.optim.RMSprop(self.actor.parameters(), lr=cfg.ACTOR_LR, alpha=cfg.RMS_ALPHA, eps=cfg.RMS_EPS)
        self.q_opt  = torch.optim.RMSprop(self.critic.parameters(), lr=cfg.CRITIC_LR, alpha=cfg.RMS_ALPHA, eps=cfg.RMS_EPS)

        self.buffer = Replay(capacity=cfg.BUF_SIZE)
        self.huber = nn.SmoothL1Loss(reduction="mean")

        self.noise_scale = 0.2

    # scale [-1,1] -> [lo,hi]
    def _scale_action(self, a01: torch.Tensor):
        return 0.5*(a01+1.0)*(self.act_hi - self.act_lo) + self.act_lo

    def select_action(self, s, noise_scale=0.0, eval_mode=False):
        x = torch.from_numpy(np.asarray(s, dtype=np.float32)).unsqueeze(0)
        with torch.no_grad():
            a01 = self.actor(x)                     # [-1,1]
            a = self._scale_action(a01).squeeze(0).numpy()
        if not eval_mode and noise_scale > 0.0:
            a = a + noise_scale * (self.act_hi.numpy() - self.act_lo.numpy()) * 0.1 * np.random.randn(self.act_dim).astype(np.float32)
        # final clamp
        a = np.clip(a, self.act_lo.numpy(), self.act_hi.numpy()).astype(np.float32)
        return a

    def update(self):
        if len(self.buffer) < self.cfg.BATCH_SIZE:
            return None
        s, a, r, sp, d = self.buffer.sample(self.cfg.BATCH_SIZE)

        with torch.no_grad():
            a_tp1 = self._scale_action(self.actor_tgt(sp))
            q_tp1 = self.critic_tgt(sp, a_tp1)
            y = torch.clip(r + self.cfg.GAMMA * (1.0 - d) * q_tp1, -100.0, 100.0)  # clamp targets

        # Critic update (Huber + grad clip)
        q = self.critic(s, a)
        q_loss = self.huber(q, y)
        self.q_opt.zero_grad(set_to_none=True)
        q_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 1.0)
        self.q_opt.step()

        # Actor update
        a_pi = self._scale_action(self.actor(s))
        pi_loss = - self.critic(s, a_pi).mean()
        self.pi_opt.zero_grad(set_to_none=True)
        pi_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 1.0)
        self.pi_opt.step()

        # Soft target
        with torch.no_grad():
            for p, pt in zip(self.actor.parameters(), self.actor_tgt.parameters()):
                pt.data.mul_(1.0 - self.cfg.TAU).add_(self.cfg.TAU * p.data)
            for p, pt in zip(self.critic.parameters(), self.critic_tgt.parameters()):
                pt.data.mul_(1.0 - self.cfg.TAU).add_(self.cfg.TAU * p.data)

        return float(q_loss.item())

