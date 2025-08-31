# Configuration constants and hyperparameters.
from dataclasses import dataclass

@dataclass
class Config:
    # -------- Simulation --------
    DT: float = 0.1              # s
    EPOCHS: int = 20
    STEPS_PER_EP: int = 600      # 60 s
    EVAL_STEPS: int = 400        # 40 s

    # -------- System (swing proxy) --------
    F_NOM_HZ: float = 50.0
    H_SYS: float = 5.0           # inertia-like constant
    D_SYS: float = 1.5           # damping term
    # Calibration knob: scales how much ΔP_slack (MW) affects d f / dt
    SWING_GAIN: float = 0.0025   # << smaller = “softer” system, avoids hitting ±2 Hz clamp
    DF_CLAMP_HZ: float = 2.0

    # -------- DER/ESS limits & rate limits --------
    DER_P_MIN: float = -1.0
    DER_P_MAX: float =  1.0
    DER_Q_MIN: float = -0.5
    DER_Q_MAX: float =  0.5
    ESS_P_MIN: float = -1.0
    ESS_P_MAX: float =  1.0
    SOC_MIN: float = 0.10
    SOC_MAX: float = 0.90
    RATE_LIMIT_MW: float = 0.10  # per time step

    # -------- Disturbances --------
    LAA_FRAC: float = 0.10
    LAA_WIN:  tuple = (60, 120)   # steps (6–12 s)
    FDIA_BIAS_HZ: float = -0.05   # −50 mHz bias on the measurement
    FDIA_WIN: tuple = (80, 120)
    LINE_WIN: tuple = (70, 120)

    # -------- Reward weights (rebalanced) --------
    # We primarily penalize TRUE Δf; include a small term on MEASURED Δf
    ALPHA_F_TRUE: float = 3.0
    ALPHA_F_MEAS: float = 1.0
    # Voltage penalty as hinge beyond ±5% (handled in env)
    ALPHA_V: float = 0.3
    ALPHA_IDS: float = 0.5
    RAL_W: float = 2.0
    REWARD_SCALE: float = 5.0     # divide final cost by this (smaller → stronger signal)

    # -------- DDPG (paper-style RMSprop) --------
    GAMMA: float = 0.99
    ACTOR_LR: float = 1e-4
    CRITIC_LR: float = 1e-4
    RMS_ALPHA: float = 0.95
    RMS_EPS: float = 1e-8
    TAU: float = 1e-3
    BATCH_SIZE: int = 128
    BUF_SIZE: int = 100_000
    WARMUP_STEPS: int = 2_000
    UPDATES_PER_STEP: int = 2
