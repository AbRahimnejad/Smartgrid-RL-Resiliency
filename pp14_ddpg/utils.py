import numpy as np
import csv


def action_bounds(cfg=None):
    # [dP1,dP2,dP3, dQ1,dQ2,dQ3, dPess]
    lo = np.array([-0.1, -0.1, -0.1,  -0.1, -0.1, -0.1,  -0.1], dtype=np.float32)
    hi = -lo
    if cfg is not None:
        # respect absolute plant limits too; step-size limited by RATE_LIMIT_MW
        lo[:3] = -cfg.RATE_LIMIT_MW
        hi[:3] =  cfg.RATE_LIMIT_MW
        lo[3:6] = -min(cfg.RATE_LIMIT_MW, abs(cfg.DER_Q_MIN))
        hi[3:6] =  min(cfg.RATE_LIMIT_MW, abs(cfg.DER_Q_MAX))
        lo[6]    = -cfg.RATE_LIMIT_MW
        hi[6]    =  cfg.RATE_LIMIT_MW
    return lo, hi

# ---- Observation vectorization (normalized) ----
def obs_to_vec_full(env, obs: dict):
    """Return a normalized feature vector for the policy/critic.

    IMPORTANT: use MEASURED frequency (robust against FDIA).
    """
    cfg = env.cfg
    df = float(obs.get("df_meas", obs.get("df_true", 0.0)))
    v  = np.asarray(obs["V"], dtype=np.float32)
    soc = float(obs["soc"])
    derp = np.asarray(obs["der_p"], dtype=np.float32)
    derq = np.asarray(obs["der_q"], dtype=np.float32)

    # scales
    DF_S = max(1e-6, cfg.DF_CLAMP_HZ)       # Hz -> ~[-1,1]
    V_S  = 0.05                              # 5% voltage dev ≈ 1.0
    P_S  = max(1e-6, max(abs(cfg.DER_P_MAX), abs(cfg.DER_P_MIN)))
    Q_S  = max(1e-6, max(abs(cfg.DER_Q_MAX), abs(cfg.DER_Q_MIN)))

    # build (keep length modest)
    vdev = np.clip((v - 1.0) / V_S, -3.0, 3.0)
    x = [
        np.clip(df / DF_S, -1.0, 1.0),
        np.clip((soc - 0.5) / 0.4, -1.0, 1.0),
        *np.clip(derp / P_S, -1.0, 1.0),
        *np.clip(derq / Q_S, -1.0, 1.0),
        np.mean(vdev), np.min(vdev), np.max(vdev),
    ]
    return np.asarray(x, dtype=np.float32)

# ---- Metrics ----
def resiliency_score(df_series, v_series, attack_mask, post_idx):
    # untouched; keep your existing if you prefer
    df = np.asarray(df_series)
    v  = np.asarray(v_series)
    nadir = float(np.min(df[attack_mask])) if np.any(attack_mask) else float(np.min(df))
    volt_viols = float(np.mean(np.abs(v - 1.0) > 0.05))
    return - (abs(nadir) + volt_viols)

def compute_metrics(df_series, v_series, dt, post_idx):
    df = np.asarray(df_series); v = np.asarray(v_series)
    # Nadir, ROCOF, recovery (to |df|<0.05 Hz), voltage violations
    nadir = float(np.min(df))
    rocof = float(np.max(np.abs(np.diff(df)))/dt) if len(df) > 1 else 0.0
    thr = 0.05
    rec = None
    for k in range(post_idx, len(df)):
        if abs(df[k]) < thr:
            rec = (k - post_idx) * dt
            break
    if rec is None: rec = float('nan')
    volt_viol = float(np.mean(np.abs(v-1.0) > 0.05))
    return dict(nadir_hz=nadir, rocof_hz_per_s=rocof, recovery_time_s=rec, volt_viol_pct=100.0*volt_viol)



def save_metrics_csv(path, rows, header):
    """
    Write evaluation rows to CSV with a UTF-8 BOM so Excel opens it cleanly on Windows.
    `rows` is an iterable of tuples/lists matching `header`.
    """
    import csv
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(header)
        for r in rows:
            w.writerow(list(r))

