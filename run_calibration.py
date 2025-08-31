
"""
Calibration helper: surrogate step with Kp (ANDES hook is left to run_andes_replay.py).
"""
import argparse, numpy as np, matplotlib.pyplot as plt
from pp14_ddpg.config import Config
from pp14_ddpg.env import PP14Env
from pp14_ddpg.utils import compute_metrics, save_metrics_csv

def surrogate_step(cfg, Kp, step_frac=0.05, steps=400, seed=777):
    env = PP14Env(cfg, ral_on=True); env.reset(seed=seed)
    t_mid = steps//2
    df_series=[]; v_series=[]
    for t in range(steps):
        laa = step_frac if t>=t_mid else 0.0
        step_res = env.step(np.zeros(7, np.float32), laa, 0.0, False)
        measured_df = step_res.obs["df_true"]
        u = float(np.clip(-Kp * measured_df, cfg.DER_P_MIN, cfg.DER_P_MAX))
        env.step(np.array([u,0,0, 0,0,0, u], np.float32), laa, 0.0, False)
        df_series.append(env.df_hz); v_series.append(float(np.nanmean(step_res.obs["V"])) if step_res.obs["V"] is not None else 1.0)
    return np.array(df_series), np.array(v_series), t_mid

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kp", type=float, default=0.7)
    args = ap.parse_args()
    cfg = Config()
    df_s, v_s, t_mid = surrogate_step(cfg, args.kp, step_frac=0.05, steps=cfg.EVAL_STEPS, seed=321)
    t = np.arange(cfg.EVAL_STEPS)*cfg.DT
    plt.figure(); plt.plot(t, df_s, label="Surrogate")
    plt.axvline(t[t_mid], color="k", ls="--", lw=1, label="Step")
    plt.xlabel("Time [s]"); plt.ylabel("Δf [Hz]"); plt.title("Calibration: Surrogate step (Kp)")
    plt.grid(True, alpha=0.3); plt.legend(); plt.tight_layout(); plt.savefig("calib_surrogate_step_df.png", dpi=150); plt.close()
    met_s = compute_metrics(df_s, v_s, cfg.DT, post_idx=t_mid)
    save_metrics_csv("calibration_metrics.csv",
                     [["surrogate", args.kp, met_s["nadir_hz"], met_s["rocof_hz_per_s"], met_s["recovery_time_s"], met_s["volt_viol_pct"]]],
                     ["model","Kp","nadir_hz","rocof_hz_per_s","recovery_time_s","volt_viol_pct"])
    print("Calibration artifacts: calib_surrogate_step_df.png, calibration_metrics.csv")

if __name__ == "__main__":
    main()
