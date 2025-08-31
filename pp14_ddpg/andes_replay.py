
"""
Replay a surrogate trace in ANDES and compute cross-metrics.

Usage:
    python -m pp14_ddpg.andes_replay traces/trace_NN_LAA_RAL1.csv --andes-case path\to\ieee14.xlsx --out andes_LAA
"""
import argparse, os, numpy as np, pandas as pd, matplotlib.pyplot as plt
from .utils import compute_metrics, save_metrics_csv

def read_trace(trace_csv: str):
    df = pd.read_csv(trace_csv)
    t = df["t"].to_numpy()
    df_true = df["df_true_hz"].to_numpy()
    vmean = df["Vmean_pu"].to_numpy()
    actions = df[["a_dP1_mw","a_dP2_mw","a_dP3_mw","a_dQ1_mvar","a_dQ2_mvar","a_dQ3_mvar","a_ess_mw"]].to_numpy()
    laa = df["laa_frac"].to_numpy() if "laa_frac" in df else np.zeros_like(t)
    return t, df_true, vmean, actions, laa

def apply_timeseries_to_andes(sys, t, laa, actions):
    """
    TEMPLATE mapping: adapt to your ANDES model (device names/indices). Typical steps:
      1) Identify a Load 'PL' you can scale by (1 + laa[t]).
      2) Create play-in references for DERs/ESS (governor or P injection).
    """
    print("[INFO] Adapt 'apply_timeseries_to_andes()' to your case.")
    return False

def replay_with_andes(trace_csv: str, andes_case: str, out_prefix: str = "andes_replay"):
    try:
        from andes.core.system import System
    except Exception as e:
        print("[ERROR] ANDES import failed:", e); return False

    if not os.path.exists(andes_case):
        print(f"[ERROR] ANDES case not found: {andes_case}"); return False

    t, df_true_surr, vmean_surr, actions, laa = read_trace(trace_csv)
    dt = float(np.mean(np.diff(t))) if len(t) > 1 else 0.1
    sys = System(andes_case)

    ok = apply_timeseries_to_andes(sys, t, laa, actions)
    if ok is False:
        print("[WARN] Mapping not set. Computing surrogate metrics only, saving placeholders.")
        post_idx = int(np.where(laa>0)[0][-1])+1 if np.any(laa>0) else len(t)//2
        met_s = compute_metrics(df_true_surr, vmean_surr, dt, post_idx=post_idx)
        save_metrics_csv(out_prefix + "_metrics.csv",
                         [["surrogate", met_s["nadir_hz"], met_s["rocof_hz_per_s"], met_s["recovery_time_s"], met_s["volt_viol_pct"],
                           "andes", np.nan, np.nan, np.nan, np.nan]],
                         ["model","nadir_hz","rocof_hz_per_s","recovery_time_s","volt_viol_pct",
                          "model2","nadir_hz2","rocof_hz_per_s2","recovery_time_s2","volt_viol_pct2"])
        plt.figure(); plt.plot(t, df_true_surr, label="Surrogate Δf")
        plt.xlabel("Time [s]"); plt.ylabel("Δf [Hz]"); plt.title("Surrogate from Trace")
        plt.grid(True, alpha=0.3); plt.tight_layout(); plt.savefig(out_prefix+"_surrogate_df.png", dpi=150); plt.close()
        return False

    try:
        sys.run()
    except Exception as e:
        print("[ERROR] ANDES run failed:", e); return False

    # Extract arrays from your case (replace with actual variable access):
    df_andes = df_true_surr.copy()
    vmean_andes = vmean_surr.copy()

    post_idx = int(np.where(laa>0)[0][-1])+1 if np.any(laa>0) else len(t)//2
    met_s = compute_metrics(df_true_surr, vmean_surr, dt, post_idx=post_idx)
    met_a = compute_metrics(df_andes, vmean_andes, dt, post_idx=post_idx)
    save_metrics_csv(out_prefix + "_metrics.csv",
                     [["surrogate", met_s["nadir_hz"], met_s["rocof_hz_per_s"], met_s["recovery_time_s"], met_s["volt_viol_pct"],
                       "andes", met_a["nadir_hz"], met_a["rocof_hz_per_s"], met_a["recovery_time_s"], met_a["volt_viol_pct"]]],
                     ["model","nadir_hz","rocof_hz_per_s","recovery_time_s","volt_viol_pct",
                      "model2","nadir_hz2","rocof_hz_per_s2","recovery_time_s2","volt_viol_pct2"])

    plt.figure()
    plt.plot(t, df_true_surr, label="Surrogate Δf")
    plt.plot(t, df_andes, label="ANDES Δf", alpha=0.7)
    plt.xlabel("Time [s]"); plt.ylabel("Δf [Hz]"); plt.title("Δf Surrogate vs ANDES")
    plt.grid(True, alpha=0.3); plt.legend(); plt.tight_layout()
    plt.savefig(out_prefix+"_df_compare.png", dpi=150); plt.close()
    return True

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("trace", type=str)
    ap.add_argument("--andes-case", type=str, default="")
    ap.add_argument("--out", type=str, default="andes_replay")
    args = ap.parse_args()
    if not args.andes_case:
        print("[INFO] No --andes-case provided; will compute surrogate-only metrics/plots.")
    replay_with_andes(args.trace, args.andes_case, out_prefix=args.out)
