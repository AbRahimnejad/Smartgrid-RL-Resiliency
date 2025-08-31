"""Training & evaluation with metrics, traces, and Monte Carlo sweeps."""
import os, numpy as np, torch, matplotlib.pyplot as plt
from tqdm import trange
from .config import Config
from .env import PP14Env
from .agent import DDPGAgent
from .utils import (
    obs_to_vec_full, action_bounds, resiliency_score,
    compute_metrics, save_metrics_csv
)
from .export import EpisodeRecorder

# ---------- helpers ----------
def _safe_mean_voltage(V):
    V = np.asarray(V, dtype=np.float32)
    if V.size == 0 or np.all(~np.isfinite(V)):
        return 1.0
    return float(np.nanmean(np.where(np.isfinite(V), V, 1.0)))

# ---------- training ----------
def train_ddpg(cfg: Config, ral_on: bool = True, seed: int = 0):
    env = PP14Env(cfg, ral_on=ral_on)
    o = env.reset(seed=seed)
    s0 = obs_to_vec_full(env, o)
    lo, hi = action_bounds(cfg)
    agent = DDPGAgent(obs_dim=s0.shape[0], act_dim=len(lo), act_lo=lo, act_hi=hi, cfg=cfg)

    losses, eval_returns = [], []
    step_count = 0
    last_action = np.zeros(len(lo), dtype=np.float32)

    for epoch in trange(cfg.EPOCHS, desc="Epochs", unit="epoch"):
        o = env.reset(seed=epoch)
        s = obs_to_vec_full(env, o)
        agent.noise_scale = 0.25                 # modest exploration
        last_action[:] = 0.0

        # curriculum for LAA: ramp 2% -> target across epochs
        laa_curr = float(np.interp(epoch, [0, max(1, cfg.EPOCHS-1)], [0.02, cfg.LAA_FRAC]))

        for t in trange(cfg.STEPS_PER_EP, desc=f"Ep{epoch}", leave=False, unit="step"):
            laa  = laa_curr         if (cfg.LAA_WIN[0]  <= t < cfg.LAA_WIN[1]  and np.random.rand()<0.5) else 0.0
            fdia = cfg.FDIA_BIAS_HZ if (cfg.FDIA_WIN[0] <= t < cfg.FDIA_WIN[1] and np.random.rand()<0.5) else 0.0
            line = (cfg.LINE_WIN[0] <= t < cfg.LINE_WIN[1]) and (np.random.rand()<0.5)

            a   = agent.select_action(s, noise_scale=1.0, eval_mode=False)
            step = env.step(a, laa, fdia, line)  # reward uses TRUE/MEAS mix internally
            sp  = obs_to_vec_full(env, step.obs)

            # small action magnitude + smoothness penalties (scaled)
            act_pen = 1e-3 * float(np.sum(a**2)) + 1e-3 * float(np.sum((a - last_action)**2))
            r = float(step.reward - act_pen)
            last_action = a

            agent.buffer.push(s, a, r, sp, False)
            s = sp
            step_count += 1

            if step_count >= cfg.WARMUP_STEPS:
                for _ in range(cfg.UPDATES_PER_STEP):
                    loss = agent.update()
                    if loss is not None:
                        losses.append(loss)

        # short sanity eval (no-attack)
        env_eval = PP14Env(cfg, ral_on=ral_on)
        obs_e = env_eval.reset(seed=100 + epoch)
        s = obs_to_vec_full(env_eval, obs_e)
        ret = 0.0
        for _ in trange(200, desc=f"Eval Ep{epoch}", leave=False, unit="step"):
            a = agent.select_action(s, noise_scale=0.0, eval_mode=True)
            step = env_eval.step(a, 0.0, 0.0, False)
            s = obs_to_vec_full(env_eval, step.obs)
            ret += step.reward
        eval_returns.append(ret)

    # Plot training curves
    plt.figure()
    if len(losses) > 0:
        L = np.array(losses, dtype=np.float64)
        k = max(50, min(500, len(L)//20))
        Ls = np.convolve(L, np.ones(k)/k, mode="valid") if k > 1 else L
        plt.plot(Ls, label="Critic loss (smoothed)")
    plt.plot(eval_returns, label="Eval return (per epoch)")
    plt.xlabel("Epoch / updates")
    plt.title("DDPG Training (RMSprop, RAL ON)")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig("pp14_ddpg_training.png", dpi=150)
    plt.close()

    torch.save(agent.actor.state_dict(), "pp14_ddpg_actor.pt")
    torch.save(agent.critic.state_dict(), "pp14_ddpg_critic.pt")
    return agent

# ---------- baselines & eval ----------
def kp_baseline_episode(cfg: Config, Kp: float, attack: str, steps: int, seed: int = 0):
    """Kp baseline that uses MEASURED df (includes FDIA), reward uses TRUE/MEAS mix inside env."""
    env = PP14Env(cfg, ral_on=True)
    env.reset(seed=seed)
    df_series, v_series, total = [], [], 0.0
    for t in range(steps):
        laa  = cfg.LAA_FRAC     if (attack == "LAA"  and cfg.LAA_WIN[0]  <= t < cfg.LAA_WIN[1]) else 0.0
        fdia = cfg.FDIA_BIAS_HZ if (attack == "FDIA" and cfg.FDIA_WIN[0] <= t < cfg.FDIA_WIN[1]) else 0.0
        line = (attack == "LINE") and (cfg.LINE_WIN[0] <= t < cfg.LINE_WIN[1])

        measured_df = env.get_measured_df(fdia)     # consistent measurement
        u = float(np.clip(-Kp * measured_df, cfg.DER_P_MIN, cfg.DER_P_MAX))
        delta = np.array([u, 0.0, 0.0,  0.0, 0.0, 0.0,  u], dtype=np.float32)

        step = env.step(delta, laa, fdia, line)
        df_series.append(step.obs["df_true"])
        v_series.append(_safe_mean_voltage(step.obs["V"]))
        total += step.reward

    return np.array(df_series), np.array(v_series), total

def eval_nn_vs_kp(cfg: Config, agent, Kp: float = 0.7, export_dir: str = "traces"):
    os.makedirs(export_dir, exist_ok=True)
    rows = []

    for att in ["LINE", "LAA", "FDIA"]:
        # --- NN
        env = PP14Env(cfg, ral_on=True)
        obs = env.reset(seed=123)
        s = obs_to_vec_full(env, obs)
        df_series, v_series = [], []
        rec_nn = EpisodeRecorder(dt=cfg.DT, controller="NN", scenario=att, ral_on=1)

        for t in trange(cfg.EVAL_STEPS, desc=f"Eval-{att}-NN", leave=False, unit="step"):
            laa  = cfg.LAA_FRAC     if (att == "LAA"  and cfg.LAA_WIN[0]  <= t < cfg.LAA_WIN[1]) else 0.0
            fdia = cfg.FDIA_BIAS_HZ if (att == "FDIA" and cfg.FDIA_WIN[0] <= t < cfg.FDIA_WIN[1]) else 0.0
            line = (att == "LINE") and (cfg.LINE_WIN[0] <= t < cfg.LINE_WIN[1])

            a = agent.select_action(s, noise_scale=0.0, eval_mode=True)
            step = env.step(a, laa, fdia, line)
            s = obs_to_vec_full(env, step.obs)

            df_series.append(step.obs["df_true"])
            v_series.append(_safe_mean_voltage(step.obs["V"]))
            rec_nn.log_step(t, step.obs, a, env, laa, fdia, line)

        path_nn = rec_nn.save_csv(export_dir)

        # --- Kp
        env2 = PP14Env(cfg, ral_on=True)
        env2.reset(seed=456)
        df_kp, v_kp = [], []
        rec_kp = EpisodeRecorder(dt=cfg.DT, controller="KP", scenario=att, ral_on=1)
        for t in trange(cfg.EVAL_STEPS, desc=f"Eval-{att}-KP", leave=False, unit="step"):
            laa  = cfg.LAA_FRAC     if (att == "LAA"  and cfg.LAA_WIN[0]  <= t < cfg.LAA_WIN[1]) else 0.0
            fdia = cfg.FDIA_BIAS_HZ if (att == "FDIA" and cfg.FDIA_WIN[0] <= t < cfg.FDIA_WIN[1]) else 0.0
            line = (att == "LINE") and (cfg.LINE_WIN[0] <= t < cfg.LINE_WIN[1])

            measured_df = env2.get_measured_df(fdia)
            u = float(np.clip(-Kp * measured_df, cfg.DER_P_MIN, cfg.DER_P_MAX))
            delta = np.array([u, 0.0, 0.0,  0.0, 0.0, 0.0,  u], dtype=np.float32)

            step = env2.step(delta, laa, fdia, line)
            df_kp.append(step.obs["df_true"])
            v_kp.append(_safe_mean_voltage(step.obs["V"]))
            rec_kp.log_step(t, step.obs, delta, env2, laa, fdia, line)

        path_kp = rec_kp.save_csv(export_dir)

        df_series = np.asarray(df_series); v_series = np.asarray(v_series)
        df_kp     = np.asarray(df_kp);     v_kp     = np.asarray(v_kp)

        if att == "LAA":
            mask = (np.arange(cfg.EVAL_STEPS) >= cfg.LAA_WIN[0]) & (np.arange(cfg.EVAL_STEPS) < cfg.LAA_WIN[1]); post = cfg.LAA_WIN[1]
        elif att == "FDIA":
            mask = (np.arange(cfg.EVAL_STEPS) >= cfg.FDIA_WIN[0]) & (np.arange(cfg.EVAL_STEPS) < cfg.FDIA_WIN[1]); post = cfg.FDIA_WIN[1]
        else:
            mask = (np.arange(cfg.EVAL_STEPS) >= cfg.LINE_WIN[0]) & (np.arange(cfg.EVAL_STEPS) < cfg.LINE_WIN[1]); post = cfg.LINE_WIN[1]

        score_nn = resiliency_score(df_series, v_series, mask, post)
        score_kp = resiliency_score(df_kp, v_kp, mask, post)

        met_nn = compute_metrics(df_series, v_series, cfg.DT, post_idx=post)
        met_kp = compute_metrics(df_kp, v_kp, cfg.DT, post_idx=post)

        t_axis = np.arange(cfg.EVAL_STEPS) * cfg.DT
        plt.figure()
        plt.plot(t_axis, df_series, label="NN")
        plt.plot(t_axis, df_kp, label="Kp")
        plt.xlabel("Time [s]"); plt.ylabel("Δf [Hz]")
        plt.title(f"{att} - RAL ON (surrogate)")
        plt.grid(True, alpha=0.3); plt.legend(); plt.tight_layout()
        plt.savefig(f"pp14_DDPG_{att}_df.png", dpi=150); plt.close()

        rows.append((att,"NN",1,score_nn, met_nn["nadir_hz"], met_nn["rocof_hz_per_s"],
                     met_nn["recovery_time_s"], met_nn["volt_viol_pct"], os.path.basename(path_nn)))
        rows.append((att,"KP",1,score_kp, met_kp["nadir_hz"], met_kp["rocof_hz_per_s"],
                     met_kp["recovery_time_s"], met_kp["volt_viol_pct"], os.path.basename(path_kp)))

    save_metrics_csv("pp14_surrogate_metrics.csv", rows,
                     ["scenario","controller","RAL","resiliency_score",
                      "nadir_hz","rocof_hz_per_s","recovery_time_s","volt_viol_pct","trace_file"])

def monte_carlo_eval(cfg: Config, agent, Kp: float, seeds=(101,102,103,104,105), op_scales=(0.95,1.0,1.05)):
    import pandas as pd
    records = []
    for s in seeds:
        for scale in op_scales:
            for att in ["LINE","LAA","FDIA"]:
                # NN
                env = PP14Env(cfg, ral_on=True)
                obs = env.reset(seed=s)
                env.oper_load = env.base_load * scale
                env.net.load.p_mw = env.oper_load.copy()
                df_series, v_series = [], []
                state = obs_to_vec_full(env, obs)
                for t in range(cfg.EVAL_STEPS):
                    laa  = cfg.LAA_FRAC     if (att == "LAA"  and cfg.LAA_WIN[0]  <= t < cfg.LAA_WIN[1]) else 0.0
                    fdia = cfg.FDIA_BIAS_HZ if (att == "FDIA" and cfg.FDIA_WIN[0] <= t < cfg.FDIA_WIN[1]) else 0.0
                    line = (att == "LINE") and (cfg.LINE_WIN[0] <= t < cfg.LINE_WIN[1])
                    a = agent.select_action(state, noise_scale=0.0, eval_mode=True)
                    step = env.step(a, laa, fdia, line)
                    state = obs_to_vec_full(env, step.obs)
                    df_series.append(step.obs["df_true"])
                    v_series.append(_safe_mean_voltage(step.obs["V"]))
                df_series = np.asarray(df_series); v_series = np.asarray(v_series)
                post = {"LAA":cfg.LAA_WIN[1], "FDIA":cfg.FDIA_WIN[1], "LINE":cfg.LINE_WIN[1]}[att]
                met_nn = compute_metrics(df_series, v_series, cfg.DT, post_idx=post)
                records.append(dict(seed=s, op_scale=scale, scenario=att, controller="NN", **met_nn))

                # KP
                env = PP14Env(cfg, ral_on=True)
                obs = env.reset(seed=s)
                env.oper_load = env.base_load * scale
                env.net.load.p_mw = env.oper_load.copy()
                df_series, v_series = [], []
                for t in range(cfg.EVAL_STEPS):
                    laa  = cfg.LAA_FRAC     if (att == "LAA"  and cfg.LAA_WIN[0]  <= t < cfg.LAA_WIN[1]) else 0.0
                    fdia = cfg.FDIA_BIAS_HZ if (att == "FDIA" and cfg.FDIA_WIN[0] <= t < cfg.FDIA_WIN[1]) else 0.0
                    line = (att == "LINE") and (cfg.LINE_WIN[0] <= t < cfg.LINE_WIN[1])
                    measured_df = env.get_measured_df(fdia)
                    u = float(np.clip(-Kp * measured_df, cfg.DER_P_MIN, cfg.DER_P_MAX))
                    delta = np.array([u,0,0, 0,0,0, u], dtype=np.float32)
                    step = env.step(delta, laa, fdia, line)
                    df_series.append(step.obs["df_true"])
                    v_series.append(_safe_mean_voltage(step.obs["V"]))
                df_series = np.asarray(df_series); v_series = np.asarray(v_series)
                met_kp = compute_metrics(df_series, v_series, cfg.DT, post_idx=post)
                records.append(dict(seed=s, op_scale=scale, scenario=att, controller="KP", **met_kp))

    df = pd.DataFrame.from_records(records)
    df.to_csv("pp14_mc_metrics_surrogate.csv", index=False, encoding="utf-8-sig")

    # quick boxplots for nadir
    for att in ["LINE","LAA","FDIA"]:
        sub = df[df["scenario"]==att]
        data = [sub[sub["controller"]=="NN"]["nadir_hz"], sub[sub["controller"]=="KP"]["nadir_hz"]]
        plt.figure()
        plt.boxplot(data, labels=["NN","Kp"])
        plt.title(f"MC Nadir — {att} (surrogate)")
        plt.ylabel("Nadir [Hz]")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(f"pp14_MC_nadir_{att}.png", dpi=150)
        plt.close()

def main():
    cfg = Config()
    agent = train_ddpg(cfg, ral_on=True, seed=0)

    # coarse Kp tune against average return
    def avg_return_for_Kp(K):
        atts = ["LAA", "LINE", "FDIA"]; seeds = [11, 12]
        R = []
        for s in seeds:
            for a in atts:
                R.append(kp_baseline_episode(cfg, K, a, steps=cfg.STEPS_PER_EP, seed=s)[2])
        return sum(R) / len(R)

    Kp = 0.7
    for _ in range(6):
        Rp = avg_return_for_Kp(Kp + 0.05)
        Rm = avg_return_for_Kp(Kp - 0.05)
        grad = - (Rp - Rm) / (2 * 0.05)
        Kp -= 0.05 * grad
        Kp = float(np.clip(Kp, 0.2, 3.0))

    with open("pp14_final_Kp_from_ddpg_run.txt", "w", encoding="utf-8") as f:
        f.write(f"Kp_star ~ {Kp:.3f} MW/Hz\n")

    eval_nn_vs_kp(cfg, agent, Kp=Kp, export_dir="traces")
    monte_carlo_eval(cfg, agent, Kp=Kp)

    print(
        "Artifacts written:\n"
        "- pp14_ddpg_training.png\n"
        "- pp14_ddpg_actor.pt / pp14_ddpg_critic.pt\n"
        "- pp14_surrogate_metrics.csv\n"
        "- pp14_mc_metrics_surrogate.csv\n"
        "- traces/trace_[NN|KP]_[LINE|LAA|FDIA]_RAL1.csv\n"
        "- pp14_DDPG_[LINE|LAA|FDIA]_df.png\n"
        "- pp14_MC_nadir_[scenario].png\n"
        "- pp14_final_Kp_from_ddpg_run.txt"
    )

if __name__ == "__main__":
    main()
