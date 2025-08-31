
"""
Sanity tests for PP14 env (IEEE-14 + swing) after persistent operating-load fix.
Run: python tests_pp14.py
"""
import numpy as np
from pp14_ddpg.config import Config
from pp14_ddpg.env import PP14Env

def run_steps(env, steps, laa=False, fdia=False, line=False):
    cfg = env.cfg; env.reset(seed=42); df=[]; solv=[]
    for t in range(steps):
        laa_frac = (cfg.LAA_FRAC if (laa and cfg.LAA_WIN[0] <= t < cfg.LAA_WIN[1]) else 0.0)
        fdia_bias = (cfg.FDIA_BIAS_HZ if (fdia and cfg.FDIA_WIN[0] <= t < cfg.FDIA_WIN[1]) else 0.0)
        line_fault = (line and cfg.LINE_WIN[0] <= t < cfg.LINE_WIN[1])
        step = env.step(np.zeros(7, np.float32), laa_frac, fdia_bias, line_fault)
        df.append(step.obs["df_true"]); solv.append(step.obs["solvable"])
    return np.array(df), np.array(solv)

def test_noattack_stable_frequency():
    cfg = Config(); env = PP14Env(cfg, ral_on=True); df, solv = run_steps(env, steps=200)
    assert np.mean(solv) > 0.95, "PF solvability too low"
    assert np.max(np.abs(df)) < 0.08, f"Δf drifted too much: max|df|={np.max(np.abs(df)):.3f} Hz"

def test_fdia_ids_triggers():
    cfg = Config(); env = PP14Env(cfg, ral_on=True); env.reset(seed=1); ids_seen = 0
    for t in range(cfg.EVAL_STEPS):
        fdia_bias = (cfg.FDIA_BIAS_HZ if (cfg.FDIA_WIN[0] <= t < cfg.FDIA_WIN[1]) else 0.0)
        step = env.step(np.zeros(7, np.float32), 0.0, fdia_bias, False); ids_seen += step.obs["ids"]
    assert ids_seen > 0, "IDS never triggered under FDIA window"

def test_laa_open_loop_drop_and_recover():
    cfg = Config(); env = PP14Env(cfg, ral_on=False); df,_ = run_steps(env, steps=cfg.EVAL_STEPS, laa=True)
    assert np.min(df) < -0.05, "LAA did not cause a noticeable frequency drop"
    post = cfg.LAA_WIN[1]
    assert np.mean(df[post:post+20]) > np.mean(df[post-20:post]), "No recovery trend after LAA"

def test_action_rate_limit():
    cfg = Config(); env = PP14Env(cfg, ral_on=True); env.reset(seed=7)
    a_big = np.array([10,10,10, 0,0,0, 10], dtype=np.float32)
    step = env.step(a_big, 0.0, 0.0, False)
    from_pp = env.net.sgen.p_mw.values[:3].copy(); from_ess = float(env.net.storage.p_mw.sum())
    assert np.all(np.abs(from_pp) <= cfg.RATE_LIMIT_MW + 1e-6), "DER rate limit failed"
    assert abs(from_ess) <= cfg.RATE_LIMIT_MW + 1e-6, "ESS rate limit failed"

if __name__ == "__main__":
    ok=0; total=4
    try: test_noattack_stable_frequency(); print("[PASS] noattack_stable_frequency"); ok+=1
    except AssertionError as e: print("[FAIL] noattack_stable_frequency:", e)
    try: test_fdia_ids_triggers(); print("[PASS] fdia_ids_triggers"); ok+=1
    except AssertionError as e: print("[FAIL] fdia_ids_triggers:", e)
    try: test_laa_open_loop_drop_and_recover(); print("[PASS] laa_open_loop_drop_and_recover"); ok+=1
    except AssertionError as e: print("[FAIL] laa_open_loop_drop_and_recover:", e)
    try: test_action_rate_limit(); print("[PASS] action_rate_limit"); ok+=1
    except AssertionError as e: print("[FAIL] action_rate_limit:", e)
    print(f"{ok}/{total} tests passed")
