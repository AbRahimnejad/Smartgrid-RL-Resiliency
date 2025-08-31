"""PP14 environment: IEEE-14 + DER/ESS + swing-proxy with correct slack reference,
FDIA measurement model, IDS flag, and physically-consistent SoC integration.
"""
from __future__ import annotations
import numpy as np
import pandapower as pp
import pandapower.networks as pn


class StepResult:
    __slots__ = ("obs", "reward")
    def __init__(self, obs: dict, reward: float):
        self.obs = obs
        self.reward = float(reward)


class PP14Env:
    """
    Action (7-dim per step, rate-limited):
        a = [dP1, dP2, dP3,  dQ1, dQ2, dQ3,  dPess]  [MW, MVAr, MW]

    Observations dict (subset used by policy via obs_to_vec_full):
        - df_true : physical frequency deviation [Hz]
        - df_meas : measured frequency (df_true + FDIA bias) [Hz]
        - V       : bus voltages [p.u.] (np.ndarray)
        - soc     : storage state of charge [0..1]
        - der_p   : DER active injections [MW] length 3
        - der_q   : DER reactive injections [MVAr] length 3
        - ids     : 1.0 if FDIA is active this step, else 0.0
        - solvable: 1.0 if PF solved, else 0.0
    """
    def __init__(self, cfg, ral_on: bool = True):
        self.cfg = cfg
        self.ral_on = ral_on

        # Build IEEE-14 and ensure DER/ESS assets exist
        self.net = pn.case14()
        self._ensure_der_tables()

        # Base & operating loads
        self.base_load = self.net.load.p_mw.values.copy()
        self.oper_load = self.base_load.copy()   # set each reset()

        # Frequency states
        self.df_true = 0.0   # plant (unbiased)
        self.df_meas = 0.0   # measurement (biased by FDIA)
        self.df_prev = 0.0

        # DER/ESS internal states
        self.der_p = np.zeros(3, dtype=np.float32)
        self.der_q = np.zeros(3, dtype=np.float32)
        self.ess_p = 0.0
        self.soc   = 0.5

        # Reference slack active power (MW) at operating point
        self.Pslack_ref_mw = 0.0

        # Choose a reproducible line for the "LINE" event
        self.fault_line_idx = int(np.argmax(self.net.line.length_km.values))

    # --------------------- helpers ---------------------
    def _ensure_der_tables(self):
        """Create three sgens (DERs) and one storage if missing."""
        net = self.net
        target_buses = [9, 10, 11]  # arbitrary but fixed
        while len(net.sgen) < 3:
            bus = target_buses[len(net.sgen)]
            pp.create_sgen(net, bus=bus, p_mw=0.0, q_mvar=0.0, name=f"DER{len(net.sgen)+1}")
        if len(net.storage) == 0:
            # Default 2 MWh capacity; SoC managed in env using this capacity
            pp.create_storage(net, bus=9, p_mw=0.0, max_e_mwh=2.0, soc_percent=50.0, name="ESS")

    def _run_pf_safe(self):
        """Run PF; return (solvable, p_slack_MW, V_pu array)."""
        try:
            pp.runpp(self.net, calculate_voltage_angles=False, init="results")
            solvable = True
        except Exception:
            solvable = False
        # Slack active power (sum in case of multiple ext_grids)
        try:
            p_slack = float(self.net.res_ext_grid.p_mw.sum())
        except Exception:
            p_slack = 0.0
        # Voltages (safe, finite)
        try:
            V = self.net.res_bus.vm_pu.values.astype(np.float32)
            V = np.where(np.isfinite(V), V, 1.0).astype(np.float32)
        except Exception:
            V = np.ones(len(self.net.bus), dtype=np.float32)
        return solvable, p_slack, V

    def get_measured_df(self, fdia_bias_hz: float = 0.0) -> float:
        """Return what the controller/measurement would see *this* step."""
        c = self.cfg
        return float(np.clip(self.df_true + fdia_bias_hz, -c.DF_CLAMP_HZ, c.DF_CLAMP_HZ))

    # --------------------- reset ---------------------
    def reset(self, seed: int | None = None):
        if seed is not None:
            np.random.seed(seed)

        # Pick an operating point (±5%) and freeze it for this episode
        op_factor = np.random.uniform(0.95, 1.05)
        self.oper_load = self.base_load * op_factor
        self.net.load.p_mw = self.oper_load.copy()

        # Reset DER/ESS
        self.der_p[:] = 0.0
        self.der_q[:] = 0.0
        self.ess_p = 0.0
        self.soc   = 0.5

        # All lines in service
        self.net.line["in_service"] = True

        # Frequency state
        self.df_true = 0.0
        self.df_meas = 0.0
        self.df_prev = 0.0

        # Run PF at operating point; memorize reference slack
        solv, p_slack, V = self._run_pf_safe()
        self.Pslack_ref_mw = p_slack

        obs = dict(
            df_true=0.0,
            df_meas=0.0,
            V=V,
            soc=self.soc,
            der_p=self.der_p.copy(),
            der_q=self.der_q.copy(),
            ids=0.0,
            solvable=float(solv),
        )
        return obs

    # --------------------- step ---------------------
    def step(self, action: np.ndarray, laa_frac: float, fdia_bias_hz: float, line_fault: bool):
        cfg = self.cfg

        # 1) Disturbances to PHYSICS (loads/lines). NO FDIA here.
        if laa_frac != 0.0:
            # LAA scales the frozen operating load
            self.net.load.p_mw = self.oper_load * (1.0 + laa_frac)
        else:
            # No attack: hold the operating point exactly
            self.net.load.p_mw = self.oper_load.copy()
        self.net.line.at[self.fault_line_idx, "in_service"] = (not line_fault)

        # 2) Apply action with per-step rate limit & device bounds
        a = np.asarray(action, dtype=np.float32)
        a = np.clip(a, -cfg.RATE_LIMIT_MW, cfg.RATE_LIMIT_MW)

        self.der_p = np.clip(self.der_p + a[:3], cfg.DER_P_MIN, cfg.DER_P_MAX)
        self.der_q = np.clip(self.der_q + a[3:6], cfg.DER_Q_MIN, cfg.DER_Q_MAX)
        self.ess_p = float(np.clip(self.ess_p + a[6], cfg.ESS_P_MIN, cfg.ESS_P_MAX))

        for i in range(3):
            self.net.sgen.at[i, "p_mw"]   = float(self.der_p[i])
            self.net.sgen.at[i, "q_mvar"] = float(self.der_q[i])
        self.net.storage.at[0, "p_mw"] = float(self.ess_p)

        # 3) SoC update (MW -> MWh via DT[h], normalized by capacity)
        try:
            e_max_mwh = float(self.net.storage.at[0, "max_e_mwh"])
        except Exception:
            e_max_mwh = 2.0
        dt_h = cfg.DT / 3600.0
        self.soc = float(np.clip(
            self.soc - (self.ess_p * dt_h) / max(1e-6, e_max_mwh),
            cfg.SOC_MIN, cfg.SOC_MAX
        ))

        # 4) PF & read current slack
        solvable, p_slack, V = self._run_pf_safe()

        # 5) Swing proxy with calibration gain (correct sign)
        # Positive ΔP_slack (more electrical demand) -> Δf should DROP.
        deltaP_slack = p_slack - self.Pslack_ref_mw  # MW
        dfdot = (-cfg.D_SYS * self.df_true - cfg.SWING_GAIN * deltaP_slack) / (2.0 * cfg.H_SYS)  # Hz/s
        self.df_prev = self.df_true
        self.df_true = float(np.clip(self.df_true + cfg.DT * dfdot,
                                     -cfg.DF_CLAMP_HZ, cfg.DF_CLAMP_HZ))

        # 6) Measurement model (FDIA ONLY on measurement)
        ids_flag = 1.0 if abs(fdia_bias_hz) > 1e-12 else 0.0
        self.df_meas = float(np.clip(self.df_true + (fdia_bias_hz if ids_flag else 0.0),
                                     -cfg.DF_CLAMP_HZ, cfg.DF_CLAMP_HZ))

        # 7) Reward:
        #   - primarily penalize TRUE Δf
        #   - small penalty on MEASURED Δf for robustness
        #   - voltage hinge penalty beyond ±5%
        #   - IDS penalty, amplified by RAL if enabled
        v_dev = np.abs(V - 1.0)
        v_excess = np.maximum(0.0, v_dev - 0.05)  # only penalize violations > 5%

        cost_f_true = cfg.ALPHA_F_TRUE * abs(self.df_true)
        cost_f_meas = cfg.ALPHA_F_MEAS * abs(self.df_meas)
        cost_v      = cfg.ALPHA_V * float(np.mean(v_excess))
        cost_ids    = cfg.ALPHA_IDS * ids_flag

        cost = cost_f_true + 0.25 * cost_f_meas + cost_v + cost_ids
        if self.ral_on and ids_flag:
            cost *= cfg.RAL_W
        reward = - cost / cfg.REWARD_SCALE

        obs = dict(
            df_true=self.df_true,
            df_meas=self.df_meas,
            V=V,
            soc=self.soc,
            der_p=self.der_p.copy(),
            der_q=self.der_q.copy(),
            ids=ids_flag,
            solvable=float(solvable),
        )
        return StepResult(obs=obs, reward=reward)
