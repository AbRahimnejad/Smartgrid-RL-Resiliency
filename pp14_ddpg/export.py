
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, Any, List
import os, csv, numpy as np

@dataclass
class EpisodeRecorder:
    dt: float
    controller: str
    scenario: str
    ral_on: int
    rows: List[List[Any]] = field(default_factory=list)
    def header(self) -> List[str]:
        return ["t","df_true_hz","df_meas_hz","ids","solvable","Vmean_pu",
                "a_dP1_mw","a_dP2_mw","a_dP3_mw","a_dQ1_mvar","a_dQ2_mvar","a_dQ3_mvar","a_ess_mw",
                "p_der1_mw","p_der2_mw","p_der3_mw","q_der1_mvar","q_der2_mvar","q_der3_mvar","p_ess_mw",
                "laa_frac","fdia_bias_hz","line_fault"]
    def log_step(self, t_idx: int, obs: Dict[str, Any], action: np.ndarray, env, laa_frac: float, fdia_bias: float, line_fault: bool):
        a = np.asarray(action, dtype=float).tolist()
        p_der = env.net.sgen.p_mw.values[:3].astype(float).tolist()
        q_der = env.net.sgen.q_mvar.values[:3].astype(float).tolist()
        p_ess = float(env.net.storage.p_mw.sum()) if len(env.net.storage) else 0.0
        V = obs.get("V", None)
        vmean = float(np.nanmean(V)) if V is not None and len(V) else 1.0
        row = [round((t_idx+1)*self.dt,6),
               float(obs["df_true"]), float(obs.get("df_meas", obs["df_true"])),
               int(obs.get("ids",0)), int(obs.get("solvable",1)), vmean,
               a[0],a[1],a[2], a[3],a[4],a[5], a[6],
               p_der[0],p_der[1],p_der[2], q_der[0],q_der[1],q_der[2], p_ess,
               float(laa_frac), float(fdia_bias), int(bool(line_fault))]
        self.rows.append(row)
    def save_csv(self, out_dir: str) -> str:
        os.makedirs(out_dir, exist_ok=True)
        fn = f"trace_{self.controller}_{self.scenario}_RAL{self.ral_on}.csv"
        path = os.path.join(out_dir, fn)
        with open(path, "w", newline="", encoding="utf-8-sig") as f:
            w = csv.writer(f); w.writerow(self.header()); w.writerows(self.rows)
        return path
