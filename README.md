
# PP14-RL-Resilience  
**IEEE-14 Power System Resilience with DDPG, Pandapower Surrogate, and ANDES Validation**

This repository provides a reinforcement learning (RL) framework for **frequency and voltage resilience control** on the IEEE-14 bus test system.  

The workflow is:  
1. Train a **DDPG agent** on a *pandapower-based surrogate* (power flow + calibrated swing proxy).  
2. Export policy traces for multiple attack scenarios.  
3. Replay these traces in **ANDES** for dynamic validation with generator/governor/AVR models.  

---

## Features

- **Surrogate simulation**  
  - Pandapower solves AC power flow each step.  
  - Slack bus imbalance → frequency deviation via calibrated swing equation.  
  - `SWING_GAIN` parameter ensures surrogate ≈ ANDES (within ~10%).  

- **Events & attacks**  
  - Load-Altering Attack (LAA)  
  - False Data Injection Attack (FDIA)  
  - Line outage  

- **Controllers**  
  - Deep Deterministic Policy Gradient (DDPG) agent  
  - Baseline proportional (Kp) controller  

- **Metrics**  
  - Frequency nadir [Hz]  
  - ROCOF [Hz/s]  
  - Recovery time [s]  
  - Bus voltage violations [%]  
  - ESS state-of-charge trajectory  

---

## Repository Structure

```text
pp14-rl-resilience/
├─ pp14_ddpg/ # core package
│ ├─ init.py
│ ├─ config.py # system + RL hyperparameters
│ ├─ env.py # IEEE-14 + DER/ESS + swing proxy
│ ├─ agent.py # DDPG implementation
│ ├─ train.py # training loop, eval, Monte Carlo sweeps
│ ├─ utils.py # helpers (metrics, obs vectors, CSV writer)
│ ├─ export.py # EpisodeRecorder (trace logging)
├─ run_pp14_ddpg.py # main training script (surrogate)
├─ run_andes_replay.py # ANDES validation with exported traces
├─ tests_pp14.py # environment unit tests
└─ README.md # this file

```

## Installation (Windows)

1. **Clone the repo** (or create it on GitHub and drag-drop the files):  
   ```powershell
   git clone https://github.com/<your-username>/<repo-name>.git
   cd <repo-name>
  ```
2. Create a Conda environment
  ```powershell
  conda create -n pp14 python=3.10 -y
  conda activate pp14
  ```


4. Install dependencies
```powershell
pip install -e .
pip install pandapower andes torch matplotlib tqdm pandas


## Usage (Windows)
1. Run sanity tests

Check that the environment is stable and attacks trigger properly:

```powershell
python tests_pp14.py


Expected output:
[PASS] noattack_stable_frequency
[PASS] fdia_ids_triggers
[PASS] laa_open_loop_drop_and_recover
[PASS] action_rate_limit
4/4 tests passed


2. Train the RL agent (surrogate)
```powershell
python run_pp14_ddpg.py

Artifacts Produced

- **pp14_ddpg_training.png** → Learning curves (critic loss, eval returns)  
- **pp14_ddpg_actor.pt**, **pp14_ddpg_critic.pt** → Trained DDPG models (PyTorch state_dicts)  
- **pp14_surrogate_metrics.csv** → Scenario metrics (NN vs Kp across attacks)  
- **pp14_mc_metrics_surrogate.csv** → Monte Carlo robustness sweeps (multiple seeds & operating points)  
- **pp14_DDPG_[LINE|LAA|FDIA]_df.png** → Frequency traces under different attack scenarios  
- **traces/trace_[NN|KP]_[scenario]_RAL1.csv** → Exported action traces (NN policy vs Kp baseline)  
- **pp14_final_Kp_from_ddpg_run.txt** → Tuned proportional baseline controller (Kp)  


3. Replay in ANDES (dynamic validation)

After training, validate the policy in ANDES:

```powershell
python run_andes_replay.py --andes_case ieee14.json --traces_dir traces

## ANDES Validation

This workflow bridges the surrogate environment (pandapower + swing proxy) with a dynamic simulator (ANDES):

- Load the IEEE-14 dynamic case in ANDES.  
- Replay exported action sequences (from NN policy and Kp baseline).  
- Generate Δf(t), V(t), and other dynamic plots.  
- Output validation CSV metrics.  

### Calibration Workflow
1. Run baseline Kp controller in both surrogate and ANDES.  
2. Compare key dynamic metrics:  
   - Frequency nadir  
   - ROCOF  
   - Recovery time  
3. Adjust `SWING_GAIN` in `config.py` until surrogate error ≤ 10%.  

---

## Development Notes

- **Clear Python cache** to ensure new changes are applied:  
  ```bash
  rmdir /s /q __pycache__
- To remove environment:
```powershell
conda deactivate
conda remove -n pp14 --all


## Citation


If you use this repository in academic work, please cite:

- [pandapower: IEEE Transactions on Power Systems, 2018](https://doi.org/10.1109/TPWRS.2018.2829021)

- [ANDES: Xu et al., 2020+](https://doi.org/10.1109/TPWRS.2020.3022980)


