# ============================================================
# Simulation SAC - JOURNÉE avec ComfortReward
# Formule : pénalité basée sur |T_in - T_spt|^3 (piecewise)
# ============================================================
from citylearn.citylearn import CityLearnEnv
from citylearn.reward_function import RewardFunction, ComfortReward
from citylearn.agents.sac import SAC
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np
import os
import time
import types

# 1. Sonde
env_probe = CityLearnEnv(
    'citylearn_challenge_2023_phase_2_local_evaluation',
    central_agent=True,
    reward_function=RewardFunction,
)
DATASET_LEN = len(env_probe.buildings[0].energy_simulation.non_shiftable_load)
print(f"Longueur dataset : {DATASET_LEN} h = {DATASET_LEN/24:.1f} jours")
del env_probe

# 2. Choix de la journée
JOUR_CIBLE = 0
JOUR_CIBLE = min(JOUR_CIBLE, DATASET_LEN // 24 - 1)
START_STEP = JOUR_CIBLE * 24
END_STEP   = START_STEP + 23

print(f"Journée simulée : jour {JOUR_CIBLE} (steps {START_STEP} → {END_STEP})")

# 3. Environnement
env = CityLearnEnv(
    'citylearn_challenge_2023_phase_2_local_evaluation',
    central_agent=True,
    reward_function=RewardFunction,
    simulation_start_time_step=START_STEP,
    simulation_end_time_step=END_STEP,
    simulate_power_outage=False,
)

# --- Activer ComfortReward ---
env_metadata = {
    'central_agent': env.central_agent,
    'buildings': [{'name': b.name} for b in env.buildings],
}
try:
    env.reward_function = ComfortReward(env_metadata)
    print("✅ Reward ComfortReward activé")
except Exception as e:
    print(f"⚠️  ComfortReward indisponible ({e}) → RewardFunction conservé")

N_STEPS    = END_STEP - START_STEP + 1
N_EPISODES = 100

print(f"Bâtiments : {[b.name for b in env.buildings]}")
print(f"N_STEPS   : {N_STEPS}")
print(f"Reward    : {type(env.reward_function).__name__}")

# 4. Agent SAC + patch anti-bug
agent = SAC(env)

def safe_get_normalized_observations(self, i, o):
    return np.array(o, dtype=float)

agent.get_normalized_observations = types.MethodType(
    safe_get_normalized_observations, agent
)
print("✅ Patch SAC appliqué\n")

# 5. Utilitaire
def to_float_sum(x):
    if x is None: return 0.0
    if isinstance(x, dict):
        return float(np.sum([to_float_sum(v) for v in x.values()]))
    if isinstance(x, (list, tuple, np.ndarray)):
        if len(x) == 0: return 0.0
        first = x[0]
        if np.isscalar(first) or isinstance(first, (int, float, np.number)):
            return float(np.sum(x))
        return float(np.sum([to_float_sum(v) for v in x]))
    try: return float(x)
    except: return 0.0

# 6. Entraînement
print(f"Entraînement SAC ({N_EPISODES} épisodes de {N_STEPS}h)...")
print("⚠️  NE TOUCHE À RIEN pendant l'entraînement\n")

episode_rewards = []
t0 = time.time()

for ep in range(N_EPISODES):
    agent.learn(episodes=1, deterministic_finish=False)
    total = to_float_sum(env.episode_rewards[-1]) if hasattr(env, 'episode_rewards') else 0.0
    episode_rewards.append(total)

    if (ep + 1) % 10 == 0 or ep == 0:
        print(f"  Ep {ep+1:3d}/{N_EPISODES} | Reward = {total:12.2f} | "
              f"Temps = {time.time()-t0:5.1f}s")

print("\nEntraînement terminé !")

# 7. KPIs
kpis = env.evaluate()
print("\nKPIs :")
print(kpis[['cost_function', 'value']].to_string(index=False))

# 8. Évaluation finale
observations, _ = env.reset()
rewards_hist = []
terminated = False
while not terminated:
    actions = agent.predict(observations, deterministic=True)
    observations, reward, terminated, truncated, info = env.step(actions)
    terminated = terminated or truncated
    rewards_hist.append(to_float_sum(reward))

T = len(rewards_hist)
hours = np.arange(T)

# 9. Extraction des données (demandes, PV, prix, températures)
demand_non_shift = np.zeros(T)
demand_cooling   = np.zeros(T)
demand_heating   = np.zeros(T)
demand_dhw       = np.zeros(T)
production_pv    = np.zeros(T)
price_series     = np.zeros(T)
temp_indoor      = np.zeros(T)
temp_setpoint    = np.zeros(T)

def pad_to(arr, n):
    arr = np.array(arr, dtype=float)[:n]
    if len(arr) < n: arr = np.pad(arr, (0, n - len(arr)))
    return arr

# --- Données du simulation ---
for b in env.buildings:
    es = b.energy_simulation
    try: demand_non_shift += pad_to(es.non_shiftable_load, T)
    except: pass
    try: demand_cooling   += pad_to(es.cooling_demand,     T)
    except: pass
    try: demand_heating   += pad_to(es.heating_demand,     T)
    except: pass
    try: demand_dhw       += pad_to(es.dhw_demand,         T)
    except: pass
    try: production_pv    += pad_to(es.solar_generation,   T)
    except: pass
    try: price_series     += pad_to(b.pricing.electricity_pricing, T)
    except: pass

if len(env.buildings) > 1:
    price_series /= len(env.buildings)

demand_total = demand_non_shift + demand_cooling + demand_heating + demand_dhw

# --- Température intérieure : extraire depuis l'historique du bâtiment 1 ---
try:
    obs_hist = env.buildings[0].observations
    temp_names = env.observation_names[0]
    # Chercher l'index de indoor_dry_bulb_temperature
    if 'indoor_dry_bulb_temperature' in temp_names:
        idx_temp = temp_names.index('indoor_dry_bulb_temperature')
        for t in range(min(T, len(obs_hist))):
            row = obs_hist[t]
            if isinstance(row, dict):
                temp_indoor[t] = row.get('indoor_dry_bulb_temperature', 0)
            else:
                temp_indoor[t] = row[idx_temp] if idx_temp < len(row) else 0
except Exception as e:
    print(f"Attention extraction temp : {e}")

# 10. Annotation des pics
def annotate_peaks(ax, x, y, n_peaks=5, color='black', fmt="{:.1f}"):
    if len(y) == 0: return
    y = np.asarray(y)
    order = np.argsort(y)[::-1]
    picked = []
    min_gap = max(1, len(y) // 10)
    for idx in order:
        if all(abs(idx - p) > min_gap for p in picked):
            picked.append(idx)
        if len(picked) >= n_peaks: break
    for idx in picked:
        ax.annotate(fmt.format(y[idx]), xy=(x[idx], y[idx]),
            xytext=(5, 8), textcoords='offset points', fontsize=9,
            color=color, fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.2', fc='white', ec=color, alpha=0.85),
            arrowprops=dict(arrowstyle='->', color=color, lw=0.8))

# 11. Figures
os.makedirs("figures_comfort_jour", exist_ok=True)
reward_label = type(env.reward_function).__name__

# Reward
plt.figure(figsize=(13, 4))
plt.plot(hours, rewards_hist, color='#1f77b4', linewidth=0.9)
plt.axhline(0, color='black', linewidth=0.7)
plt.xlabel("Heure de la journée")
plt.ylabel(f"Reward {reward_label}")
plt.title(f"Reward {reward_label} — Journée (jour {JOUR_CIBLE}, {N_EPISODES} épisodes)")
plt.grid(alpha=0.3); plt.tight_layout()
plt.savefig("figures_comfort_jour/01_reward.png", dpi=150); plt.show(block=False)

# Température intérieure + setpoint + pics
plt.figure(figsize=(13, 4))
plt.plot(hours, temp_indoor, color='#d62728', linewidth=0.9, label='T_in mesurée')
if np.any(temp_setpoint > 0):
    plt.plot(hours, temp_setpoint, color='#2ca02c', linewidth=0.9, linestyle='--', label='T_spt (setpoint)')
plt.xlabel("Heure de la journée"); plt.ylabel("Température (°C)")
plt.title(f"Température intérieure — Journée ({reward_label})")
annotate_peaks(plt.gca(), hours, temp_indoor, n_peaks=3, color='#8b0000', fmt="{:.1f}°C")
plt.legend(loc='upper left'); plt.grid(alpha=0.3); plt.tight_layout()
plt.savefig("figures_comfort_jour/02_temperature.png", dpi=150); plt.show(block=False)

# Demande totale
plt.figure(figsize=(13, 4))
plt.plot(hours, demand_total, color='#d62728', linewidth=0.9, label='Demande totale')
plt.plot(hours, demand_cooling, color='#ff9896', linewidth=0.6, alpha=0.7, label='Climatisation')
plt.xlabel("Heure de la journée"); plt.ylabel("Demande (kWh)")
plt.title(f"Demande électrique totale — Journée ({reward_label})")
annotate_peaks(plt.gca(), hours, demand_total, n_peaks=5, color='#8b0000', fmt="{:.1f} kWh")
plt.legend(loc='upper left'); plt.grid(alpha=0.3); plt.tight_layout()
plt.savefig("figures_comfort_jour/03_demande_totale.png", dpi=150); plt.show(block=False)

# Production PV
plt.figure(figsize=(13, 4))
plt.plot(hours, production_pv, color='#2ca02c', linewidth=0.9)
plt.xlabel("Heure de la journée"); plt.ylabel("Production PV (kWh)")
plt.title(f"Production solaire — Journée ({reward_label})")
annotate_peaks(plt.gca(), hours, production_pv, n_peaks=5, color='#006400', fmt="{:.1f} kWh")
plt.grid(alpha=0.3); plt.tight_layout()
plt.savefig("figures_comfort_jour/04_production_pv.png", dpi=150); plt.show(block=False)

# Prix
plt.figure(figsize=(13, 4))
plt.plot(hours, price_series, color='#ff7f0e', linewidth=0.9)
plt.xlabel("Heure de la journée"); plt.ylabel("Prix ($/kWh)")
plt.title(f"Prix de l'électricité — Journée ({reward_label})")
annotate_peaks(plt.gca(), hours, price_series, n_peaks=5, color='#8b4500', fmt="{:.3f} $/kWh")
plt.grid(alpha=0.3); plt.tight_layout()
plt.savefig("figures_comfort_jour/05_prix.png", dpi=150); plt.show(block=False)

# Apprentissage
plt.figure(figsize=(10, 4))
plt.plot(range(1, N_EPISODES + 1), episode_rewards, color='#9467bd', linewidth=1.0)
plt.xlabel("Épisode"); plt.ylabel("Reward cumulé (24h)")
plt.title(f"Apprentissage SAC ({reward_label}) — Journée")
plt.grid(alpha=0.3); plt.tight_layout()
plt.savefig("figures_comfort_jour/06_apprentissage.png", dpi=150); plt.show(block=False)

# Synthèse 4 panneaux
fig, axes = plt.subplots(4, 1, figsize=(13, 11), sharex=True)
axes[0].plot(hours, rewards_hist, color='#1f77b4', linewidth=0.9)
axes[0].axhline(0, color='black', linewidth=0.7)
axes[0].set_ylabel(f"Reward {reward_label}")
axes[0].set_title(f"Synthèse {reward_label} — Journée (jour {JOUR_CIBLE})")
axes[0].grid(alpha=0.3)

axes[1].plot(hours, temp_indoor, color='#d62728', linewidth=0.9)
axes[1].set_ylabel("T_in (°C)"); axes[1].set_title("Température intérieure")
axes[1].grid(alpha=0.3)

axes[2].plot(hours, demand_total, color='#d62728', linewidth=0.9)
axes[2].set_ylabel("Demande (kWh)"); axes[2].set_title("Demande totale")
annotate_peaks(axes[2], hours, demand_total, n_peaks=3, color='#8b0000')
axes[2].grid(alpha=0.3)

axes[3].plot(hours, price_series, color='#ff7f0e', linewidth=0.9)
axes[3].set_ylabel("Prix ($/kWh)"); axes[3].set_xlabel("Heure de la journée")
axes[3].set_title("Prix de l'électricité")
annotate_peaks(axes[3], hours, price_series, n_peaks=3, color='#8b4500', fmt="{:.3f}")
axes[3].grid(alpha=0.3)

plt.tight_layout()
plt.savefig("figures_comfort_jour/07_synthese.png", dpi=150)
print("\nToutes les figures sont dans ./figures_comfort_jour/")
plt.show(block=True)