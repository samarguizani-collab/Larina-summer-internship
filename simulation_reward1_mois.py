# ============================================================
# Simulation SAC - Période adaptée automatiquement à la longueur du dataset
# ============================================================
from citylearn.citylearn import CityLearnEnv
from citylearn.reward_function import RewardFunction
from citylearn.agents.sac import SAC
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np
import os
import time
import types
from gymnasium import spaces

# ============================================
# 1. Charger d'abord pour connaître la longueur réelle
# ============================================
env_probe = CityLearnEnv(
    'citylearn_challenge_2023_phase_2_local_evaluation',
    central_agent=True,
    reward_function=RewardFunction,
)

b0 = env_probe.buildings[0]
DATASET_LEN = len(b0.energy_simulation.non_shiftable_load)
print(f"Longueur réelle du dataset : {DATASET_LEN} heures "
      f"= {DATASET_LEN / 24:.1f} jours")

del env_probe   # on libère

# ============================================
# 2. Choisir la fenêtre automatiquement
# ============================================
# On prend la plus grande fenêtre possible (tout le dataset)
START_STEP = 0
END_STEP   = DATASET_LEN - 1

# Si le dataset fait moins de 30 jours, on prend tout
# Sinon, on prend les 30 premiers jours
MAX_JOURS = min(30, DATASET_LEN // 24)
END_STEP  = START_STEP + MAX_JOURS * 24 - 1

print(f"Fenêtre simulée : step {START_STEP} → {END_STEP} "
      f"({MAX_JOURS} jours, {END_STEP - START_STEP + 1} heures)")

# ============================================
# 3. Créer l'environnement avec cette fenêtre
# ============================================
env = CityLearnEnv(
    'citylearn_challenge_2023_phase_2_local_evaluation',
    central_agent=True,
    reward_function=RewardFunction,
    simulation_start_time_step=START_STEP,
    simulation_end_time_step=END_STEP,
    simulate_power_outage=False,
)

N_STEPS    = END_STEP - START_STEP + 1
N_EPISODES = 50

print(f"Bâtiments : {[b.name for b in env.buildings]}")
print(f"N_STEPS   : {N_STEPS}")

# ============================================
# 4. Agent SAC + PATCH anti-bug
# ============================================
agent = SAC(env)

def safe_get_normalized_observations(self, i, o):
    return np.array(o, dtype=float)

agent.get_normalized_observations = types.MethodType(
    safe_get_normalized_observations, agent
)
print("✅ Patch SAC appliqué\n")

# ============================================
# 5. Utilitaire
# ============================================
def to_float_sum(x):
    if x is None:
        return 0.0
    if isinstance(x, dict):
        return float(np.sum([to_float_sum(v) for v in x.values()]))
    if isinstance(x, (list, tuple, np.ndarray)):
        if len(x) == 0:
            return 0.0
        first = x[0]
        if np.isscalar(first) or isinstance(first, (int, float, np.number)):
            return float(np.sum(x))
        return float(np.sum([to_float_sum(v) for v in x]))
    try:
        return float(x)
    except Exception:
        return 0.0

# ============================================
# 6. Entraînement
# ============================================
print(f"Entraînement SAC ({N_EPISODES} épisodes de {N_STEPS}h)...")
episode_rewards = []
t0 = time.time()

for ep in range(N_EPISODES):
    agent.learn(episodes=1, deterministic_finish=False)
    total = to_float_sum(env.episode_rewards[-1]) if hasattr(env, 'episode_rewards') else 0.0
    episode_rewards.append(total)

    if (ep + 1) % 5 == 0 or ep == 0:
        print(f"  Ep {ep+1:3d}/{N_EPISODES} | Reward = {total:10.2f} | "
              f"Temps = {time.time()-t0:6.1f}s")

print("\nEntraînement terminé !")

# ============================================
# 7. KPIs
# ============================================
kpis = env.evaluate()
print("\nKPIs :")
print(kpis[['cost_function', 'value']].to_string(index=False))

# ============================================
# 8. Évaluation finale
# ============================================
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

# ============================================
# 9. Extraction des demandes
# ============================================
demand_non_shift = np.zeros(T)
demand_cooling   = np.zeros(T)
demand_heating   = np.zeros(T)
demand_dhw       = np.zeros(T)
production_pv    = np.zeros(T)
price_series     = np.zeros(T)

def pad_to(arr, n):
    arr = np.array(arr, dtype=float)[:n]
    if len(arr) < n:
        arr = np.pad(arr, (0, n - len(arr)))
    return arr

for b in env.buildings:
    es = b.energy_simulation
    try: demand_non_shift += pad_to(es.non_shiftable_load, T)
    except Exception as e: print(f"  {b.name} non_shiftable_load : {e}")
    try: demand_cooling   += pad_to(es.cooling_demand,     T)
    except Exception as e: print(f"  {b.name} cooling_demand    : {e}")
    try: demand_heating   += pad_to(es.heating_demand,     T)
    except Exception as e: print(f"  {b.name} heating_demand    : {e}")
    try: demand_dhw       += pad_to(es.dhw_demand,         T)
    except Exception as e: print(f"  {b.name} dhw_demand        : {e}")
    try: production_pv    += pad_to(es.solar_generation,   T)
    except Exception as e: print(f"  {b.name} solar_generation  : {e}")
    try: price_series     += pad_to(b.pricing.electricity_pricing, T)
    except Exception as e: print(f"  {b.name} pricing           : {e}")

if len(env.buildings) > 1:
    price_series /= len(env.buildings)

demand_total = demand_non_shift + demand_cooling + demand_heating + demand_dhw

# ============================================
# 10. Annotation des pics
# ============================================
def annotate_peaks(ax, x, y, n_peaks=5, color='black', fmt="{:.1f}"):
    if len(y) == 0:
        return
    y = np.asarray(y)
    order = np.argsort(y)[::-1]
    picked = []
    min_gap = max(1, len(y) // 50)
    for idx in order:
        if all(abs(idx - p) > min_gap for p in picked):
            picked.append(idx)
        if len(picked) >= n_peaks:
            break
    for idx in picked:
        ax.annotate(
            fmt.format(y[idx]),
            xy=(x[idx], y[idx]),
            xytext=(5, 8), textcoords='offset points',
            fontsize=9, color=color, fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.2', fc='white', ec=color, alpha=0.85),
            arrowprops=dict(arrowstyle='->', color=color, lw=0.8),
        )

# ============================================
# 11. Figures
# ============================================
os.makedirs("figures_juin", exist_ok=True)

# Reward (pas d'annotation)
plt.figure(figsize=(13, 4))
plt.plot(hours, rewards_hist, color='#1f77b4', linewidth=0.7)
plt.axhline(0, color='black', linewidth=0.7)
plt.xlabel("Heures depuis le début")
plt.ylabel("Reward")
plt.title(f"Reward SAC — {N_EPISODES} épisodes")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("figures_juin/01_reward.png", dpi=150)
plt.show(block=False)

# Demande totale + clim + pics
plt.figure(figsize=(13, 4))
plt.plot(hours, demand_total, color='#d62728', linewidth=0.7, label='Demande totale')
plt.plot(hours, demand_cooling, color='#ff9896', linewidth=0.5,
         alpha=0.6, label='Climatisation')
plt.xlabel("Heures depuis le début")
plt.ylabel("Demande (kWh)")
plt.title("Demande électrique totale")
annotate_peaks(plt.gca(), hours, demand_total, n_peaks=5,
               color='#8b0000', fmt="{:.1f} kWh")
plt.legend(loc='upper left')
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("figures_juin/02_demande_totale.png", dpi=150)
plt.show(block=False)

# Détail empilé
plt.figure(figsize=(13, 4))
plt.stackplot(hours, demand_non_shift, demand_cooling, demand_heating, demand_dhw,
              labels=['Non-déplaçable', 'Climatisation', 'Chauffage', 'Eau chaude'],
              colors=['#1f77b4', '#d62728', '#ff7f0e', '#2ca02c'], alpha=0.8)
plt.xlabel("Heures depuis le début")
plt.ylabel("Demande (kWh)")
plt.title("Détail des demandes électriques")
annotate_peaks(plt.gca(), hours, demand_total, n_peaks=5, color='black')
plt.legend(loc='upper left')
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("figures_juin/03_detail_demandes.png", dpi=150)
plt.show(block=False)

# Production PV + pics
plt.figure(figsize=(13, 4))
plt.plot(hours, production_pv, color='#2ca02c', linewidth=0.7)
plt.xlabel("Heures depuis le début")
plt.ylabel("Production PV (kWh)")
plt.title("Production solaire")
annotate_peaks(plt.gca(), hours, production_pv, n_peaks=5,
               color='#006400', fmt="{:.1f} kWh")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("figures_juin/04_production_pv.png", dpi=150)
plt.show(block=False)

# Prix + pics
plt.figure(figsize=(13, 4))
plt.plot(hours, price_series, color='#ff7f0e', linewidth=0.8)
plt.xlabel("Heures depuis le début")
plt.ylabel("Prix ($/kWh)")
plt.title("Prix de l'électricité")
annotate_peaks(plt.gca(), hours, price_series, n_peaks=5,
               color='#8b4500', fmt="{:.3f} $/kWh")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("figures_juin/05_prix.png", dpi=150)
plt.show(block=False)

# Courbe d'apprentissage
plt.figure(figsize=(10, 4))
plt.plot(range(1, N_EPISODES + 1), episode_rewards,
         marker='o', markersize=3, color='#9467bd')
plt.xlabel("Épisode")
plt.ylabel("Reward cumulé")
plt.title("Apprentissage SAC")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("figures_juin/06_apprentissage.png", dpi=150)
plt.show(block=False)

# Synthèse 4 panneaux
fig, axes = plt.subplots(4, 1, figsize=(13, 11), sharex=True)

axes[0].plot(hours, rewards_hist, color='#1f77b4', linewidth=0.7)
axes[0].axhline(0, color='black', linewidth=0.7)
axes[0].set_ylabel("Reward")
axes[0].set_title(f"Synthèse SAC — {N_EPISODES} épisodes")
axes[0].grid(alpha=0.3)

axes[1].plot(hours, demand_total, color='#d62728', linewidth=0.7)
axes[1].set_ylabel("Demande (kWh)")
axes[1].set_title("Demande totale")
annotate_peaks(axes[1], hours, demand_total, n_peaks=3, color='#8b0000')
axes[1].grid(alpha=0.3)

axes[2].plot(hours, production_pv, color='#2ca02c', linewidth=0.7)
axes[2].set_ylabel("Production PV (kWh)")
axes[2].set_title("Production solaire")
annotate_peaks(axes[2], hours, production_pv, n_peaks=3, color='#006400')
axes[2].grid(alpha=0.3)

axes[3].plot(hours, price_series, color='#ff7f0e', linewidth=0.8)
axes[3].set_ylabel("Prix ($/kWh)")
axes[3].set_xlabel("Heures depuis le début")
axes[3].set_title("Prix de l'électricité")
annotate_peaks(axes[3], hours, price_series, n_peaks=3, color='#8b4500',
               fmt="{:.3f}")
axes[3].grid(alpha=0.3)

plt.tight_layout()
plt.savefig("figures_juin/07_synthese.png", dpi=150)
print("\nToutes les figures sont dans ./figures_juin/")
plt.show(block=True)