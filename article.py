from citylearn.citylearn import CityLearnEnv
from citylearn.agents.rbc import BasicRBC
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np
import os

# ============================================
# 1. Charger l'environnement
# ============================================
env = CityLearnEnv(
    'citylearn_challenge_2023_phase_2_local_evaluation',
    central_agent=False
)

print(" Environnement chargé !")
print(f"   Nombre de bâtiments : {len(env.buildings)}")
print(f"   Nombre de pas de temps : {env.time_steps}")

# ============================================
# 2. Créer l'agent et lancer la simulation
# ============================================
agent = BasicRBC(env)
observations, _ = env.reset()
rewards = []

print("\n Simulation en cours...")
while not env.terminated:
    actions = agent.predict(observations)
    observations, reward, terminated, truncated, info = env.step(actions)
    rewards.append(sum(reward))

print(" Simulation terminée !")

# ============================================
# 3. Afficher les KPIs
# ============================================
kpis = env.evaluate()
print("\n KPIs obtenus :")
print(kpis[['cost_function', 'value']].to_string(index=False))

# ============================================
# 4. Extraire les données de juin
# ============================================
building  = env.buildings[0]
n_hours   = env.time_steps
month     = np.array(building.energy_simulation.month[:n_hours])
mask_juin = month == 6

equipment = np.array(building.energy_simulation.non_shiftable_load[:n_hours])[mask_juin]
cooling   = np.array(building.energy_simulation.cooling_demand[:n_hours])[mask_juin]
dhw       = np.array(building.energy_simulation.dhw_demand[:n_hours])[mask_juin]
heating   = np.array(building.energy_simulation.heating_demand[:n_hours])[mask_juin]
solar     = np.array(building.energy_simulation.solar_generation[:n_hours])[mask_juin]
pricing   = np.array(building.pricing.electricity_pricing[:n_hours])[mask_juin]

n_juin = len(solar)
jours  = np.arange(n_juin) / 24

print(f"\n Données de juin : {n_juin} heures ({n_juin//24} jours)")
print(f"\n Statistiques juin :")
print(f"  Equipment Electric Power max : {equipment.max():.2f} kWh")
print(f"  Cooling Load max             : {cooling.max():.2f} kWh")
print(f"  DHW Heating max              : {dhw.max():.2f} kWh")
print(f"  Heating Load max             : {heating.max():.2f} kWh")
print(f"  Solar Generation max         : {solar.max():.2f} W/kW")
print(f"  Prix max                     : {pricing.max():.4f} $/kWh")

# ============================================
# 5. Graphique avec 6 sous-plots séparés
# ============================================
fig, axes = plt.subplots(6, 1, figsize=(13, 18), sharex=True)

# --- Plot 1 : Equipment Electric Power ---
idx = np.argmax(equipment)
axes[0].plot(jours, equipment, color='#d62728', linewidth=0.8,
             label='Equipment Electric Power')
axes[0].scatter(jours[idx], equipment[idx], color='black', zorder=5, s=50,
                label=f'Pic ({equipment[idx]:.2f} kWh)')
axes[0].annotate(f'{equipment[idx]:.2f} kWh',
                  (jours[idx], equipment[idx]),
                  textcoords="offset points", xytext=(0, 8),
                  ha='center', fontweight='bold')
axes[0].set_ylabel('kWh')
axes[0].set_title("Equipment Electric Power (Non Shiftable Load)")
axes[0].legend(loc='upper right', fontsize=8)
axes[0].grid(alpha=0.3)

# --- Plot 2 : Cooling Load ---
idx = np.argmax(cooling)
axes[1].plot(jours, cooling, color='#9467bd', linewidth=0.8,
             label='Cooling Load')
axes[1].scatter(jours[idx], cooling[idx], color='black', zorder=5, s=50,
                label=f'Pic ({cooling[idx]:.2f} kWh)')
axes[1].annotate(f'{cooling[idx]:.2f} kWh',
                  (jours[idx], cooling[idx]),
                  textcoords="offset points", xytext=(0, 8),
                  ha='center', fontweight='bold')
axes[1].set_ylabel('kWh')
axes[1].set_title("Cooling Load (Climatisation)")
axes[1].legend(loc='upper right', fontsize=8)
axes[1].grid(alpha=0.3)

# --- Plot 3 : DHW Heating ---
idx = np.argmax(dhw)
axes[2].plot(jours, dhw, color='#8c564b', linewidth=0.8,
             label='DHW Heating')
axes[2].scatter(jours[idx], dhw[idx], color='black', zorder=5, s=50,
                label=f'Pic ({dhw[idx]:.2f} kWh)')
axes[2].annotate(f'{dhw[idx]:.2f} kWh',
                  (jours[idx], dhw[idx]),
                  textcoords="offset points", xytext=(0, 8),
                  ha='center', fontweight='bold')
axes[2].set_ylabel('kWh')
axes[2].set_title("DHW Heating (Eau Chaude Sanitaire)")
axes[2].legend(loc='upper right', fontsize=8)
axes[2].grid(alpha=0.3)

# --- Plot 4 : Heating Load ---
idx = np.argmax(heating)
axes[3].plot(jours, heating, color='#e377c2', linewidth=0.8,
             label='Heating Load')
axes[3].scatter(jours[idx], heating[idx], color='black', zorder=5, s=50,
                label=f'Pic ({heating[idx]:.2f} kWh)')
axes[3].annotate(f'{heating[idx]:.2f} kWh',
                  (jours[idx], heating[idx]),
                  textcoords="offset points", xytext=(0, 8),
                  ha='center', fontweight='bold')
axes[3].set_ylabel('kWh')
axes[3].set_title("Heating Load (Chauffage)")
axes[3].legend(loc='upper right', fontsize=8)
axes[3].grid(alpha=0.3)

# --- Plot 5 : Solar Generation ---
idx = np.argmax(solar)
axes[4].plot(jours, solar, color='#ff7f0e', linewidth=0.8,
             label='Solar Generation')
axes[4].fill_between(jours, solar, color='#ff7f0e', alpha=0.2)
axes[4].scatter(jours[idx], solar[idx], color='black', zorder=5, s=50,
                label=f'Pic ({solar[idx]:.2f} W/kW)')
axes[4].annotate(f'{solar[idx]:.2f} W/kW',
                  (jours[idx], solar[idx]),
                  textcoords="offset points", xytext=(0, 8),
                  ha='center', fontweight='bold')
axes[4].set_ylabel('W/kW')
axes[4].set_title("Solar Generation (Production PV)")
axes[4].legend(loc='upper right', fontsize=8)
axes[4].grid(alpha=0.3)

# --- Plot 6 : Prix électricité ---
idx = np.argmax(pricing)
axes[5].plot(jours, pricing, color='#2ca02c', linewidth=0.8,
             label="Prix de l'électricité")
axes[5].scatter(jours[idx], pricing[idx], color='black', zorder=5, s=50,
                label=f'Prix max ({pricing[idx]:.4f} $/kWh)')
axes[5].annotate(f'{pricing[idx]:.4f} $/kWh',
                  (jours[idx], pricing[idx]),
                  textcoords="offset points", xytext=(0, 8),
                  ha='center', fontweight='bold')
axes[5].set_ylabel('$/kWh')
axes[5].set_xlabel("Jour de juin")
axes[5].set_title("Prix de l'électricité (ToU)")
axes[5].legend(loc='upper right', fontsize=8)
axes[5].grid(alpha=0.3)

plt.suptitle("Simulation BasicRBC — Dataset Article — Juin", 
             fontsize=14, fontweight='bold', y=1.01)
plt.tight_layout()

chemin = os.path.join(os.getcwd(), "simulation_demandes_separees.png")
plt.savefig(chemin, dpi=150, bbox_inches='tight')
print(f"\n Graphique enregistré : {chemin}")

# ============================================
# 6. Graphique Reward
# ============================================
plt.figure(figsize=(12, 4))
plt.plot(rewards, color='#1f77b4', linewidth=0.8)
plt.xlabel('Pas de temps (heures)')
plt.ylabel('Reward')
plt.title('Reward — Simulation BasicRBC')
plt.grid(alpha=0.3)
plt.tight_layout()

chemin2 = os.path.join(os.getcwd(), "simulation_reward.png")
plt.savefig(chemin2, dpi=150)
print(f" Graphique reward enregistré : {chemin2}")

plt.show(block=True)