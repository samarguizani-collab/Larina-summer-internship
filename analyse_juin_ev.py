from citylearn.citylearn import CityLearnEnv
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np
import os

# ============================================
# 1. Charger le dataset avec EVs
# ============================================
env = CityLearnEnv('citylearn_challenge_2022_phase_all_plus_evs')
building = env.buildings[1]

# ============================================
# 2. Récupérer les données sur 1 an
# ============================================
n_hours = 8760

# Données bâtiment
solar = np.array(building.energy_simulation.solar_generation[:n_hours])
demand = np.array(building.energy_simulation.non_shiftable_load[:n_hours])
cooling = np.array(building.energy_simulation.cooling_demand[:n_hours])
pricing = np.array(building.pricing.electricity_pricing[:n_hours])
month = np.array(building.energy_simulation.month[:n_hours])

# Données EV (depuis le chargeur)
charger = building.electric_vehicle_chargers[0]
print("Attributs disponibles du chargeur :")
print([a for a in dir(charger) if not a.startswith('_')])

# ============================================
# 3. Filtrer uniquement juin (mois 6)
# ============================================
mask_juin = month == 6

solar_juin    = solar[mask_juin]
demand_juin   = demand[mask_juin]
cooling_juin  = cooling[mask_juin]
pricing_juin  = pricing[mask_juin]

n_juin = len(solar_juin)
heures = np.arange(n_juin)
jours  = heures / 24

print(f"\n Données de juin : {n_juin} heures ({n_juin//24} jours)")

# ============================================
# 4. Points remarquables
# ============================================
idx_pic_pv     = np.argmax(solar_juin)
val_pic_pv     = solar_juin[idx_pic_pv]

idx_pic_demand = np.argmax(demand_juin + cooling_juin)
val_pic_demand = (demand_juin + cooling_juin)[idx_pic_demand]

idx_prix_max   = np.argmax(pricing_juin)
val_prix_max   = pricing_juin[idx_prix_max]

print(f"   Pic de demande totale : {val_pic_demand:.2f} kWh → jour {idx_pic_demand//24+1}, heure {idx_pic_demand%24}h")
print(f"   Pic PV               : {val_pic_pv:.2f} W/kWc → jour {idx_pic_pv//24+1}, heure {idx_pic_pv%24}h")
print(f"   Prix max             : {val_prix_max:.3f} $/kWh → jour {idx_prix_max//24+1}, heure {idx_prix_max%24}h")

# ============================================
# 5. Graphiques
# ============================================
fig, axes = plt.subplots(3, 1, figsize=(13, 10), sharex=True)

# --- Plot 1 : Demande totale (non_shiftable + cooling) ---
axes[0].plot(jours, demand_juin, color='#d62728', linewidth=1,
             label='Non shiftable load (kWh)')
axes[0].plot(jours, cooling_juin, color='#9467bd', linewidth=1,
             label='Cooling demand (kWh)')
axes[0].plot(jours, demand_juin + cooling_juin, color='black',
             linewidth=1.5, linestyle='--', label='Demande totale (kWh)')
axes[0].scatter(jours[idx_pic_demand], val_pic_demand, color='red',
                zorder=5, s=60, label=f'Pic demande totale ({val_pic_demand:.2f} kWh)')
axes[0].annotate(f'{val_pic_demand:.2f} kWh',
                  (jours[idx_pic_demand], val_pic_demand),
                  textcoords="offset points", xytext=(0, 10),
                  ha='center', fontweight='bold', color='red')
axes[0].set_ylabel('Demande (kWh)')
axes[0].set_title("Demande énergétique — Juin")
axes[0].legend(loc='upper right', fontsize=8)
axes[0].grid(alpha=0.3)

# --- Plot 2 : Production PV ---
axes[1].plot(jours, solar_juin, color='#ff7f0e', linewidth=1,
             label='Production solaire (PV)')
axes[1].fill_between(jours, solar_juin, color='#ff7f0e', alpha=0.2)
axes[1].scatter(jours[idx_pic_pv], val_pic_pv, color='black',
                zorder=5, s=50, label=f'Pic ({val_pic_pv:.2f} W/kWc)')
axes[1].annotate(f'{val_pic_pv:.2f} W/kWc',
                  (jours[idx_pic_pv], val_pic_pv),
                  textcoords="offset points", xytext=(0, 10),
                  ha='center', fontweight='bold')
axes[1].set_ylabel('Production PV (W/kWc)')
axes[1].set_title("Production photovoltaïque — Juin")
axes[1].legend(loc='upper right')
axes[1].grid(alpha=0.3)

# --- Plot 3 : Prix de l'électricité ---
axes[2].plot(jours, pricing_juin, color='#2ca02c', linewidth=1,
             label="Prix de l'électricité")
axes[2].scatter(jours[idx_prix_max], val_prix_max, color='black',
                zorder=5, s=50,
                label=f'Prix max ({val_prix_max:.3f} $/kWh)')
axes[2].annotate(f'{val_prix_max:.3f} $/kWh',
                  (jours[idx_prix_max], val_prix_max),
                  textcoords="offset points", xytext=(0, 10),
                  ha='center', fontweight='bold')
axes[2].set_ylabel('Prix ($/kWh)')
axes[2].set_xlabel("Jour de juin")
axes[2].set_title("Prix de l'électricité — Juin")
axes[2].legend(loc='upper right')
axes[2].grid(alpha=0.3)

plt.tight_layout()

chemin = os.path.join(os.getcwd(), "analyse_juin_ev.png")
plt.savefig(chemin, dpi=150)
print(f"\n Graphique enregistré : {chemin}")

plt.show(block=True)