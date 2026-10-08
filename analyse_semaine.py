from citylearn.citylearn import CityLearnEnv
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np
import os

# ============================================
# 1. Charger l'environnement et un bâtiment
# ============================================
env = CityLearnEnv('citylearn_challenge_2022_phase_1')
building = env.buildings[0]

# Récupérer les données sur 1 semaine (168 heures)
n_hours = 24 * 7

demand = np.array(building.energy_simulation.non_shiftable_load[:n_hours])
solar = np.array(building.energy_simulation.solar_generation[:n_hours])
pricing = np.array(building.pricing.electricity_pricing[:n_hours])

heures = np.arange(n_hours)

# ============================================
# 2. Calculer les points remarquables
# ============================================
idx_pic_demande = np.argmax(demand)
val_pic_demande = demand[idx_pic_demande]

idx_pic_pv = np.argmax(solar)
val_pic_pv = solar[idx_pic_pv]

idx_prix_max = np.argmax(pricing)
val_prix_max = pricing[idx_prix_max]

print(" RÉSULTATS DE L'ANALYSE (1 semaine)")
print(f"   Pic de demande      : {val_pic_demande:.2f} kWh  → jour {idx_pic_demande//24 + 1}, heure {idx_pic_demande%24}h")
print(f"   Pic de production PV : {val_pic_pv:.2f} kWh  → jour {idx_pic_pv//24 + 1}, heure {idx_pic_pv%24}h")
print(f"  Prix maximum         : {val_prix_max:.3f} $/kWh → jour {idx_prix_max//24 + 1}, heure {idx_prix_max%24}h")

# ============================================
# 3. Graphique avec 3 sous-plots, chacun avec son point remarquable
# ============================================
fig, axes = plt.subplots(3, 1, figsize=(13, 10), sharex=True)

# --- Plot 1 : Demande ---
axes[0].plot(heures, demand, color='#d62728', linewidth=1.5, label='Demande (non_shiftable_load)')
axes[0].scatter(idx_pic_demande, val_pic_demande, color='black', zorder=5, s=50,
                label=f'Pic ({val_pic_demande:.2f} kWh)')
axes[0].annotate(f'{val_pic_demande:.2f} kWh', (idx_pic_demande, val_pic_demande),
                  textcoords="offset points", xytext=(0, 10), ha='center', fontweight='bold')
axes[0].set_ylabel('Demande (kWh)')
axes[0].set_title("Demande énergétique du bâtiment (1 semaine)")
axes[0].legend(loc='upper right')
axes[0].grid(alpha=0.3)

# --- Plot 2 : Production PV ---
axes[1].plot(heures, solar, color='#ff7f0e', linewidth=1.5, label='Production solaire (PV)')
axes[1].fill_between(heures, solar, color='#ff7f0e', alpha=0.2)
axes[1].scatter(idx_pic_pv, val_pic_pv, color='black', zorder=5, s=50,
                label=f'Pic ({val_pic_pv:.2f} kWh)')
axes[1].annotate(f'{val_pic_pv:.2f} kWh', (idx_pic_pv, val_pic_pv),
                  textcoords="offset points", xytext=(0, 10), ha='center', fontweight='bold')
axes[1].set_ylabel('Production PV (kWh)')
axes[1].set_title("Production photovoltaïque (1 semaine)")
axes[1].legend(loc='upper right')
axes[1].grid(alpha=0.3)

# --- Plot 3 : Prix de l'électricité ---
axes[2].plot(heures, pricing, color='#2ca02c', linewidth=1.5, label="Prix de l'électricité")
axes[2].scatter(idx_prix_max, val_prix_max, color='black', zorder=5, s=50,
                label=f'Prix max ({val_prix_max:.3f} $/kWh)')
axes[2].annotate(f'{val_prix_max:.3f} $/kWh', (idx_prix_max, val_prix_max),
                  textcoords="offset points", xytext=(0, 10), ha='center', fontweight='bold')
axes[2].set_ylabel('Prix ($/kWh)')
axes[2].set_xlabel("Heure (sur 1 semaine)")
axes[2].set_xticks(range(0, n_hours + 1, 24))
axes[2].set_title("Prix de l'électricité (1 semaine)")
axes[2].legend(loc='upper right')
axes[2].grid(alpha=0.3)

plt.tight_layout()

# ============================================
# 4. Sauvegarder ET afficher la fenêtre
# ============================================
chemin_sortie = os.path.join(os.getcwd(), "analyse_semaine.png")
plt.savefig(chemin_sortie, dpi=150)
print(f" Graphique enregistré ici : {chemin_sortie}")

plt.show(block=True)