from citylearn.citylearn import CityLearnEnv
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np
import os
 
# ============================================
# 1. Charger l'environnement et un bâtiment
# ============================================
env = CityLearnEnv('citylearn_challenge_2023_phase_1')
building = env.buildings[0]
 
# ============================================
# 2. Trouver un mois d'été (juin=6, juillet=7, aout=8)
# ============================================
month_array = np.array(building.energy_simulation.month)
summer_mask = np.isin(month_array, [6, 7, 8])
summer_indices = np.where(summer_mask)[0]
 
# On prend le premier mois d'été complet trouvé (~30 jours = 720h)
start = summer_indices[0]
n_hours = 24 * 30
end = start + n_hours
 
# ============================================
# 3. Récupérer les données (TÂCHE 1 : demande totale avec cooling)
# ============================================
non_shiftable_load = np.array(building.energy_simulation.non_shiftable_load[start:end])
cooling_demand = np.array(building.energy_simulation.cooling_demand[start:end])
 
# Demande totale = charge fixe + climatisation (été = forte clim)
demand_total = non_shiftable_load + cooling_demand
 
# ============================================
# 4. PV ajusté à une capacité réelle (TÂCHE 2)
# ============================================
# solar_generation est en W/kW (production de 1 kWc), PAS en kWh -> voir data.py
solar_per_kwc = np.array(building.energy_simulation.solar_generation[start:end])  # W/kW
 
# Choix de la capacité installée (à adapter selon les besoins du bâtiment)
capacite_kwc = 5  # exemple : 5 kWc installés
solar_kwc_installes = solar_per_kwc * capacite_kwc  # W/kW * kWc -> W (puissance produite)
 
pricing = np.array(building.pricing.electricity_pricing[start:end])
 
heures = np.arange(n_hours)
jours = heures / 24
 
# ============================================
# 5. Points remarquables
# ============================================
idx_pic_demande = np.argmax(demand_total)
val_pic_demande = demand_total[idx_pic_demande]
 
idx_pic_pv = np.argmax(solar_kwc_installes)
val_pic_pv = solar_kwc_installes[idx_pic_pv]
 
idx_prix_max = np.argmax(pricing)
val_prix_max = pricing[idx_prix_max]
 
print(" RÉSULTATS DE L'ANALYSE (1 mois d'été, avec cooling + PV ajusté)")
print(f"  Mois d'été détecté   : mois {month_array[start]}")
print(f"  Pic de demande totale (charge + clim) : {val_pic_demande:.2f} kWh")
print(f"  Pic de production PV ({capacite_kwc} kWc) : {val_pic_pv:.2f} W")
print(f"  Prix maximum         : {val_prix_max:.3f} $/kWh")
 
# ============================================
# 6. Graphique (unités corrigées)
# ============================================
fig, axes = plt.subplots(3, 1, figsize=(13, 10), sharex=True)
 
axes[0].plot(jours, demand_total, color='#d62728', linewidth=1,
             label='Demande totale (non_shiftable_load + cooling_demand)')
axes[0].scatter(jours[idx_pic_demande], val_pic_demande, color='black', zorder=5, s=50,
                label=f'Pic ({val_pic_demande:.2f} kWh)')
axes[0].annotate(f'{val_pic_demande:.2f} kWh', (jours[idx_pic_demande], val_pic_demande),
                  textcoords="offset points", xytext=(0, 10), ha='center', fontweight='bold')
axes[0].set_ylabel('Demande (kWh)')
axes[0].set_title("Demande énergétique totale du bâtiment - été (climatisation incluse)")
axes[0].legend(loc='upper right')
axes[0].grid(alpha=0.3)
 
axes[1].plot(jours, solar_kwc_installes, color='#ff7f0e', linewidth=1,
             label=f'Production PV installée ({capacite_kwc} kWc)')
axes[1].fill_between(jours, solar_kwc_installes, color='#ff7f0e', alpha=0.2)
axes[1].scatter(jours[idx_pic_pv], val_pic_pv, color='black', zorder=5, s=50,
                label=f'Pic ({val_pic_pv:.2f} W)')
axes[1].annotate(f'{val_pic_pv:.2f} W', (jours[idx_pic_pv], val_pic_pv),
                  textcoords="offset points", xytext=(0, 10), ha='center', fontweight='bold')
axes[1].set_ylabel('Production PV (W)')  # correcte car deja convertie, capacité x profil W/kW
axes[1].set_title(f"Production photovoltaïque - installation de {capacite_kwc} kWc")
axes[1].legend(loc='upper right')
axes[1].grid(alpha=0.3)
 
axes[2].plot(jours, pricing, color='#2ca02c', linewidth=1, label="Prix de l'électricité")
axes[2].scatter(jours[idx_prix_max], val_prix_max, color='black', zorder=5, s=50,
                label=f'Prix max ({val_prix_max:.3f} $/kWh)')
axes[2].annotate(f'{val_prix_max:.3f} $/kWh', (jours[idx_prix_max], val_prix_max),
                  textcoords="offset points", xytext=(0, 10), ha='center', fontweight='bold')
axes[2].set_ylabel('Prix ($/kWh)')
axes[2].set_xlabel("Jour du mois d'été")
axes[2].set_title("Prix de l'électricité")
axes[2].legend(loc='upper right')
axes[2].grid(alpha=0.3)
 
plt.tight_layout()
 
chemin_sortie = os.path.join(os.getcwd(), "analyse_mois_ete_corrige.png")
plt.savefig(chemin_sortie, dpi=150)
print(f" Graphique enregistré ici : {chemin_sortie}")
 
plt.show(block=True)