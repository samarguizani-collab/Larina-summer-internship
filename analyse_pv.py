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

n_hours = 8760  # 1 année complète, nécessaire pour analyser les saisons
solar = np.array(building.energy_simulation.solar_generation[:n_hours])
month = np.array(building.energy_simulation.month[:n_hours])
hour_of_day = np.array(building.energy_simulation.hour[:n_hours])

heures = np.arange(n_hours)
jours = heures / 24

# ============================================
# 2. Pic de production PV (maximum absolu)
# ============================================
idx_pic_pv = np.argmax(solar)
val_pic_pv = solar[idx_pic_pv]
mois_pic = month[idx_pic_pv]
heure_pic = hour_of_day[idx_pic_pv]

# ============================================
# 3. Quelle saison est la plus favorable ?
#    -> on calcule la production moyenne par mois
# ============================================
production_par_mois = {}
for m in range(1, 13):
    valeurs_du_mois = solar[month == m]
    if len(valeurs_du_mois) > 0:
        production_par_mois[m] = valeurs_du_mois.mean()

mois_noms = ['Jan', 'Fév', 'Mar', 'Avr', 'Mai', 'Jun',
             'Jul', 'Aoû', 'Sep', 'Oct', 'Nov', 'Déc']

mois_meilleur = max(production_par_mois, key=production_par_mois.get)
mois_pire = min(production_par_mois, key=production_par_mois.get)

# ============================================
# 4. Durée moyenne de production par jour
#    -> nombre d'heures où solar_generation > 0, en moyenne par jour
# ============================================
production_active = solar > 0  # True quand il y a de la production
heures_actives_par_jour = []
for j in range(n_hours // 24):
    jour_data = production_active[j*24:(j+1)*24]
    heures_actives_par_jour.append(jour_data.sum())

duree_moyenne = np.mean(heures_actives_par_jour)

# ============================================
# 5. Afficher les résultats dans le terminal
# ============================================
print("=" * 55)
print(" ANALYSE DE LA PRODUCTION PHOTOVOLTAÏQUE (PV)")
print("=" * 55)
print(f"\n PIC DE PRODUCTION MAXIMUM")
print(f"   Valeur  : {val_pic_pv:.2f} kWh")
print(f"   Date    : mois {int(mois_pic)} ({mois_noms[int(mois_pic)-1]}), à {int(heure_pic)}h")

print(f"\n SAISON LA PLUS FAVORABLE")
print(f"   Meilleur mois : {mois_noms[mois_meilleur-1]} (production moyenne = {production_par_mois[mois_meilleur]:.2f} kWh)")
print(f"   Mois le plus faible : {mois_noms[mois_pire-1]} (production moyenne = {production_par_mois[mois_pire]:.2f} kWh)")

print(f"\n DURÉE DE PRODUCTION")
print(f"   Durée moyenne de production par jour : {duree_moyenne:.1f} heures/jour")
print("=" * 55)

# ============================================
# 6. Graphique 1 : Courbe annuelle avec le pic
# ============================================
plt.figure(figsize=(14, 4))
plt.plot(jours, solar, color='#ff7f0e', linewidth=0.5)
plt.fill_between(jours, solar, color='#ff7f0e', alpha=0.2)
plt.scatter(jours[idx_pic_pv], val_pic_pv, color='black', zorder=5, s=50,
            label=f'Pic max : {val_pic_pv:.2f} kWh')
plt.annotate(f'{val_pic_pv:.2f} kWh', (jours[idx_pic_pv], val_pic_pv),
             textcoords="offset points", xytext=(0, 10), ha='center', fontweight='bold')
plt.xlabel("Jour de l'année")
plt.ylabel('Production PV (kWh)')
plt.title('Production photovoltaïque sur 1 an')
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
chemin1 = os.path.join(os.getcwd(), "pv_annee.png")
plt.savefig(chemin1, dpi=150)

# ============================================
# 7. Graphique 2 : Production moyenne par mois (saisonnalité)
# ============================================
plt.figure(figsize=(10, 5))
mois_list = list(production_par_mois.keys())
valeurs_list = [production_par_mois[m] for m in mois_list]
couleurs = ['#2ca02c' if m == mois_meilleur else ('#d62728' if m == mois_pire else '#ff7f0e')
            for m in mois_list]
plt.bar([mois_noms[m-1] for m in mois_list], valeurs_list, color=couleurs)
plt.ylabel('Production PV moyenne (kWh)')
plt.title('Production PV moyenne par mois (saisonnalité)')
plt.grid(alpha=0.3, axis='y')
plt.tight_layout()
chemin2 = os.path.join(os.getcwd(), "pv_saisonnalite.png")
plt.savefig(chemin2, dpi=150)

# ============================================
# 8. Graphique 3 : Profil journalier moyen (durée de production)
# ============================================
profil_horaire = []
for h in range(1, 25):
    profil_horaire.append(solar[hour_of_day == h].mean())

plt.figure(figsize=(10, 5))
plt.bar(range(1, 25), profil_horaire, color='#ff7f0e', alpha=0.7)
plt.axhline(0, color='gray', linewidth=0.5)
plt.xlabel('Heure de la journée')
plt.ylabel('Production PV moyenne (kWh)')
plt.title(f"Profil journalier moyen de production PV (durée moyenne : {duree_moyenne:.1f}h/jour)")
plt.xticks(range(1, 25))
plt.grid(alpha=0.3, axis='y')
plt.tight_layout()
chemin3 = os.path.join(os.getcwd(), "pv_profil_journalier.png")
plt.savefig(chemin3, dpi=150)

print(f"\n3 graphiques enregistrés :")
print(f"   - {chemin1}")
print(f"   - {chemin2}")
print(f"   - {chemin3}")

plt.show(block=True)