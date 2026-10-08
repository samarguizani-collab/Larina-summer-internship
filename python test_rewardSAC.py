from citylearn.agents.sac import SAC as Agent
from citylearn.citylearn import CityLearnEnv
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import time

# ============================================================
# PARAMÈTRES
# ============================================================
N_STEPS = 24 * 30      # 1 mois = 720 heures
N_EPISODES = 20         # suffisamment d'épisodes pour que SAC apprenne réellement

# ============================================================
# INITIALISATION
# ============================================================
env = CityLearnEnv(
    'citylearn_challenge_2022_phase_1',
    central_agent=False,
    simulation_start_time_step=0,
    simulation_end_time_step=N_STEPS
)

model = Agent(env)
print(f"Nombre de pas de temps par épisode : {env.time_steps}")
print(f"Nombre d'épisodes prévus : {N_EPISODES}")
print(f"Nombre total de pas d'entraînement : {env.time_steps * N_EPISODES}")

# ============================================================
# ENTRAINEMENT épisode par épisode (pour suivre la progression)
# ============================================================
episode_total_rewards = []
start = time.time()

try:
    for e in range(N_EPISODES):
        # deterministic_finish=False pendant l'entraînement pour garder l'exploration
        model.learn(episodes=1, deterministic_finish=False)

        elapsed = time.time() - start
        print(f"Épisode {e+1}/{N_EPISODES} terminé - temps écoulé: {elapsed:.1f}s")

except KeyboardInterrupt:
    print(f"\nEntraînement interrompu manuellement après {time.time()-start:.1f}s.")

print(f"Entraînement total terminé en {time.time()-start:.1f} secondes")

# ============================================================
# TEST FINAL — évaluation avec la politique apprise (déterministe, sans exploration)
# ============================================================
observations, _ = env.reset()
rewards = []

while not env.terminated:
    actions = model.predict(observations, deterministic=True)
    observations, reward, info, terminated, truncated = env.step(actions)
    rewards.append(sum(reward))

# ============================================================
# GRAPHIQUE
# ============================================================
plt.plot(rewards)
plt.title(f'Reward - Décentralisé Indépendant (SAC) - après {N_EPISODES} épisodes d\'entraînement')
plt.xlabel('Time step (heures)')
plt.ylabel('Reward')
plt.savefig('test_reward_sac_entraine.png', dpi=150)
plt.show()
print("Graphique enregistré : test_reward_sac_entraine.png")

# ============================================================
# KPIs 
# ============================================================
try:
    kpis = model.env.evaluate()
    kpis = kpis.pivot(index='cost_function', columns='name', values='value').round(3)
    kpis = kpis.dropna(how='all')
    print("\n=== KPIs finaux ===")
    print(kpis)
except Exception as ex:
    print(f"Impossible de calculer les KPIs : {ex}")