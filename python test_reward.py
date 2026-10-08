from citylearn.agents.base import BaselineAgent as Agent
from citylearn.citylearn import CityLearnEnv
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

# ============================================================
# PARAMÈTRES (identiques au test SAC pour comparaison équitable)
# ============================================================
N_STEPS = 24 * 30     # 1 mois = 720 heures

# ============================================================
# INITIALISATION
# ============================================================
env = CityLearnEnv(
    'citylearn_challenge_2022_phase_1',
    central_agent=True,
    simulation_start_time_step=0,
    simulation_end_time_step=N_STEPS
)
model = Agent(env)

print(f"Nombre de pas de temps par épisode : {env.time_steps}")

# ============================================================
# TEST
# ============================================================
observations, _ = env.reset()
rewards = []

while not env.terminated:
    actions = model.predict(observations)
    observations, reward, info, terminated, truncated = env.step(actions)
    rewards.append(sum(reward))

# ============================================================
# GRAPHIQUE
# ============================================================
plt.plot(rewards)
plt.title('Reward calculé par CityLearn - code de base (BaselineAgent) - 1 mois')
plt.xlabel('Time step (heures)')
plt.ylabel('Reward')
plt.savefig('test_reward_code_de_base_mois.png', dpi=150)
plt.show()
print("Graphique enregistré : test_reward_code_de_base_mois.png")