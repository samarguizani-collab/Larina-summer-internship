from citylearn.citylearn import CityLearnEnv

env = CityLearnEnv('citylearn_challenge_2022_phase_1')
observations, _ = env.reset()

print("✅ CityLearn fonctionne !")
print(f"Nombre de bâtiments : {len(env.buildings)}")
print(f"Nombre de pas de temps : {env.time_steps}")
