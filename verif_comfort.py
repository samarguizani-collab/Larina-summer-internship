from citylearn.citylearn import CityLearnEnv
from citylearn.reward_function import RewardFunction

env = CityLearnEnv(
    'citylearn_challenge_2023_phase_2_local_evaluation',
    central_agent=True,
    reward_function=RewardFunction,
)

b = env.buildings[0]
print(f"Bâtiment : {b.name}")
print(f"cooling_device : {'OK' if getattr(b, 'cooling_device', None) else 'ABSENT'}")
print(f"heating_device : {'OK' if getattr(b, 'heating_device', None) else 'ABSENT'}")
print(f"dhw_device     : {'OK' if getattr(b, 'dhw_device',     None) else 'ABSENT'}")

# Observer les clés disponibles
observations, _ = env.reset()
print("\nType observations[0] :", type(observations[0]))
print("Nombre d'observations :", len(observations))

# Noms officiels des observations
print("\nNoms des observations :", env.observation_names[0])

# Chercher les clés liées à la température
if isinstance(observations[0], dict):
    print("\nClés température :", [k for k in observations[0].keys() if 'temp' in k.lower()])