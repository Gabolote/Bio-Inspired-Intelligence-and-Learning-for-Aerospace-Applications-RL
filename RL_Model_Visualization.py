import gymnasium as gym
import highway_env
from stable_baselines3 import DQN, PPO
from RL_Highway_Env import PersonalHighwayEnv, config


# --- CONFIGURACIÓN CLAVE ---
config2 = {
    "observation": {"type": "Kinematics", "vehicles_count": 6, "features": ["x", "y", "vx", "vy"]},
    "action": {
        "type": "ContinuousAction",
        "acceleration_range": (-3.0, 3.0),
        "steering_range": (-0.05, 0.05), 
        "speed_range": (15, 35),
        "longitudinal": True,
        "lateral": True,
        "clip": True,
    },
    "lanes_count": 3,
    "vehicles_count": 50, 
    "controlled_vehicles": 1,
    "initial_lane_id": None,
    "duration": 40,
    "vehicles_density": 1,
    
    # PESOS DE RECOMPENSA 
    "speed_reward_weight": 1.0,         
    "right_lane_reward_weight": 0.1,    
    "headway_penalty_weight": -0.5,     
    "steering_penalty_weight": -0.05,   
    "collision_reward_weight": -5.0,   
    
    "reward_speed_range": [20, 35], 
    
    "simulation_frequency": 10,
    "policy_frequency": 5,
    "other_vehicles_type": "highway_env.vehicle.behavior.IDMVehicle",
}


env = gym.make("PersonalHighwayEnv", render_mode="human", config=config2)
model = PPO.load("Modelo_gamma99", env=env)
#env = gym.make("highway-fast-v0", render_mode="human")
#model = DQN.load("highway_dqn/model")

for i in range(20):
    done = truncated = False
    obs, info = env.reset()
    step = 0
    while not (done or truncated):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, truncated, info = env.step(action)
        print(f"step={step} accel_action={action[0]:.2f} steer_action={action[1]:.2f} "
              f"speed_x={env.unwrapped.vehicle.velocity[0]:.2f} reward={reward:.3f}")
        env.render()
        step += 1
