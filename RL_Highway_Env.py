import numpy as np
import gymnasium as gym
import highway_env
from highway_env.envs.highway_env import HighwayEnv
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.evaluation import evaluate_policy
import torch
from stable_baselines3.common.utils import LinearSchedule
from stable_baselines3.common.callbacks import EvalCallback

class PersonalHighwayEnv(HighwayEnv):
    def _rewards(self, action: np.ndarray) -> dict:
        
        forward_speed = self.vehicle.speed
        scaled_speed = highway_env.utils.lmap(forward_speed, self.config["reward_speed_range"], [0, 1])
        speed_reward = np.clip(scaled_speed, 0, 1)

        neighbours = self.road.network.all_side_lanes(self.vehicle.lane_index)
        lane = self.vehicle.lane_index[2] 
        right_lane_reward = lane / max(len(neighbours) - 1, 1)

        headway_penalty = 0.0
        front_vehicle, _ = self.road.neighbour_vehicles(self.vehicle, lane_index=self.vehicle.lane_index)
        if front_vehicle:
            distance = front_vehicle.position[0] - self.vehicle.position[0]
            
            if 0 < distance < 20.0:
                headway_penalty = 1.0 - (distance / 20.0) 

        steering_penalty = abs(action[1])

        return {
            "speed_reward": speed_reward,
            "right_lane_reward": right_lane_reward,
            "headway_penalty": headway_penalty,
            "steering_penalty": steering_penalty,
            "collision_reward": float(self.vehicle.crashed),
        }

    def _reward(self, action: np.ndarray) -> float:
        
        rewards = self._rewards(action)
        
        reward = sum(
            self.config.get(name + "_weight", 0) * value
            for name, value in rewards.items()
        )
        # Rewrite the reward if the vehicle has crashed, nothing else matters if the vehicle has crashed
        if self.vehicle.crashed:
            reward = self.config["collision_reward_weight"]
            
        return reward

    def _is_terminated(self) -> bool:
        return (self.vehicle.crashed) or (not self.vehicle.on_road)


gym.register(id="PersonalHighwayEnv", entry_point="__main__:PersonalHighwayEnv")

config = {
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
    "vehicles_count": 15,
    "controlled_vehicles": 1,
    "initial_lane_id": None,
    "duration": 40,
    "vehicles_density": 1.2, 

    # REWARD SHAPING WEIGHTS
    "speed_reward_weight": 1.0,         
    "right_lane_reward_weight": 0.1,    
    "headway_penalty_weight": -0.5,   # For not leaving a safe distance from the vehicle in front 
    "steering_penalty_weight": -0.05,   # AVoiding zig-zagging 
    "collision_reward_weight": -5.0,  
    
    "reward_speed_range": [20, 35],
    
    "simulation_frequency": 10,
    "policy_frequency": 5,
    "other_vehicles_type": "highway_env.vehicle.behavior.IDMVehicle",
}

def make_env():
    def _init():
        env = gym.make("PersonalHighwayEnv", config=config)
        env = Monitor(env)
        env.reset()
        return env
    return _init

if __name__ == "__main__":
    n_envs = 6 
    env = SubprocVecEnv([make_env() for _ in range(n_envs)])

    policy_kwargs = dict(
        net_arch=dict(pi=[128, 128], vf=[128, 128]), 
        activation_fn=torch.nn.ReLU,
    )

    modelo = PPO(
        "MlpPolicy",
        env,
        device="cpu", 
        verbose=1,
        learning_rate=LinearSchedule(3e-4, 1e-5, 1.0),
        n_steps=2048,
        batch_size=256,
        n_epochs=10,
        gamma=0.99,        
        ent_coef=0, 
        policy_kwargs=policy_kwargs,
        tensorboard_log="./graficas_autopista/",
    )

    print("Iniciando entrenamiento:")
    modelo.learn(total_timesteps=1_000_000, progress_bar=True) 
    modelo.save("coche_autonomo_ppo")


    env.close()

    # --- SHOW THE RESULTS ---
    env_eval = gym.make("PersonalHighwayEnv", render_mode="human", config=config)
    model = PPO.load("coche_autonomo_ppo")

    while True:
        done = truncated = False
        obs, info = env_eval.reset()
        while not (done or truncated):
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, done, truncated, info = env_eval.step(action)
            env_eval.render()