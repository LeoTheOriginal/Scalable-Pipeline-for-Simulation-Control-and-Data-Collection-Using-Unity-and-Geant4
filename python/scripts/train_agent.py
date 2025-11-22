import sys
import os
import time
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
import numpy as np

# 1. Fix importów (żeby widział src)
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '..'))
sys.path.append(project_root)

from src.training.environment import Geant4ParticleEnv


def train():
    # Katalogi na logi i modele
    models_dir = os.path.join(project_root, "data", "models", "ppo_geant4")
    logs_dir = os.path.join(project_root, "data", "logs")

    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(logs_dir, exist_ok=True)

    print("=== START TRENINGU RL ===")
    print(f"Logi: {logs_dir}")
    print(f"Modele: {models_dir}")

    # 2. Inicjalizacja Środowiska
    try:
        env = Geant4ParticleEnv()
    except Exception as e:
        print(f"❌ Błąd tworzenia środowiska: {e}")
        return

    # 3. Tworzenie Agenta PPO
    # MlpPolicy = Sieć neuronowa typu Multi-Layer Perceptron (standardowa)
    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        tensorboard_log=logs_dir,
        learning_rate=0.0003,
        n_steps=2048,
    )

    # Callback do zapisywania modelu co 10k kroków
    checkpoint_callback = CheckpointCallback(
        save_freq=10000,
        save_path=models_dir,
        name_prefix="geant4_model"
    )

    # 4. Trening (Główna pętla)
    TIMESTEPS = 100_000
    print(f"Rozpoczynanie nauki przez {TIMESTEPS} kroków...")
    start_time = time.time()

    model.learn(total_timesteps=TIMESTEPS, callback=checkpoint_callback)

    end_time = time.time()
    print(f"✅ Trening zakończony w {(end_time - start_time):.2f} sekund.")

    # 5. Zapisz finalny model
    final_path = os.path.join(models_dir, "geant4_final")
    model.save(final_path)
    print(f"Model zapisany w: {final_path}.zip")

    # 6. Szybki test (Ewaluacja)
    print("\n=== TESTOWANIE WYTRENOWANEGO MODELU ===")
    obs, _ = env.reset()
    total_reward = 0
    steps = 0
    done = False

    print("Symulacja jednej trajektorii z użyciem AI...")
    while not done:
        # Model przewiduje akcję (deterministic=True wyłącza losowość)
        action, _states = model.predict(obs, deterministic=True)

        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        steps += 1
        done = terminated or truncated

    print(f"Wynik AI: Kroki={steps}, Nagroda={total_reward:.2f}")
    print("(Dla porównania: Random Agent miał ok. -1100.00)")


if __name__ == "__main__":
    train()