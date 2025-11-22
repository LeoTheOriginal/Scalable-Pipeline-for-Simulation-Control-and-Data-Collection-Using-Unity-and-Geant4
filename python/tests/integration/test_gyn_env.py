import sys
import os
import numpy as np
import pytest  # Opcjonalnie, jeśli używasz pytest, ale zadziała też jako zwykły skrypt

# Ustawienie ścieżek: wychodzimy z tests/integration/ do głównego katalogu python/
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '../../'))
sys.path.append(project_root)

from src.training.environment import Geant4ParticleEnv


def test_gym_environment_loop():
    """
    Prosty 'Smoke Test': Uruchamia środowisko na 1 epizod z losowym agentem.
    Sprawdza, czy C++ się nie wywala i czy wymiary danych się zgadzają.
    """
    print("\n=== ROZPOCZYNAM TEST INTEGRACYJNY ŚRODOWISKA GYM ===")

    try:
        print("[Test] Tworzenie instancji Geant4ParticleEnv...")
        env = Geant4ParticleEnv()
    except Exception as e:
        print(f"❌ BŁĄD KRYTYCZNY: Nie udało się zainicjować środowiska.\n{e}")
        return

    print("[Test] Resetowanie środowiska (pierwszy strzał Geant4)...")
    obs, info = env.reset()

    # Weryfikacja wymiarów
    assert obs.shape == (7,), f"Błąd wymiarów obserwacji! Oczekiwano (7,), otrzymano {obs.shape}"
    print(f"✅ Reset OK. Stan początkowy: {obs}")
    print(f"   (x,y,z): {obs[:3]}, Energia: {obs[6]:.2f} MeV")

    total_reward = 0
    steps = 0
    done = False

    print("[Test] Rozpoczynam pętlę symulacji (Random Agent)...")

    while not done:
        # Losowa akcja: Agent strzela na ślepo wektorem przesunięcia
        action = env.action_space.sample()

        # Krok środowiska
        next_obs, reward, terminated, truncated, info = env.step(action)

        # Weryfikacja typów
        assert isinstance(reward, float), "Nagroda musi być floatem!"
        assert next_obs.shape == (7,), "Następna obserwacja ma zły kształt!"

        total_reward += reward
        steps += 1
        done = terminated or truncated

        if steps % 10 == 0:
            print(f"   -> Krok {steps}: Reward={reward:.4f}, Następna poz Z={next_obs[2]:.2f}")

    print(f"\n✅ TEST ZAKOŃCZONY SUKCESEM!")
    print(f"   Wykonano kroków: {steps}")
    print(f"   Całkowita nagroda: {total_reward:.2f}")
    print("====================================================")


if __name__ == "__main__":
    test_gym_environment_loop()