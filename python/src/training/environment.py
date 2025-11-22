import gym
from gym import spaces
import numpy as np
import os
import sys

# ============================================================================
# 1. KONFIGURACJA DLL I IMPORT C++
# ============================================================================
# (Kopiujemy ten blok, bo środowisko może być importowane niezależnie od serwera)

# --- ZMIEŃ NA SWOJĄ ŚCIEŻKĘ ---
PATH_TO_GEANT4_BIN = r"C:\Geant4\install\bin"
# ------------------------------

if os.name == 'nt' and os.path.exists(PATH_TO_GEANT4_BIN):
    try:
        os.add_dll_directory(PATH_TO_GEANT4_BIN)
    except Exception:
        pass

# Dodajemy root projektu do sys.path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '../../'))
if project_root not in sys.path:
    sys.path.append(project_root)

try:
    from src.simulation import geant4_sim
except ImportError as e:
    raise ImportError(f"Nie można załadować modułu Geant4. Sprawdź kompilację i DLL. Błąd: {e}")


# ============================================================================
# 2. KLASA ŚRODOWISKA GYM
# ============================================================================

class Geant4ParticleEnv(gym.Env):
    """
    Środowisko RL, w którym agent uczy się przewidywać kolejny krok cząstki.

    Obserwacja (State): [x, y, z, px, py, pz, energy] (7 wartości)
    Akcja (Action):     [dx, dy, dz] (Zmiana pozycji - wektor przesunięcia)
    """

    def __init__(self):
        super(Geant4ParticleEnv, self).__init__()

        # Inicjalizacja silnika C++
        print("[Env] Initializing Geant4 Manager...")
        self.sim_manager = geant4_sim.SimulationManager()

        # Definicja przestrzeni akcji (Agent przewiduje przesunięcie dx, dy, dz)
        # Zakładamy, że w jednym kroku cząstka nie skoczy dalej niż np. 5 cm (to i tak dużo)
        self.action_space = spaces.Box(low=-0.2, high=0.2, shape=(3,), dtype=np.float32)

        # Definicja przestrzeni obserwacji
        # [x, y, z, px, py, pz, energy]
        # Granice ustawiamy szeroko (np. +/- 100 cm, 20 MeV)
        low_obs = np.array([-100, -100, -100, -20, -20, -20, 0], dtype=np.float32)
        high_obs = np.array([100, 100, 100, 20, 20, 20, 20], dtype=np.float32)
        self.observation_space = spaces.Box(low=low_obs, high=high_obs, dtype=np.float32)

        # Bufor na aktualną trajektorię (Teacher Forcing)
        self.current_trajectory = None
        self.current_step_idx = 0
        self.trajectory_length = 0

    def reset(self, seed=None, options=None):
        """
        Rozpoczyna nową 'epizod' = nową cząstkę.
        W rzeczywistości odpala Geant4, pobiera całą trajektorię i zwraca pierwszy punkt.
        """
        super().reset(seed=seed)

        # 1. Uruchom Geant4 (C++)
        # To zwraca dict: {'x': [...], 'y': [...], ...}
        raw_data = self.sim_manager.run_single()

        # 2. Skonwertuj do wygodniejszego formatu (lista punktów)
        # Transpozycja danych do formatu: [[x,y,z,px,py,pz,e], [x,y,z...], ...]
        self.current_trajectory = np.stack([
            raw_data['x'], raw_data['y'], raw_data['z'],
            raw_data['px'], raw_data['py'], raw_data['pz'],
            raw_data['energy']
        ], axis=1).astype(np.float32)

        self.trajectory_length = len(self.current_trajectory)
        self.current_step_idx = 0

        # Zabezpieczenie: Czasem Geant4 zwraca 0 kroków (jeśli cząstka od razu zginie)
        # Wtedy powtarzamy reset aż trafimy na "żywą" cząstkę
        if self.trajectory_length < 2:
            return self.reset(seed=seed)

        # Zwracamy stan początkowy (Krok 0)
        initial_observation = self.current_trajectory[0]
        return initial_observation, {}

    def step(self, action):
        """
        Wykonuje krok w środowisku.
        Action: Przewidywane przesunięcie [dx, dy, dz] przez Agenta.
        """
        # Sprawdzamy, czy nie koniec trajektorii
        if self.current_step_idx >= self.trajectory_length - 1:
            # To nie powinno się zdarzyć, jeśli logika jest ok, ale dla bezpieczeństwa:
            return np.zeros(7, dtype=np.float32), 0.0, True, False, {}

        # 1. Pobieramy PRAWDZIWY stan obecny i następny (Ground Truth)
        current_state_gt = self.current_trajectory[self.current_step_idx]
        next_state_gt = self.current_trajectory[self.current_step_idx + 1]

        # Prawdziwa pozycja i następna pozycja
        current_pos = current_state_gt[:3]
        next_pos_true = next_state_gt[:3]

        # Prawdziwe przesunięcie (to, co Agent powinien był zgadnąć)
        delta_true = next_pos_true - current_pos

        # 2. Obliczamy Nagrodę (Reward)
        # Nagroda to ujemny błąd średniokwadratowy (MSE) lub dystans euklidesowy.
        # Chcemy, żeby Agent (action) był jak najbliżej delta_true.

        # Dystans między predykcją a prawdą
        error_distance = np.linalg.norm(action - delta_true)

        # Formuła nagrody:
        # Im mniejszy błąd, tym bliżej 0. Im większy błąd, tym bardziej ujemna nagroda.
        # Dodajemy mały bonus (+1.0) za przetrwanie kroku, żeby zachęcić do "życia".
        reward = - (error_distance ** 2) + 1.0

        # 3. Przesuwamy symulację
        self.current_step_idx += 1

        # Czy to koniec cząstki?
        terminated = (self.current_step_idx >= self.trajectory_length - 1)
        truncated = False

        # Zwracamy:
        # - next_state_gt: Prawdziwy następny stan (bo uczymy metodą Teacher Forcing -
        #   nawet jak Agent się pomylił, przenosimy go w poprawne miejsce w następnym kroku,
        #   żeby nie błądził w nieskończoność)
        return next_state_gt, float(reward), terminated, truncated, {}

    def close(self):
        pass