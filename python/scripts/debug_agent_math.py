import sys
import os
import numpy as np
from stable_baselines3 import PPO

# ============================================================================
# KONFIGURACJA
# ============================================================================
path_to_geant4_bin = r"C:\Geant4\install\bin"
if os.name == 'nt' and os.path.exists(path_to_geant4_bin):
    os.add_dll_directory(path_to_geant4_bin)

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '..'))
sys.path.append(project_root)

try:
    from src.simulation import geant4_sim

    print("[System] ✅ Geant4 loaded.")
except ImportError as e:
    print(f"[System] ❌ Error loading Geant4: {e}")
    sys.exit(1)

MODEL_PATH = os.path.join(project_root, "data", "models", "ppo_geant4", "geant4_final")
print(f"[AI] Loading model: {MODEL_PATH}")

try:
    model = PPO.load(MODEL_PATH)
    print("[AI] ✅ Model loaded.")
except Exception as e:
    print(f"[AI] ❌ Error loading model: {e}")
    sys.exit(1)


# ============================================================================
# LOGIKA DEBUGOWANIA
# ============================================================================

def run_debug_session():
    manager = geant4_sim.SimulationManager()

    print("\n--- GENEROWANIE CZĄSTKI GEANT4 (GROUND TRUTH) ---")
    raw = None
    while raw is None or len(raw['x']) < 30:
        raw = manager.run_single()

    steps_real = len(raw['x'])
    print(f"Pobrano cząstkę Geant4: {steps_real} kroków.")

    # Start Agenta
    current_state = np.array([
        raw['x'][0], raw['y'][0], raw['z'][0],
        raw['px'][0], raw['py'][0], raw['pz'][0],
        raw['energy'][0]
    ], dtype=np.float32)

    print("\n--- SYMULACJA AGENTA (do 50 kroków) ---")
    ai_history = []
    limit_steps = 50  # Wymuszamy 50 kroków bez względu na wszystko

    for i in range(limit_steps):
        ai_history.append(current_state.copy())
        action, _ = model.predict(current_state, deterministic=True)
        current_state += action
        # USUNIĘTO WARUNEK STOPU - niech leci gdzie chce

    steps_ai = len(ai_history)

    # Tabela porównawcza
    print("\n" + "=" * 145)
    print(
        f"{'Krok':<4} | {'REAL Pos (X,Y,Z)':<26} | {'AI Pos (X,Y,Z)':<26} | {'AI Action (dX,dY,dZ)':<26} | {'REAL E':<8} | {'AI E':<8}")
    print("=" * 145)

    limit_display = 50

    for i in range(limit_display):
        # Dane Real
        if i < steps_real:
            pos_real = f"({raw['x'][i]:6.3f}, {raw['y'][i]:6.3f}, {raw['z'][i]:6.3f})"
            e_real = f"{raw['energy'][i]:6.3f}"
        else:
            pos_real = "[KONIEC]"
            e_real = "   -   "

        # Dane AI
        if i < steps_ai:
            ax, ay, az = ai_history[i][0], ai_history[i][1], ai_history[i][2]
            ae = ai_history[i][6]
            pos_ai = f"({ax:6.3f}, {ay:6.3f}, {az:6.3f})"
            e_ai = f"{ae:6.3f}"

            if i < steps_ai - 1:
                nx = ai_history[i + 1][0] - ax
                ny = ai_history[i + 1][1] - ay
                nz = ai_history[i + 1][2] - az
                act_str = f"[{nx:6.3f}, {ny:6.3f}, {nz:6.3f}]"
            else:
                act_str = "[OSTATNI]"
        else:
            pos_ai = "-"
            e_ai = "-"
            act_str = "-"

        print(f"{i:<4} | {pos_real:<26} | {pos_ai:<26} | {act_str:<26} | {e_real:<8} | {e_ai:<8}")

    print("=" * 145)

    # Wnioski
    last_ai = ai_history[-1]
    total_vec_ai = last_ai[:3] - ai_history[0][:3]

    print("\n--- WNIOSKI ---")
    print(f"Całkowite przesunięcie agenta: {total_vec_ai}")

    # Sprawdzamy czy idzie do przodu (X > 0)
    if total_vec_ai[0] > 1.0:
        print("✅ Agent poprawnie idzie w głąb fantomu (oś X).")
    elif total_vec_ai[0] < -1.0:
        print("❌ Agent cofa się przed fantom!")
    else:
        print("❌ Agent kręci się w miejscu.")


if __name__ == "__main__":
    run_debug_session()