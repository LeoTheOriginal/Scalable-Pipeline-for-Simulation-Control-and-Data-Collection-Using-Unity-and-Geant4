import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

# --- KONFIGURACJA ---
RESULTS_DIR = r"C:\Thesis\unity\GeantML_Test\Assets\Models\results"
OUTPUT_DIR = "plots"
OUTPUT_FILENAME = "reward_variance"

# Konfiguracja serii danych
RUNS_CONFIG = {
    'ppo_base_v1': {'label': 'PPO', 'color': '#1f77b4'},  # Niebieski
    'sac_base_v1': {'label': 'SAC', 'color': '#ff7f0e'},  # Pomarańczowy
    'ppo_lstm_base_v1': {'label': 'PPO+LSTM', 'color': '#2ca02c'}  # Zielony
}

TAG_REWARD = "Environment/Cumulative Reward"

# Parametry okna (Window Size)
# TensorBoard w ML-Agents zapisuje punkty co ok. 10-20k kroków.
# Aby uzyskać "okno 10000" lub sensowne wygładzenie, bierzemy np. 10-20 ostatnich punktów pomiarowych.
ROLLING_WINDOW_POINTS = 20


def get_event_file(run_dir):
    files = glob.glob(os.path.join(run_dir, "**", "events.out.tfevents*"), recursive=True)
    if files:
        return max(files, key=os.path.getsize)
    return None


def main():
    if not os.path.exists(RESULTS_DIR):
        print(f"BŁĄD: Nie znaleziono ścieżki: {RESULTS_DIR}")
        return

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    plt.figure(figsize=(10, 6))
    plt.rcParams.update({'font.size': 12, 'font.family': 'serif'})
    plt.grid(True, which='major', linestyle='--', alpha=0.6)

    found_any = False

    for run_id, config in RUNS_CONFIG.items():
        run_path = os.path.join(RESULTS_DIR, run_id)
        event_file = get_event_file(run_path)

        if not event_file:
            print(f"Pominięto (brak pliku): {run_id}")
            continue

        print(f"Przetwarzanie: {run_id}...")
        try:
            ea = EventAccumulator(event_file, size_guidance={'scalars': 0})
            ea.Reload()

            if TAG_REWARD in ea.Tags()['scalars']:
                events = ea.Scalars(TAG_REWARD)
                steps = [x.step for x in events]
                values = [x.value for x in events]

                # Konwersja na Pandas Series dla łatwego liczenia rolling std
                s = pd.Series(values)

                # Obliczanie Rolling Standard Deviation
                rolling_std = s.rolling(window=ROLLING_WINDOW_POINTS).std()

                # Rysowanie (zaczynamy od punktu, gdzie okno jest pełne)
                plt.plot(steps, rolling_std,
                         label=config['label'],
                         color=config['color'],
                         linewidth=2)
                found_any = True

        except Exception as e:
            print(f"Błąd: {e}")

    if not found_any:
        print("Brak danych do wykresu.")
        return

    plt.title("Reward Variance (Rolling Std Dev)")
    plt.xlabel("Training Steps")
    plt.ylabel("Standard Deviation of Reward")
    plt.legend(loc='upper right', frameon=True)
    plt.xlim(left=0)

    # Formatowanie osi X (1M, 2M...)
    def millions(x, pos):
        return '%1.1fM' % (x * 1e-6)

    from matplotlib.ticker import FuncFormatter
    plt.gca().xaxis.set_major_formatter(FuncFormatter(millions))

    # Zapis
    png_path = os.path.join(OUTPUT_DIR, f"{OUTPUT_FILENAME}.png")
    plt.tight_layout()
    plt.savefig(png_path, dpi=300)

    print("-" * 50)
    print(f"Wygenerowano wykres: {png_path}")
    print("-" * 50)


if __name__ == "__main__":
    main()