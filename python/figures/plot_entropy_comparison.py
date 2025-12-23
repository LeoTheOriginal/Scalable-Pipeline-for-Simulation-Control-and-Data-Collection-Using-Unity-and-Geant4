import os
import glob
import numpy as np
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

# --- KONFIGURACJA ---
RESULTS_DIR = r"C:\Thesis\unity\GeantML_Test\Assets\Models\results"
OUTPUT_DIR = "plots"
OUTPUT_FILENAME = "entropy_comparison"

# Mapowanie: Nazwa folderu na dysku -> Nazwa w Legendzie + Kolor
# Upewnij się, że klucze (np. 'ppo_base_v1') dokładnie pasują do nazw folderów w results!
RUNS_CONFIG = {
    'ppo_base_v1': {'label': 'PPO', 'color': '#1f77b4'},  # Niebieski
    'sac_base_v1': {'label': 'SAC', 'color': '#ff7f0e'},  # Pomarańczowy
    'ppo_lstm_base_v1': {'label': 'PPO+LSTM', 'color': '#2ca02c'}  # Zielony
}

TAG_ENTROPY = "Policy/Entropy"
SMOOTHING = 0.95  # Wygładzanie wykresu (0 = brak, 0.99 = mocne)


def smooth(scalars, weight):
    """Funkcja wygładzająca (Exponential Moving Average) - jak w TensorBoard"""
    last = scalars[0]
    smoothed = []
    for point in scalars:
        smoothed_val = last * weight + (1 - weight) * point
        smoothed.append(smoothed_val)
        last = smoothed_val
    return smoothed


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

    # Konfiguracja wykresu
    plt.figure(figsize=(10, 6))
    plt.rcParams.update({'font.size': 12, 'font.family': 'serif'})
    plt.grid(True, which='major', linestyle='--', alpha=0.6)
    plt.grid(True, which='minor', linestyle=':', alpha=0.3)

    # Iteracja po zdefiniowanych runach
    found_any = False

    for run_id, config in RUNS_CONFIG.items():
        run_path = os.path.join(RESULTS_DIR, run_id)
        event_file = get_event_file(run_path)

        if not event_file:
            print(f"OSTRZEŻENIE: Nie znaleziono danych dla {run_id}")
            continue

        print(f"Wczytywanie: {run_id}...")
        try:
            ea = EventAccumulator(event_file, size_guidance={'scalars': 0})
            ea.Reload()

            if TAG_ENTROPY in ea.Tags()['scalars']:
                events = ea.Scalars(TAG_ENTROPY)
                steps = [x.step for x in events]
                values = [x.value for x in events]

                # Wygładzanie
                if SMOOTHING > 0:
                    values = smooth(values, SMOOTHING)

                # Rysowanie
                plt.plot(steps, values,
                         label=config['label'],
                         color=config['color'],
                         linewidth=2)
                found_any = True
            else:
                print(f" -> Brak tagu {TAG_ENTROPY}")

        except Exception as e:
            print(f" -> Błąd: {e}")

    if not found_any:
        print("Nie udało się narysować żadnych danych.")
        return

    # Wykończenie wykresu
    plt.title("Policy Entropy Evolution During Training")
    plt.xlabel("Training Steps")
    plt.ylabel("Entropy")
    plt.legend(loc='best', frameon=True)
    plt.xlim(left=0)

    # Opcjonalnie: formatowanie osi X (np. 1M zamiast 1000000)
    def millions(x, pos):
        return '%1.1fM' % (x * 1e-6)

    from matplotlib.ticker import FuncFormatter
    plt.gca().xaxis.set_major_formatter(FuncFormatter(millions))

    # Zapis
    png_path = os.path.join(OUTPUT_DIR, f"{OUTPUT_FILENAME}.png")
    pdf_path = os.path.join(OUTPUT_DIR, f"{OUTPUT_FILENAME}.pdf")

    plt.tight_layout()
    plt.savefig(png_path, dpi=300)
    plt.savefig(pdf_path)

    print("-" * 50)
    print(f"SUKCES! Wykres zapisano w:\n{png_path}")
    print("-" * 50)
    # plt.show() # Odkomentuj, jeśli chcesz zobaczyć okno z wykresem


if __name__ == "__main__":
    main()