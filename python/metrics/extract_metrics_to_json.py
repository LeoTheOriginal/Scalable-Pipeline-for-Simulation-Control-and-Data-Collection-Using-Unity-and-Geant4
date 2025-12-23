import os
import glob
import json
import numpy as np
import pandas as pd
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

# --- KONFIGURACJA ---

# 1. SKĄD CZYTAMY (Ścieżka do logów TensorBoard - stała)
INPUT_RESULTS_DIR = r"C:\Thesis\unity\GeantML_Test\Assets\Models\results"

# 2. GDZIE ZAPISUJEMY (Folder lokalny tam, gdzie uruchamiasz skrypt)
OUTPUT_SUBDIR = "json"
OUTPUT_FILENAME = "training_metrics.json"

# Parametry
SMOOTHING_WINDOW = 50

# Tagi
TAGS = {
    "Reward": "Environment/Cumulative Reward",
    "Entropy": "Policy/Entropy",
    "ValueLoss": "Losses/Value Loss",
    "QLoss": "Losses/Q1 Loss",
    "Length": "Environment/Episode Length"
}


class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer): return int(obj)
        if isinstance(obj, np.floating): return float(obj)
        if isinstance(obj, np.ndarray): return obj.tolist()
        return super(NpEncoder, self).default(obj)


def get_event_file(run_dir):
    # Szukamy plików tfevents rekurencyjnie w danym runie
    files = glob.glob(os.path.join(run_dir, "**", "events.out.tfevents*"), recursive=True)
    if files:
        return max(files, key=os.path.getsize)
    return None


def extract_run_data(run_name, file_path):
    print(f"-> Analiza: {run_name}")
    try:
        ea = EventAccumulator(file_path, size_guidance={'scalars': 0})
        ea.Reload()
    except Exception as e:
        print(f"   [BŁĄD] {e}")
        return None

    available_tags = ea.Tags()['scalars']

    metrics = {
        "run_id": run_name,
        "final_reward_mean": 0.0,
        "final_reward_std": 0.0,
        "reward_variance_last_10pct": 0.0,
        "steps_to_50pct": None,
        "steps_to_90pct": None,
        "final_entropy": 0.0,
        "final_loss": 0.0,
        "loss_type": "None"
    }

    # REWARD
    if TAGS["Reward"] in available_tags:
        events = ea.Scalars(TAGS["Reward"])
        steps = np.array([x.step for x in events])
        values = np.array([x.value for x in events])

        metrics["final_reward_mean"] = np.mean(values[-SMOOTHING_WINDOW:])
        metrics["final_reward_std"] = np.std(values[-SMOOTHING_WINDOW:])

        tail_idx = int(len(values) * 0.9)
        metrics["reward_variance_last_10pct"] = np.var(values[tail_idx:])

        max_val = np.max(values)
        metrics["steps_to_50pct"] = next((int(s) for s, v in zip(steps, values) if v >= 0.5 * max_val), None)
        metrics["steps_to_90pct"] = next((int(s) for s, v in zip(steps, values) if v >= 0.9 * max_val), None)

    # ENTROPY
    if TAGS["Entropy"] in available_tags:
        events = ea.Scalars(TAGS["Entropy"])
        metrics["final_entropy"] = np.mean([x.value for x in events[-SMOOTHING_WINDOW:]])

    # LOSS
    loss_tag = None
    if TAGS["ValueLoss"] in available_tags:
        loss_tag = TAGS["ValueLoss"]
        metrics["loss_type"] = "Value Loss"
    elif TAGS["QLoss"] in available_tags:
        loss_tag = TAGS["QLoss"]
        metrics["loss_type"] = "Q1 Loss"

    if loss_tag:
        events = ea.Scalars(loss_tag)
        metrics["final_loss"] = np.mean([x.value for x in events[-SMOOTHING_WINDOW:]])

    return metrics


def main():
    if not os.path.exists(INPUT_RESULTS_DIR):
        print(f"BŁĄD: Nie znaleziono folderu wejściowego: {INPUT_RESULTS_DIR}")
        return

    # Tworzenie folderu wyjściowego w BIEŻĄCYM katalogu
    current_dir = os.getcwd()
    output_dir = os.path.join(current_dir, OUTPUT_SUBDIR)
    os.makedirs(output_dir, exist_ok=True)

    output_file_path = os.path.join(output_dir, OUTPUT_FILENAME)

    print(f"1. Czytam logi z: {INPUT_RESULTS_DIR}")
    print(f"2. Zapiszę wynik do: {output_file_path}")
    print("-" * 50)

    # Skanowanie
    run_folders = [f.path for f in os.scandir(INPUT_RESULTS_DIR) if f.is_dir()]
    all_data = []

    for run_folder in run_folders:
        run_name = os.path.basename(run_folder)
        # Pomijamy folder json jeśli akurat tam jest
        if run_name == OUTPUT_SUBDIR: continue

        event_file = get_event_file(run_folder)
        if event_file:
            data = extract_run_data(run_name, event_file)
            if data:
                all_data.append(data)

    if not all_data:
        print("Nie znaleziono danych.")
        return

    # Zapis
    with open(output_file_path, 'w', encoding='utf-8') as f:
        json.dump(all_data, f, cls=NpEncoder, indent=4)

    print("-" * 50)
    print(f"GOTOWE. Utworzono plik: {OUTPUT_SUBDIR}/{OUTPUT_FILENAME}")

    # Podgląd
    df = pd.DataFrame(all_data)
    print("\nPodgląd tabeli:")
    print(df[["run_id", "final_reward_mean", "final_loss"]].to_string(index=False))


if __name__ == "__main__":
    main()