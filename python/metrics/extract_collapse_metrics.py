import os
import glob
import json
import numpy as np
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

# --- KONFIGURACJA ---
RESULTS_DIR = r"C:\Thesis\unity\GeantML_Test\Assets\Models\results"
OUTPUT_FILENAME = "collapse_metrics.json"

TAGS = {
    "Curiosity": "Losses/Curiosity Forward Loss",
    "EpisodeLength": "Environment/Episode Length",
    "Entropy": "Policy/Entropy"
}


def get_event_file(run_dir):
    files = glob.glob(os.path.join(run_dir, "**", "events.out.tfevents*"), recursive=True)
    return max(files, key=os.path.getsize) if files else None


def analyze_run(run_name, file_path):
    print(f"Analiza: {run_name}...")
    try:
        ea = EventAccumulator(file_path, size_guidance={'scalars': 0})
        ea.Reload()
    except:
        return None

    tags = ea.Tags()['scalars']
    metrics = {"run_id": run_name}

    # 1. ENTROPY (Min in first 500k steps)
    if TAGS["Entropy"] in tags:
        events = ea.Scalars(TAGS["Entropy"])
        # Filtrujemy kroki < 500,000
        early_values = [e.value for e in events if e.step < 500000]
        if early_values:
            metrics["min_entropy_500k"] = min(early_values)
        else:
            metrics["min_entropy_500k"] = "N/A"

    # 2. CURIOSITY (Final value)
    if TAGS["Curiosity"] in tags:
        events = ea.Scalars(TAGS["Curiosity"])
        # Średnia z ostatnich 50 punktów
        metrics["final_curiosity"] = np.mean([e.value for e in events[-50:]])
    else:
        metrics["final_curiosity"] = "N/A"

    # 3. EPISODE LENGTH (Trend)
    if TAGS["EpisodeLength"] in tags:
        events = ea.Scalars(TAGS["EpisodeLength"])
        values = [e.value for e in events]
        start_val = np.mean(values[:20])
        end_val = np.mean(values[-20:])
        max_val = max(values)

        if end_val > start_val * 1.1:
            trend = "Increasing"
        elif end_val < start_val * 0.9:
            # Jeśli spadło, sprawdzamy czy to nie "collapse" (blisko 0)
            if end_val < 20:
                trend = "COLLAPSE"
            else:
                trend = "Optimizing"
        else:
            trend = "Stable"

        metrics["length_trend"] = f"{trend} ({int(start_val)}->{int(end_val)})"

    return metrics


def main():
    runs = [f.path for f in os.scandir(RESULTS_DIR) if f.is_dir()]
    results = []

    for run in runs:
        if "json" in run: continue
        evt = get_event_file(run)
        if evt:
            data = analyze_run(os.path.basename(run), evt)
            if data: results.append(data)

    # Wyświetl tabelkę w konsoli
    print("\n" + "=" * 60)
    print(f"{'RUN':<20} | {'MIN ENTROPY':<12} | {'CURIOSITY':<10} | {'LENGTH TREND'}")
    print("-" * 60)
    for r in results:
        ent = r.get('min_entropy_500k', 0)
        if isinstance(ent, float): ent = f"{ent:.3f}"

        cur = r.get('final_curiosity', 0)
        if isinstance(cur, float): cur = f"{cur:.4f}"

        print(f"{r['run_id']:<20} | {ent:<12} | {cur:<10} | {r.get('length_trend', '?')}")
    print("=" * 60)


if __name__ == "__main__":
    main()