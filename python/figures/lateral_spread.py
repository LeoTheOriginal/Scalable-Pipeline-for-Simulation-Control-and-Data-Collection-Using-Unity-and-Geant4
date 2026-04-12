import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import numpy as np
import os

# =============================================================================
# KONFIGURACJA
# =============================================================================

# Plik z surowymi danymi (X,Y,Z) z funkcji ExportRaw
FILE_PATH = r"C:\Thesis\python\data\geant4_raw_points.csv"
OUTPUT_DIR = r"C:\Thesis\python\figures"

# USTAWIENIA OSI (Dopasuj do BeamAxis w Unity)
# Jeśli w Unity strzelasz wzdłuż NIEBIESKIEJ strzałki (Z):
DEPTH_COL = 'Z'  # To będzie nasza OŚ PIONOWA na wykresie
LATERAL_COL_1 = 'X'  # To będzie OŚ POZIOMA
LATERAL_COL_2 = 'Y'  # To będzie OŚ POZIOMA (drugi widok)

# Jeśli strzelasz wzdłuż CZERWONEJ (X):
# DEPTH_COL = 'X'
# LATERAL_COL_1 = 'Y'
# LATERAL_COL_2 = 'Z'

# Wygląd
plt.style.use('dark_background')
CMAP = 'inferno'


# =============================================================================
# WIZUALIZACJA "STOJĄCA" (UNITY STYLE)
# =============================================================================

def generate_vertical_textures():
    if not os.path.exists(FILE_PATH):
        print("Brak pliku! Wyeksportuj RAW Point Cloud z Unity.")
        return

    print(f"Wczytywanie: {FILE_PATH}")
    df = pd.read_csv(FILE_PATH)

    # Pobieramy dane
    depth = df[DEPTH_COL]
    lat1 = df[LATERAL_COL_1]
    lat2 = df[LATERAL_COL_2]

    # Obliczamy zakresy (99.9% żeby uciąć szum, ale zachować kształt)
    max_depth = np.percentile(depth, 99.9)
    min_depth = np.percentile(depth, 0.1)

    # Dla głębokości zakładamy, że startuje od 0 (lub od min jeśli ujemne)
    # Rysujemy trochę więcej w górę (*1.1)
    plot_depth_min = min(0, min_depth)
    plot_depth_max = max_depth * 1.1

    # Dla boków szukamy max rozrzutu i robimy symetrię +/-
    max_lat = max(np.percentile(np.abs(lat1), 99.9), np.percentile(np.abs(lat2), 99.9)) * 1.1
    plot_lat_min = -max_lat
    plot_lat_max = max_lat

    print(f"Zakresy wykresu: Depth={plot_depth_max:.2f}, Lateral=+/-{max_lat:.2f}")

    # Rysujemy 3 widoki obok siebie
    fig, axes = plt.subplots(1, 3, figsize=(18, 7), constrained_layout=True)

    # --- WIDOK 1: TARCZA (Beam's Eye View) ---
    # Tutaj patrzymy "od dołu" na grzyba
    ax1 = axes[0]
    h1 = ax1.hist2d(lat1, lat2, bins=150, cmap=CMAP, norm=LogNorm(),
                    range=[[plot_lat_min, plot_lat_max], [plot_lat_min, plot_lat_max]])
    ax1.set_title("CROSS SECTION\n(Widok 'od lufy')", fontweight='bold', color='white')
    ax1.set_xlabel(f"Lateral {LATERAL_COL_1}")
    ax1.set_ylabel(f"Lateral {LATERAL_COL_2}")
    ax1.set_aspect('equal')
    ax1.grid(True, alpha=0.1)

    # --- WIDOK 2: PROFIL 1 (Stojący Grzyb) ---
    # NA ODWRÓT: Oś X to Lateral, Oś Y to Depth
    ax2 = axes[1]
    h2 = ax2.hist2d(lat1, depth, bins=[150, 150], cmap=CMAP, norm=LogNorm(),
                    range=[[plot_lat_min, plot_lat_max], [plot_depth_min, plot_depth_max]])

    ax2.set_title(f"SIDE PROFILE\n({LATERAL_COL_1} vs Depth)", fontweight='bold', color='white')
    ax2.set_xlabel(f"Lateral {LATERAL_COL_1} [cm]")
    ax2.set_ylabel(f"Depth {DEPTH_COL} [cm]")  # Głębokość na Y!
    ax2.axvline(0, color='white', linestyle='--', alpha=0.3)
    ax2.set_aspect('equal')  # Kluczowe, żeby grzyb nie był chudy/gruby
    ax2.grid(True, alpha=0.1)

    # --- WIDOK 3: PROFIL 2 (Stojący Grzyb obrócony o 90 stopni) ---
    ax3 = axes[2]
    h3 = ax3.hist2d(lat2, depth, bins=[150, 150], cmap=CMAP, norm=LogNorm(),
                    range=[[plot_lat_min, plot_lat_max], [plot_depth_min, plot_depth_max]])

    ax3.set_title(f"TOP PROFILE\n({LATERAL_COL_2} vs Depth)", fontweight='bold', color='white')
    ax3.set_xlabel(f"Lateral {LATERAL_COL_2} [cm]")
    ax3.set_ylabel(f"Depth {DEPTH_COL} [cm]")
    ax3.axvline(0, color='white', linestyle='--', alpha=0.3)
    ax3.set_aspect('equal')
    ax3.grid(True, alpha=0.1)

    # Zapis
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    save_path = os.path.join(OUTPUT_DIR, "unity_style_vertical.png")
    plt.savefig(save_path, dpi=300, facecolor='black')
    print(f"Gotowe! Obraz zapisany w: {save_path}")
    plt.show()


if __name__ == "__main__":
    generate_vertical_textures()