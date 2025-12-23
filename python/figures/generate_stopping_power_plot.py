import matplotlib.pyplot as plt
import numpy as np
import os

# --- KONFIGURACJA ŚCIEŻEK ---
DATA_FOLDER = 'data'
PLOTS_FOLDER = 'plots'
FILENAME = 'nist_water_data.txt'

# Ścieżki pełne
FILE_PATH = os.path.join(DATA_FOLDER, FILENAME)
OUTPUT_PATH_PDF = os.path.join(PLOTS_FOLDER, 'stopping_power.pdf')
OUTPUT_PATH_PNG = os.path.join(PLOTS_FOLDER, 'stopping_power.png') # Dodano ścieżkę PNG

# Upewnij się, że folder wyjściowy istnieje
os.makedirs(PLOTS_FOLDER, exist_ok=True)


def load_nist_data(filepath):
    energies = []
    stopping_powers = []

    print(f"Wczytywanie danych z: {filepath}")

    try:
        with open(filepath, 'r') as f:
            lines = f.readlines()

        for line in lines:
            parts = line.strip().split()

            # Oczekujemy formatu: [Energia] [Stopping Power]
            # Filtrujemy nagłówki i puste linie
            if len(parts) == 2:
                try:
                    e = float(parts[0])
                    sp = float(parts[1])
                    energies.append(e)
                    stopping_powers.append(sp)
                except ValueError:
                    continue  # Pomija linie z tekstem (np. nagłówki)

        return np.array(energies), np.array(stopping_powers)

    except FileNotFoundError:
        print(f"BŁĄD: Nie znaleziono pliku {filepath}")
        return np.array([]), np.array([])


# --- GŁÓWNA LOGIKA ---

# 1. Wczytanie danych
energy, stopping_power = load_nist_data(FILE_PATH)

if len(energy) == 0:
    print("Brak danych do wygenerowania wykresu.")
    exit()

# 2. Konfiguracja stylu wykresu
plt.figure(figsize=(8, 5))
plt.rcParams.update({
    'font.size': 12,
    'font.family': 'serif',
    'mathtext.fontset': 'dejavuserif'
})

# 3. Rysowanie danych
plt.semilogx(energy, stopping_power, 'b-', linewidth=2, label='Total Stopping Power')

# 4. Zaznaczenie punktu 10 MeV
target_energy = 10.0
# Interpolacja dla dokładnej pozycji Y
target_sp = np.interp(target_energy, energy, stopping_power)

# Rysowanie czerwonej kropki
plt.plot(target_energy, target_sp, 'ro', zorder=5, label='10 MeV (Simulation Energy)')

# --- POPRAWIONA ADNOTACJA ---
# Używamy textcoords='offset points', aby odsunąć tekst o stałą liczbę punktów od kropki.
plt.annotate(
    '10 MeV',
    xy=(target_energy, target_sp),
    xytext=(-20, 25),  # Przesunięcie: 20pkt w lewo, 25pkt w górę
    textcoords='offset points',
    arrowprops=dict(facecolor='black', shrink=0.05, width=1, headwidth=6),
    fontsize=11,
    fontweight='bold',
    ha='right',  # Wyrównanie tekstu do prawej
    va='bottom'  # Wyrównanie do dołu
)

# 5. Opisy i formatowanie
plt.xlabel('Kinetic Energy (MeV)', fontsize=12)
plt.ylabel(r'Total Stopping Power ($\mathrm{MeV} \cdot \mathrm{cm}^2 / \mathrm{g}$)', fontsize=12)
plt.title('Electron Stopping Power in Liquid Water (NIST ESTAR)', fontsize=14, pad=15)

plt.grid(True, which="major", ls="-", alpha=0.6)
plt.grid(True, which="minor", ls=":", alpha=0.3)
plt.legend(loc='upper center', frameon=True, fancybox=True, framealpha=0.9)

# 6. Zapis do folderu plots
plt.tight_layout()

# Zapis PDF
plt.savefig(OUTPUT_PATH_PDF)
print(f"SUKCES: Wykres PDF zapisano w: {OUTPUT_PATH_PDF}")

# Zapis PNG (dpi=300 dla wysokiej jakości)
plt.savefig(OUTPUT_PATH_PNG, dpi=300)
print(f"SUKCES: Wykres PNG zapisano w: {OUTPUT_PATH_PNG}")

# plt.show()