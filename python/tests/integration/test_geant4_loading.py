import sys
import os

# 1. Ustawienie ścieżek Pythona (żeby widział src)
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '../../'))
sys.path.append(project_root)

# ============================================================================
# 2. NAPRAWA BŁĘDU DLL (Kluczowy krok na Windows)
# ============================================================================
# Musisz wpisać tutaj ścieżkę do folderu 'bin' Twojej instalacji Geant4.
# Zazwyczaj jest to coś w stylu: C:\Geant4\install\bin  lub C:\Thesis\geant4\install\bin
# Znajdź folder, w którym masz pliki typu 'G4global.dll'
# ============================================================================

# Zmień tę ścieżkę na Twoją prawdziwą!
path_to_geant4_bin = r"C:\Geant4\install\bin"

if os.name == 'nt' and os.path.exists(path_to_geant4_bin):
    try:
        # To mówi Pythonowi 3.8+: "Tu szukaj brakujących DLL-ek"
        os.add_dll_directory(path_to_geant4_bin)
        print(f"Added DLL directory: {path_to_geant4_bin}")
    except Exception as e:
        print(f"Could not add DLL directory: {e}")
else:
    print(f"⚠️ UWAGA: Nie znaleziono folderu DLL Geant4: {path_to_geant4_bin}")
    print("Upewnij się, że ścieżka w skrypcie jest poprawna!")

# ============================================================================

try:
    # Teraz import powinien zadziałać
    from src.simulation import geant4_sim

    print("✅ SUKCES: Biblioteka załadowana z src.simulation!")

    print("Inicjalizacja Geant4 (może chwilę potrwać)...")
    manager = geant4_sim.SimulationManager()
    print("✅ Manager utworzony.")

    print("Symulacja 1 zdarzenia...")
    result = manager.run_single()

    print(f"✅ Wynik otrzymany!")
    print(f"Liczba kroków: {len(result['x'])}")

    # Sprawdzenie czy mamy dane (zakładając, że coś trafiło w fantom)
    if len(result['x']) > 0:
        print(f"Pierwsza pozycja: ({result['x'][0]:.2f}, {result['y'][0]:.2f}, {result['z'][0]:.2f})")
        print(f"Pęd początkowy: ({result['px'][0]:.2f}, {result['py'][0]:.2f}, {result['pz'][0]:.2f})")
    else:
        print("⚠️ Ostrzeżenie: Otrzymano pustą trajektorię (może cząstka nie trafiła w fantom?)")

except ImportError as e:
    print(f"❌ BŁĄD IMPORTU: {e}")
    print(f"Szukano w: {os.path.join(project_root, 'src', 'simulation')}")
    print("Czy przebudowałeś projekt CMake po dodaniu LIBRARY_OUTPUT_DIRECTORY?")
except Exception as e:
    print(f"❌ BŁĄD URUCHOMIENIA: {e}")