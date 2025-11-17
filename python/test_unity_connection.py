"""
Simple Unity Connection Test with Clear Instructions
"""

from unity_interface.environment_manager import UnityEnvironmentManager
import time

print("="*60)
print("Unity ML-Agents Connection Test")
print("="*60)

print("\n📋 CHECKLIST - Sprawdź przed uruchomieniem:")
print("  [ ] Unity jest otwarte")
print("  [ ] Scena TestScene jest załadowana")
print("  [ ] PhantomAgent jest w scenie")
print("  [ ] NIE jesteś w Play mode (przycisk Play NIE świeci)")
print("  [ ] Behavior Name = 'PhantomAgent'")

print("\n✅ Jeśli wszystko gotowe, naciśnij Enter...")
input()

print("\n" + "="*60)
print("KROK 1: Python czeka na połączenie...")
print("="*60)

try:
    env = UnityEnvironmentManager(
        environment_path=None,
        worker_id=0,
        base_port=5004
    )

    print("\n⏳ Python słucha na porcie 5005...")
    print("\n" + "="*60)
    print("🎮 KROK 2: TERAZ KLIKNIJ PLAY (▶️) W UNITY!")
    print("="*60)
    print("\nPrzełącz się do Unity i kliknij Play.")
    print("Czekam 60 sekund na połączenie...\n")

    # Initialize environment - to uruchomi połączenie
    success = env.initialize()

    if success:
        print("\n" + "="*60)
        print("✅ POŁĄCZONO Z UNITY!")
        print("="*60)

        # Get info
        obs_space = env.get_observation_space()
        print(f"\n📊 Informacje o środowisku:")
        print(f"  • Observation shapes: {obs_space['observation_shapes']}")
        print(f"  • Action size: {obs_space['action_size']}")
        print(f"  • Action type: {obs_space['action_type']}")

        # Reset
        print("\n🔄 Resetowanie środowiska...")
        state = env.reset()
        print(f"✅ Reset zakończony!")
        print(f"  • Liczba agentów: {len(state['agents'])}")

        # Test steps
        print("\n🏃 Wykonuję 5 kroków testowych...")
        import numpy as np

        for i in range(5):
            actions = np.random.randn(obs_space['action_size'])
            result = env.step(actions)
            print(f"  Krok {i+1}: Reward={result['rewards'][0]:.4f}")
            time.sleep(0.3)

        print("\n" + "="*60)
        print("🎉 TEST ZAKOŃCZONY SUKCESEM!")
        print("="*60)
        print("\n✅ Python może komunikować się z Unity!")
        print("✅ Agent odpowiada na akcje!")
        print("✅ Wszystko działa poprawnie!")

        # Cleanup
        env.close()

    else:
        print("\n❌ Nie udało się zainicjalizować środowiska")
        print("\nSprawdź:")
        print("  1. Czy kliknąłeś Play w Unity?")
        print("  2. Czy w Unity Console nie ma błędów?")
        print("  3. Czy Behavior Name = 'PhantomAgent'?")

except Exception as e:
    print(f"\n❌ Błąd: {e}")
    print("\n🔍 Troubleshooting:")
    print("  1. Uruchom ponownie Unity")
    print("  2. Sprawdź czy ML-Agents package jest zainstalowany")
    print("  3. Sprawdź Console w Unity - czy są błędy?")
    print("  4. Upewnij się że PhantomAgent ma wszystkie komponenty")

print("\n" + "="*60)
