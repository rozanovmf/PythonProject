import sys
import time
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'          # 0 = все логи, 1 = INFO отключены, 2 = WARNING отключены, 3 = ERROR только
from src.flash_analyzer import FlashAnalyzer
from src.signal_detector import SignalDetector
from flash_model import load_model


def print_menu():
    print("\n" + "=" * 50)
    print("        ДЕТЕКТОР СВЕТОВЫХ ВСПЫШЕК")
    print("=" * 50)
    print("1 — Анализ кривых блеска")
    print("2 — Детектирование вспышек в сигнале")
    print("3 — Статистика вспышек")
    print("4 — ML классификация вспышек")
    print("5 — Генерация тестовых данных")
    print("0 — Выход")


def choose_flash_type():
    flash_types = ["Метеор", "Спутник", "Искусственный", "Естественный"]

    print("\nВыберите тип вспышки:")
    for i, flash_type in enumerate(flash_types, 1):
        print(f"{i} — {flash_type}")

    try:
        choice = int(input("Введите номер: "))
        if 1 <= choice <= len(flash_types):
            return flash_types[choice - 1]
        else:
            print("Неверный выбор.")
            return None
    except ValueError:
        print("Пожалуйста, введите число.")
        return None


def main():
    analyzer = FlashAnalyzer()
    detector = SignalDetector()
    model = load_model()

    while True:
        print_menu()
        choice = input("Ваш выбор: ")

        if choice == "1":
            flash_type = choose_flash_type()
            if flash_type:
                hours = int(input("Часов наблюдения (по умолчанию 12): ") or "12")
                analyzer.plot_light_curve(flash_type, hours=hours)

        elif choice == "2":
            flash_type = choose_flash_type()
            if flash_type:
                # Генерируем тестовые данные
                times, intensities, _ = analyzer.generate_flash_data(flash_type, hours=6)
                print("Анализ сигнала...")
                detector.plot_signal_analysis(intensities, times=times)

        elif choice == "3":
            flash_type = choose_flash_type()
            if flash_type:
                days = int(input("Дней наблюдения (по умолчанию 3): ") or "3")
                analyzer.plot_flash_statistics(flash_type, days=days)

        elif choice == "4":
            flash_type = choose_flash_type()
            if flash_type:
                # Генерируем данные для классификации
                _, intensities, _ = analyzer.generate_flash_data(flash_type, hours=2)
                result, probabilities = model.predict_flash_type(intensities[:100])
                print(f"\nРезультат классификации: {result}")
                print("Вероятности:")
                class_names = ['Фон', 'Короткая', 'Длинная']
                for name, prob in zip(class_names, probabilities):
                    print(f"  {name}: {prob:.3f}")

        elif choice == "5":
            from generator import demo_generator
            print("Генерация тестовых изображений...")
            demo_generator()
            print("Тестовые данные сохранены в папку 'demo_flash_frames'")

        elif choice == "0":
            print("Выход...")
            time.sleep(1)
            sys.exit(0)

        else:
            print("Некорректный выбор.")


if __name__ == "__main__":
    main()