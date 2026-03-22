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

    while True:  # ← бесконечный цикл — меню будет возвращаться всегда
        print("\n" + "=" * 50)
        print("        ДЕТЕКТОР СВЕТОВЫХ ВСПЫШЕК")
        print("=" * 50)
        print("1 — Анализ кривых блеска")
        print("2 — Детектирование вспышек в сигнале")
        print("3 — Статистика вспышек")
        print("4 — ML классификация вспышек")
        print("5 — Генерация тестовых данных")
        print("6 — Анализ реальной последовательности (видео/FITS/папка)")
        print("0 — Выход")

        choice = input("Ваш выбор: ").strip()

        if choice == "1":
            flash_type = choose_flash_type()
            if flash_type:
                try:
                    n_frames_str = input("Количество кадров для генерации (по умолчанию 60): ").strip()
                    n_frames = int(n_frames_str) if n_frames_str else 60

                    analyzer.plot_light_curve_from_frames(
                        flash_type=flash_type,
                        n_frames=n_frames,
                        interval_sec=1.0
                    )

                    # Даём время посмотреть график
                    input("\nНажмите Enter для возврата в главное меню...")
                except ValueError:
                    print("Ошибка: введите число")
                except Exception as e:
                    print(f"Ошибка: {e}")
                    input("\nНажмите Enter для возврата...")
        elif choice == "2":
            flash_type = choose_flash_type()
            if flash_type:
                try:
                    n_frames_str = input("Количество кадров для анализа (по умолчанию 60): ").strip()
                    n_frames = int(n_frames_str) if n_frames_str else 60
                    analyzer.plot_signal_with_frames(
                        flash_type=flash_type,
                        n_frames=n_frames,
                        interval_sec=1.0
                    )
                except ValueError:
                    print("Ошибка: введите число")
                except Exception as e:
                    print(f"Ошибка: {e}")
                    input("\nНажмите Enter...")
        elif choice == "3":
            flash_type = choose_flash_type()
            if flash_type:
                days = int(input("Количество дней (по умолчанию 3): ") or 3)
                analyzer.plot_flash_statistics(flash_type, days=days)
                input("\nНажмите Enter для возврата...")

        elif choice == "4":
            flash_type = choose_flash_type()
            if flash_type:
                _, intensities, _ = analyzer.generate_flash_data(flash_type, hours=2)
                result, probabilities = model.predict_flash_type(intensities[:100])
                print(f"\nРезультат: {result}")
                for name, prob in zip(['Фон', 'Короткая', 'Длинная'], probabilities):
                    print(f"  {name}: {prob:.3f}")
                input("\nНажмите Enter...")

        elif choice == "5":
            from generator import demo_generator
            print("Генерация демонстрации...")
            demo_generator()
            print("Готово. Папка: demo_flash_frames")
            input("\nНажмите Enter...")
        elif choice == "6":
            path = input("Путь к папке/видео/FITS: ").strip()
            while True:
                limit_str = input("Порог яркости для выделения объектов (0.00 – 1.00, рекомендуем 0.02–0.15): ").strip()
                try:
                    limit = float(limit_str)
                    if 0.0 <= limit <= 1.0:
                        break
                    else:
                        print("Порог должен быть числом от 0.0 до 1.0")
                except ValueError:
                    print("Введите число (например: 0.04 или 0.08)")
                # Передаём порог в метод
            analyzer.analyze_real_sequence(path, threshold=limit)
        elif choice == "0":
            print("Выход...")
            break

        else:
            print("Некорректный выбор.")
            time.sleep(1)

        print()  # пустая строка перед следующим меню


if __name__ == "__main__":
    main()