import argparse
from src.flash_analyzer import FlashAnalyzer
from src.signal_detector import SignalDetector
from flash_model import load_model
from generator import FlashImageGenerator, demo_generator


def main():
    parser = argparse.ArgumentParser(
        description='Детектор световых вспышек — анализ кривых блеска, детекция и классификация',
        formatter_class=argparse.RawTextHelpFormatter
    )

    subparsers = parser.add_subparsers(dest='command', help='Доступные команды')

    # ────────────────────────────────────────────────
    # 1. analyze — построение кривой блеска
    # ────────────────────────────────────────────────
    analyze_parser = subparsers.add_parser(
        'analyze',
        help='Построение графика кривой блеска с детектированными вспышками'
    )
    analyze_parser.add_argument(
        '--type',
        choices=['Метеор', 'Спутник', 'Искусственный', 'Естественный'],
        required=True,
        help='Тип вспышки для симуляции'
    )
    analyze_parser.add_argument(
        '--hours',
        type=int,
        default=12,
        help='Количество часов наблюдения (по умолчанию: 12)'
    )
    analyze_parser.add_argument(
        '--interval',
        type=int,
        default=1,
        help='Интервал между замерами в минутах (по умолчанию: 1)'
    )

    # ────────────────────────────────────────────────
    # 2. detect — анализ сигнала (пики, производная, FFT)
    # ────────────────────────────────────────────────
    detect_parser = subparsers.add_parser(
        'detect',
        help='Детектирование пиков в сигнале + частотный анализ'
    )
    detect_parser.add_argument(
        '--type',
        choices=['Метеор', 'Спутник', 'Искусственный', 'Естественный'],
        required=True,
        help='Тип вспышки для симуляции'
    )
    detect_parser.add_argument(
        '--hours',
        type=int,
        default=6,
        help='Количество часов наблюдения (по умолчанию: 6)'
    )

    # ────────────────────────────────────────────────
    # 3. stats — статистика количества вспышек
    # ────────────────────────────────────────────────
    stats_parser = subparsers.add_parser(
        'stats',
        help='Статистика количества вспышек за несколько дней'
    )
    stats_parser.add_argument(
        '--type',
        choices=['Метеор', 'Спутник', 'Искусственный', 'Естественный'],
        required=True,
        help='Тип вспышки'
    )
    stats_parser.add_argument(
        '--days',
        type=int,
        default=3,
        help='Количество дней симуляции (по умолчанию: 3)'
    )
    stats_parser.add_argument(
        '--simulations',
        type=int,
        default=20,
        help='Количество прогонов симуляции для статистики (по умолчанию: 20)'
    )

    # ────────────────────────────────────────────────
    # 4. classify — ML-классификация типа вспышки
    # ────────────────────────────────────────────────
    classify_parser = subparsers.add_parser(
        'classify',
        help='Классификация вспышки с помощью 1D CNN (фон / короткая / длинная)'
    )
    classify_parser.add_argument(
        '--type',
        choices=['Метеор', 'Спутник', 'Искусственный', 'Естественный'],
        required=True,
        help='Тип вспышки для генерации примера'
    )
    classify_parser.add_argument(
        '--hours',
        type=float,
        default=2.0,
        help='Длина сигнала в часах (по умолчанию: 2)'
    )

    # ────────────────────────────────────────────────
    # 5. generate — генерация синтетических кадров
    # ────────────────────────────────────────────────
    generate_parser = subparsers.add_parser(
        'generate',
        help='Генерация синтетической последовательности изображений'
    )
    generate_parser.add_argument(
        '--frames',
        type=int,
        default=100,
        help='Количество кадров (по умолчанию: 100)'
    )
    generate_parser.add_argument(
        '--prob',
        type=float,
        default=0.05,
        help='Вероятность вспышки на каждом кадре (по умолчанию: 0.05)'
    )
    generate_parser.add_argument(
        '--out-dir',
        type=str,
        default='generated_frames',
        help='Папка для сохранения кадров (по умолчанию: generated_frames)'
    )
    generate_parser.add_argument(
        '--demo',
        action='store_true',
        help='Запустить демонстрацию (50 кадров в demo_flash_frames)'
    )

    args = parser.parse_args()

    # ────────────────────────────────────────────────
    # Обработка выбранной команды
    # ────────────────────────────────────────────────
    analyzer = FlashAnalyzer()
    detector = SignalDetector()
    model = load_model()

    if args.command == 'analyze':
        analyzer.plot_light_curve_from_frames(
            flash_type=args.type,
            hours=args.hours,
            interval_min=args.interval
        )

    elif args.command == 'detect':
        times, intensities, _ = analyzer.generate_flash_data(
            flash_type=args.type,
            hours=args.hours,
            interval_min=1
        )
        print(f"Анализ сигнала ({args.type}, {args.hours} ч)...")
        detector.plot_signal_analysis(intensities, times=times)

    elif args.command == 'stats':
        analyzer.plot_flash_statistics(
            flash_type=args.type,
            days=args.days
        )
        # При желании можно вывести более точную статистику
        print(f"\nСимуляций: {args.simulations} (можно изменить через --simulations)")

    elif args.command == 'classify':
        _, intensities, _ = analyzer.generate_flash_data(
            flash_type=args.type,
            hours=args.hours,
            interval_min=0.1  # более плотная выборка для ML
        )
        # Берём первые 100 точек (или меньше — модель интерполирует)
        signal = intensities[:200]  # можно взять больше — интерполяция внутри модели
        result, probabilities = model.predict_flash_type(signal)
        print(f"\nКлассификация ({args.type}): {result}")
        class_names = ['Фон (без вспышки)', 'Короткая вспышка', 'Длинная вспышка']
        for name, prob in zip(class_names, probabilities):
            print(f"  {name:18} : {prob:.4f}")

    elif args.command == 'generate':
        if args.demo:
            print("Запуск демонстрации (50 кадров)...")
            demo_generator()
        else:
            print(f"Генерация {args.frames} кадров (p={args.prob}) → {args.out_dir}")
            generator = FlashImageGenerator(img_size=(256, 256), noise_level=15.0)
            list(generator.generate_sequence(
                n_frames=args.frames,
                flash_prob=args.prob,
                out_dir=args.out_dir
            ))
            print("Генерация завершена.")

    elif args.command is None:
        parser.print_help()

    else:
        print("Неизвестная команда. Используйте --help")


if __name__ == '__main__':
    main()
