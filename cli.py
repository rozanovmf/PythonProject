import argparse
from src.flash_analyzer import FlashAnalyzer
from src.signal_detector import SignalDetector
from flash_model import load_model
from generator import demo_generator


def main():
    parser = argparse.ArgumentParser(description='Детектор световых вспышек')
    subparsers = parser.add_subparsers(dest='command', help='Доступные команды')

    # Анализ кривых блеска
    analyze_parser = subparsers.add_parser('analyze', help='Анализ кривых блеска')
    analyze_parser.add_argument('--type', choices=['Метеор', 'Спутник', 'Искусственный', 'Естественный'],
                                required=True, help='Тип вспышки')
    analyze_parser.add_argument('--hours', type=int, default=12, help='Часы наблюдения')

    # Детектирование сигналов
    detect_parser = subparsers.add_parser('detect', help='Детектирование вспышек в сигнале')
    detect_parser.add_argument('--type', choices=['Метеор', 'Спутник', 'Искусственный', 'Естественный'],
                               required=True, help='Тип вспышки')

    # Статистика
    stats_parser = subparsers.add_parser('stats', help='Статистика вспышек')
    stats_parser.add_argument('--type', choices=['Метеор', 'Спутник', 'Искусственный', 'Естественный'],
                               required=True, help='Тип вспышки')