import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import pandas as pd


class FlashAnalyzer:
    """
    Анализатор световых вспышек
    """

    def __init__(self):
        self.flash_types = {
            'Метеор': {'duration_sec': 0.1, 'intensity': 8.0, 'frequency': 'редко'},
            'Спутник': {'duration_sec': 2.0, 'intensity': 3.0, 'frequency': 'часто'},
            'Искусственный': {'duration_sec': 0.5, 'intensity': 6.0, 'frequency': 'средне'},
            'Естественный': {'duration_sec': 0.2, 'intensity': 7.0, 'frequency': 'редко'}
        }

    def generate_flash_data(self, flash_type, hours=24, interval_min=1, noise_level=0.1):
        """
        Генерирует синтетические данные вспышек
        """
        if flash_type not in self.flash_types:
            raise ValueError(f"Неизвестный тип вспышки: {flash_type}")

        flash_params = self.flash_types[flash_type]
        n_points = int(hours * 60 / interval_min)

        times = []
        intensities = []
        flash_events = []

        # Генерируем временную шкалу
        start_time = datetime.utcnow()

        for i in range(n_points):
            current_time = start_time + timedelta(minutes=i * interval_min)
            times.append(current_time)

            # Базовая интенсивность с шумом
            base_intensity = 1.0 + np.random.normal(0, noise_level)

            # Случайные вспышки в зависимости от частоты
            flash_prob = {'редко': 0.01, 'средне': 0.03, 'часто': 0.08}[flash_params['frequency']]

            if np.random.random() < flash_prob:
                # Создаем вспышку
                flash_intensity = base_intensity * flash_params['intensity'] * np.random.uniform(0.8, 1.2)
                duration_points = max(1, int(flash_params['duration_sec'] / (interval_min * 60)))

                # Добавляем вспышку с гауссовым профилем
                for j in range(duration_points):
                    if i + j < n_points:
                        flash_time = times[i + j] if i + j < len(times) else current_time + timedelta(minutes=j)
                        peak_pos = duration_points / 2
                        gaussian_factor = np.exp(-0.5 * ((j - peak_pos) / (duration_points / 4)) ** 2)
                        flash_intensity_j = flash_intensity * gaussian_factor

                        intensities.append(float(flash_intensity_j))
                        flash_events.append(1)
                    else:
                        break
            else:
                intensities.append(float(base_intensity))
                flash_events.append(0)

        return times[:len(intensities)], intensities, flash_events

    def detect_flashes(self, intensities, threshold=3.0, min_duration=1):
        """
        Детектирует вспышки в данных интенсивности
        """
        intensities = np.array(intensities)
        mean_intensity = np.mean(intensities)
        std_intensity = np.std(intensities)

        # Порог для детектирования вспышек
        detection_threshold = mean_intensity + threshold * std_intensity

        # Находим пики выше порога
        flash_mask = intensities > detection_threshold

        # Группируем соседние вспышки
        flash_groups = []
        current_group = []

        for i, is_flash in enumerate(flash_mask):
            if is_flash:
                current_group.append(i)
            elif current_group:
                if len(current_group) >= min_duration:
                    flash_groups.append(current_group)
                current_group = []

        if current_group and len(current_group) >= min_duration:
            flash_groups.append(current_group)

        return flash_groups, detection_threshold

    def plot_light_curve(self, flash_type, hours=12, interval_min=1):
        """Строит кривую блеска с вспышками"""
        times, intensities, flash_events = self.generate_flash_data(
            flash_type, hours=hours, interval_min=interval_min
        )

        plt.figure(figsize=(12, 6))
        plt.plot(times, intensities, 'b-', alpha=0.7, linewidth=1, label='Интенсивность')

        # Подсвечиваем вспышки
        flash_times = [times[i] for i, flash in enumerate(flash_events) if flash == 1]
        flash_ints = [intensities[i] for i, flash in enumerate(flash_events) if flash == 1]

        if flash_times:
            plt.scatter(flash_times, flash_ints, color='red', s=30, zorder=5,
                        label='Вспышки', alpha=0.8)

        # Детектируем вспышки автоматически
        flash_groups, threshold = self.detect_flashes(intensities)
        plt.axhline(y=threshold, color='orange', linestyle='--',
                    label=f'Порог детектирования ({threshold:.2f})')

        plt.title(f'Кривая блеска - {flash_type}')
        plt.xlabel('Время')
        plt.ylabel('Относительная интенсивность')
        plt.legend()
        plt.grid(alpha=0.3)
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()

        print(f"Обнаружено вспышек: {len(flash_groups)}")
        return flash_groups

    def plot_flash_statistics(self, flash_type, days=3):
        """Статистика вспышек по типам"""
        all_flash_counts = []

        for _ in range(10):  # Многократные симуляции для статистики
            _, intensities, _ = self.generate_flash_data(flash_type, hours=days * 24)
            flash_groups, _ = self.detect_flashes(intensities)
            all_flash_counts.append(len(flash_groups))

        plt.figure(figsize=(10, 6))
        plt.hist(all_flash_counts, bins=15, alpha=0.7, edgecolor='black')
        plt.title(f'Статистика вспышек - {flash_type}\n({days} дней наблюдений)')
        plt.xlabel('Количество вспышек в сутки')
        plt.ylabel('Частота')
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.show()

        avg_flashes = np.mean(all_flash_counts)
        print(f"Среднее количество вспышек в сутки: {avg_flashes:.1f}")