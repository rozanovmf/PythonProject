import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import pandas as pd
from src.image_processor import ImageProcessor
from src.signal_detector import SignalDetector

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

    def plot_light_curve_from_frames(self, flash_type="Метеор", n_frames=60, interval_sec=1.0):
        """
        Генерирует кадры ОДИН РАЗ → сохраняет в папку → извлекает интенсивность → показывает график
        Теперь график и папка используют ОДИНАКОВЫЕ файлы
        """
        from generator import FlashImageGenerator
        import matplotlib.image as mpimg
        import os

        print(f"Генерирую {n_frames} кадров для типа '{flash_type}'...")

        # === ГЕНЕРАЦИЯ ОДИН РАЗ ===
        generator = FlashImageGenerator(img_size=(256, 256), noise_level=12.0)

        # Увеличиваем вероятность вспышки для наглядности (можно потом убрать)
        prob = 0.25 if flash_type == "Метеор" else 0.15

        image_paths, flash_locations = generator.generate_and_save_sequence(
            n_frames=n_frames,
            flash_prob=prob,
            out_dir="temp_frames"
        )

        # === Сообщаем пользователю точную папку ===
        print(f" Кадры сохранены в: {os.path.abspath('temp_frames')}")
        print(f"   Всего файлов: {len(image_paths)}")

        # === Обработка изображений и извлечение кривой ===
        processor = ImageProcessor()
        detections = processor.process_image_sequence(image_paths)

        intensities = [np.mean(det['diff_map']) for det in detections]
        times = [i * interval_sec for i in range(len(intensities))]

        # === Построение графика ===
        fig = plt.figure(figsize=(15, 9))

        # Верхний график — кривая блеска
        ax1 = fig.add_subplot(2, 2, (1, 2))
        ax1.plot(times, intensities, 'b-', linewidth=2, label='Интенсивность (из кадров)')
        ax1.scatter([t for i, t in enumerate(times) if i in [loc[0] for loc in flash_locations]],
                    [intensities[i] for i, _ in flash_locations],
                    color='red', s=80, zorder=5, label='Кадры со вспышкой')
        ax1.set_title(f'Кривая блеска на основе реальных кадров — {flash_type}')
        ax1.set_xlabel('Время (секунды)')
        ax1.set_ylabel('Средняя разница с фоном')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # === ВЫБИРАЕМ РЕАЛЬНЫЙ КАДР СО ВСПЫШКОЙ (самый важный фикс) ===
        ax2 = fig.add_subplot(2, 2, 3)
        if flash_locations:
            # Берём первый кадр, где была вспышка
            flash_frame_idx = flash_locations[0][0]
            sample_path = image_paths[flash_frame_idx]
            print(f" Показываем реальный кадр со вспышкой: {sample_path}")
        else:
            # Если вспышек не было — берём средний
            sample_path = image_paths[len(image_paths) // 2]

        try:
            sample_img = mpimg.imread(sample_path)
            ax2.imshow(sample_img, cmap='gray')
            ax2.set_title(f'Кадр со вспышкой (frame_{flash_frame_idx:04d})' if flash_locations else 'Пример кадра')
            ax2.axis('off')
        except Exception as e:
            ax2.text(0.5, 0.5, f'Ошибка загрузки:\n{sample_path}\n{e}', ha='center', va='center', color='red')
            ax2.axis('off')

        # Маска последнего кадра
        ax3 = fig.add_subplot(2, 2, 4)
        try:
            last_mask = detections[-1]['foreground_mask']
            ax3.imshow(last_mask, cmap='hot')
            ax3.set_title('Маска переднего плана (последний кадр)')
            ax3.axis('off')
        except:
            ax3.text(0.5, 0.5, 'Маска недоступна', ha='center', va='center')

        plt.tight_layout()
        plt.show()

        print(f"Готово! Использовано {n_frames} кадров из папки temp_frames")
        return image_paths

    def plot_signal_with_frames(self, flash_type="Метеор", n_frames=60, interval_sec=1.0):
        """
        Аналогично plot_light_curve_from_frames, но использует SignalDetector.plot_signal_analysis
        + показывает кадры со вспышками
        """
        from generator import FlashImageGenerator
        import matplotlib.image as mpimg
        import os

        print(f"Генерирую {n_frames} кадров для типа '{flash_type}' (для детекции сигнала)...")

        # Генерация кадров один раз
        generator = FlashImageGenerator(img_size=(256, 256), noise_level=12.0)
        prob = 0.25 if flash_type == "Метеор" else 0.15
        image_paths, flash_locations = generator.generate_and_save_sequence(
            n_frames=n_frames,
            flash_prob=prob,
            out_dir="temp_frames_signal"  # отдельная папка, чтобы не затирать temp_frames
        )

        print(f"Кадры сохранены в: {os.path.abspath('temp_frames_signal')}")

        # Извлечение интенсивностей из кадров
        processor = ImageProcessor()
        detections = processor.process_image_sequence(image_paths)
        intensities = [np.mean(det['diff_map']) for det in detections]
        times = [i * interval_sec for i in range(len(intensities))]

        # Построение графика анализа сигнала
        print(f"Анализ сигнала ({flash_type}) на основе {n_frames} кадров...")
        detector = SignalDetector()
        detector.plot_signal_analysis(intensities, times=times)

        # Дополнительно показываем кадры со вспышками
        if flash_locations:
            fig, axes = plt.subplots(1, min(3, len(flash_locations)), figsize=(12, 4))
            if len(flash_locations) == 1:
                axes = [axes]  # для совместимости

            for idx, (frame_idx, _) in enumerate(flash_locations[:3]):  # показываем до 3 кадров
                try:
                    img = mpimg.imread(image_paths[frame_idx])
                    axes[idx].imshow(img, cmap='gray')
                    axes[idx].set_title(f"Вспышка в кадре {frame_idx:04d}")
                    axes[idx].axis('off')
                except:
                    axes[idx].text(0.5, 0.5, "Ошибка загрузки", ha='center', va='center')
                    axes[idx].axis('off')

            plt.suptitle("Кадры, на основе которых построен график")
            plt.tight_layout()
            plt.show()
        else:
            print("Вспышек не обнаружено в сгенерированных кадрах.")

        input("\nНажмите Enter для возврата в меню...")
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