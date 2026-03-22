import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import pandas as pd
from src.image_processor import ImageProcessor
from src.signal_detector import SignalDetector
import os
import cv2
from collections import Counter


class FlashAnalyzer:
    """
    Анализатор световых вспышек
    """

    def __init__(self):
        from src.real_data_loader import RealSequenceLoader
        self.loader = RealSequenceLoader()
        self.flash_types = {
            'Метеор': {'duration_sec': 0.1, 'intensity': 8.0, 'frequency': 'редко'},
            'Спутник': {'duration_sec': 2.0, 'intensity': 3.0, 'frequency': 'часто'},
            'Искусственный': {'duration_sec': 0.5, 'intensity': 6.0, 'frequency': 'средне'},
            'Естественный': {'duration_sec': 0.2, 'intensity': 7.0, 'frequency': 'редко'}
        }

    def analyze_real_sequence(self, source_path, n_frames=100, threshold=0.04):
        """
        Анализ реальной последовательности.
        threshold — порог яркости для выделения объектов
        """
        print(f"Загружаем реальную последовательность: {source_path}")
        print(f"Используется порог яркости: {threshold:.4f}")

        # Очистка старых аннотаций
        output_dir = "annotated_frames"
        if os.path.exists(output_dir):
            for old_file in os.listdir(output_dir):
                file_path = os.path.join(output_dir, old_file)
                if os.path.isfile(file_path):
                    try:
                        os.remove(file_path)
                        print(f"Удалён старый файл: {old_file}")
                    except Exception as e:
                        print(f"Не удалось удалить {old_file}: {e}")
        os.makedirs(output_dir, exist_ok=True)

        try:
            image_paths, frame_arrays = self.loader.load_sequence(source_path, n_frames)
            print(f"Успешно загружено {len(frame_arrays)} кадров")
        except Exception as e:
            print(f"Ошибка загрузки: {e}")
            return None, None

        processor = ImageProcessor()
        detections = processor.process_image_sequence(image_paths, threshold=threshold)

        print(f"\nСохраняем аннотированные кадры по типам в: {output_dir}")
        print(f"Порог яркости: {threshold:.4f}\n")

        # Словарь для хранения изображений по типам
        type_to_images = {
            "звезда (точечная)": [],
            "звезда с дифракцией / кластер": [],
            "галактика / крупный кластер": [],
            "вспышка / спутник / мусор / артефакт": [],
            "вспышка (вытянутый)": [],
            "мусор / артефакт": [],
            "неизвестно": []
        }

        for det in detections:
            frame_idx = det['frame']
            color_img = det['color_original'].copy()

            # Получаем кандидатов из детекции
            candidates = det.get('candidates', [])
            det['candidates'] = candidates

            if candidates:
                # Группируем кандидатов по типу
                candidates_by_type = {}
                for cand in candidates:
                    obj_type = cand.get('type', 'неизвестно')
                    if obj_type not in candidates_by_type:
                        candidates_by_type[obj_type] = []
                    candidates_by_type[obj_type].append(cand)

                # Для каждого типа рисуем свои объекты
                for obj_type, type_cands in candidates_by_type.items():
                    annotated_img = processor.draw_bounding_boxes(
                        color_img.copy(),
                        type_cands
                    )
                    # Добавляем в список для данного типа
                    type_to_images[obj_type].append((frame_idx, annotated_img))

                # Вывод статистики по кадру
                frame_types = Counter(c['type'] for c in candidates)
                print(f"  Кадр {frame_idx:4}: ", end="")
                print(", ".join(f"{t}: {c}" for t, c in frame_types.items()))

        # Сохраняем отдельные файлы по типам
        saved_files = []
        for obj_type, images_list in type_to_images.items():
            if not images_list:
                continue

            # Берём последний кадр с этим типом
            last_idx, last_img = images_list[-1]

            # Создаём безопасное имя файла
            safe_name = obj_type.replace(' ', '_').replace('/', '_').replace('\\', '_')
            filename = f"annotated_{safe_name}.jpg"
            out_path = os.path.join(output_dir, filename)

            cv2.imwrite(out_path, last_img)
            saved_files.append(out_path)
            print(f"Сохранено: {filename} ({len(images_list)} кадров с этим типом)")

        # Сохраняем также все кадры с аннотациями
        for det in detections:
            frame_idx = det['frame']
            if det.get('candidates'):
                annotated_all = processor.draw_bounding_boxes(
                    det['color_original'].copy(),
                    det['candidates']
                )
                out_path = os.path.join(output_dir, f"frame_{frame_idx:04d}_annotated.jpg")
                cv2.imwrite(out_path, annotated_all)
                saved_files.append(out_path)

        # Статистика
        type_counts_total = Counter()
        for det in detections:
            candidates = det.get('candidates', [])
            if candidates:
                type_counts_total.update(c['type'] for c in candidates)

        print("\n" + "=" * 50)
        print("Статистика обнаруженных объектов")
        print("=" * 50)
        if not type_counts_total:
            print("Объекты не найдены совсем")
        else:
            for typ, cnt in sorted(type_counts_total.items(), key=lambda x: x[1], reverse=True):
                print(f"{typ:35} : {cnt:6} шт.")

        # График анализа сигнала (только если есть данные)
        if len(detections) > 0:
            intensities = []
            for det in detections:
                diff_map = det.get('diff_map')
                if diff_map is not None:
                    intensities.append(np.mean(diff_map))
                else:
                    intensities.append(0)

            times = list(range(len(intensities)))

            if len(intensities) >= 8:
                detector = SignalDetector()
                try:
                    detector.plot_signal_analysis(intensities, times=times)
                    print("\nГрафик анализа сигнала сохранён как 'signal_analysis.png'")
                except Exception as e:
                    print(f"Не удалось построить график: {e}")
            else:
                print(f"\nВнимание: слишком мало кадров ({len(intensities)}) для частотного анализа.")

        stats = {
            "total_objects": sum(type_counts_total.values()),
            "by_type": dict(type_counts_total.most_common()),
            "frames_processed": len(detections)
        }

        print(f"\nГотово. Аннотированные кадры сохранены в: {os.path.abspath(output_dir)}")
        return saved_files, stats