import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
from collections import Counter
from src.image_processor import ImageProcessor
from src.signal_detector import SignalDetector


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
        print(f"\n{'=' * 60}")
        print(f"Начинаем анализ: {source_path}")
        print(f"Порог: {threshold:.4f}")
        print(f"{'=' * 60}\n")

        # Очистка старых аннотаций
        output_dir = "annotated_frames"
        if os.path.exists(output_dir):
            for old_file in os.listdir(output_dir):
                file_path = os.path.join(output_dir, old_file)
                if os.path.isfile(file_path):
                    try:
                        os.remove(file_path)
                    except:
                        pass
        os.makedirs(output_dir, exist_ok=True)

        try:
            image_paths, frame_arrays = self.loader.load_sequence(source_path, n_frames)
            print(f"✅ Загружено {len(frame_arrays)} кадров")

            # Для отладки: сохраняем первый кадр для визуальной проверки
            if frame_arrays:
                debug_path = os.path.join(output_dir, "debug_original_frame.png")
                cv2.imwrite(debug_path, (frame_arrays[0] * 255).astype(np.uint8))
                print(f"📷 Сохранён оригинальный кадр для отладки: {debug_path}")

        except Exception as e:
            print(f"❌ Ошибка загрузки: {e}")
            return None, None

        processor = ImageProcessor()

        # Для одиночного изображения используем специальный режим
        if len(frame_arrays) == 1:
            print("🔧 Режим анализа одиночного изображения")
            return self._analyze_single_image(image_paths[0], frame_arrays[0], processor, threshold, output_dir)

        # Для последовательности используем обычный режим
        detections = processor.process_image_sequence(image_paths, threshold=threshold)
        return self._process_detections(detections, processor, output_dir, threshold)

    def _analyze_single_image(self, image_path, image_array, processor, threshold, output_dir):
        """
        Специальный метод для анализа одиночного изображения
        """
        print(f"🔍 Анализ одиночного изображения...")

        # Загружаем цветное изображение
        color_img = cv2.imread(image_path)
        if color_img is None:
            print("❌ Не удалось загрузить цветное изображение")
            return None, None

        # Применяем различные методы детекции объектов

        # Метод 1: Пороговая обработка (простой метод)
        gray = cv2.cvtColor(color_img, cv2.COLOR_BGR2GRAY)
        gray_norm = gray.astype(np.float32) / 255.0

        # Адаптивный порог
        mean_val = np.mean(gray_norm)
        std_val = np.std(gray_norm)
        adaptive_threshold = mean_val + threshold * std_val

        print(f"📊 Статистика изображения:")
        print(f"   Средняя яркость: {mean_val:.3f}")
        print(f"   Стандартное отклонение: {std_val:.3f}")
        print(f"   Адаптивный порог: {adaptive_threshold:.3f}")

        # Создаём маску ярких областей
        bright_mask = gray_norm > adaptive_threshold

        # Морфологическая обработка
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        bright_mask = cv2.morphologyEx(bright_mask.astype(np.uint8), cv2.MORPH_OPEN, kernel)
        bright_mask = cv2.morphologyEx(bright_mask, cv2.MORPH_CLOSE, kernel)

        # Находим компоненты связности
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(bright_mask, connectivity=8)

        print(f"🔍 Найдено компонент: {num_labels - 1}")

        candidates = []
        for i in range(1, num_labels):
            area = stats[i, cv2.CC_STAT_AREA]
            if area < 3:  # Минимальная площадь
                continue

            x = stats[i, cv2.CC_STAT_LEFT]
            y = stats[i, cv2.CC_STAT_TOP]
            w = stats[i, cv2.CC_STAT_WIDTH]
            h = stats[i, cv2.CC_STAT_HEIGHT]

            aspect_ratio = max(w, h) / min(w, h) if min(w, h) > 0 else 1.0

            # Классификация
            if area <= 15:
                obj_type = "звезда (точечная)"
            elif area <= 80:
                obj_type = "звезда с дифракцией / кластер"
            elif area <= 300:
                obj_type = "галактика / крупный кластер"
            else:
                obj_type = "вспышка / спутник / мусор / артефакт"

            if aspect_ratio > 2.0:
                obj_type = "вспышка (вытянутый)"

            candidates.append({
                'bbox': (x, y, x + w, y + h),
                'area': area,
                'centroid': (centroids[i][0], centroids[i][1]),
                'aspect_ratio': aspect_ratio,
                'type': obj_type
            })

            print(f"   Объект {i}: площадь={area}, тип={obj_type}, позиция=({x},{y})")

        # Рисуем рамки
        if candidates:
            annotated_img = processor.draw_bounding_boxes(color_img, candidates)
            out_path = os.path.join(output_dir, "annotated_result.jpg")
            cv2.imwrite(out_path, annotated_img)
            print(f"✅ Сохранён результат: {out_path}")

            # Сохраняем маску для отладки
            mask_path = os.path.join(output_dir, "debug_mask.jpg")
            cv2.imwrite(mask_path, bright_mask * 255)
            print(f"🔧 Сохранена маска для отладки: {mask_path}")

            stats = {
                "total_objects": len(candidates),
                "by_type": Counter(c['type'] for c in candidates),
                "frames_processed": 1
            }

            return [out_path], stats
        else:
            print("⚠️ Объекты не найдены")

            # Сохраняем маску для отладки даже если объекты не найдены
            mask_path = os.path.join(output_dir, "debug_mask_empty.jpg")
            cv2.imwrite(mask_path, bright_mask * 255)
            print(f"🔧 Сохранена пустая маска: {mask_path}")

            # Сохраняем изображение с отладочной информацией
            debug_img = color_img.copy()
            cv2.putText(debug_img, f"Threshold: {threshold:.3f}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            cv2.putText(debug_img, f"Mean: {mean_val:.3f}, Std: {std_val:.3f}", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            debug_path = os.path.join(output_dir, "debug_info.jpg")
            cv2.imwrite(debug_path, debug_img)

            return None, None

    def _process_detections(self, detections, processor, output_dir, threshold):
        """
        Обработка результатов детекции для последовательности кадров
        """
        if not detections:
            print("❌ Нет данных для обработки")
            return None, None

        print(f"\n📊 Обработка {len(detections)} кадров...")

        saved_files = []
        type_counts_total = Counter()

        for det in detections:
            frame_idx = det['frame']
            color_img = det['color_original'].copy()
            candidates = det.get('candidates', [])

            if candidates:
                annotated_img = processor.draw_bounding_boxes(color_img, candidates)
                out_path = os.path.join(output_dir, f"frame_{frame_idx:04d}_annotated.jpg")
                cv2.imwrite(out_path, annotated_img)
                saved_files.append(out_path)

                # Статистика
                for cand in candidates:
                    obj_type = cand.get('type', 'неизвестно')
                    type_counts_total[obj_type] += 1

                print(f"   Кадр {frame_idx}: найдено {len(candidates)} объектов")

        stats = {
            "total_objects": sum(type_counts_total.values()),
            "by_type": dict(type_counts_total.most_common()),
            "frames_processed": len(detections)
        }

        print(f"\n✅ Готово. Найдено объектов: {stats['total_objects']}")

        if saved_files:
            return saved_files, stats
        else:
            return None, None