import cv2
import numpy as np

# Цвета и метки
OBJECT_COLORS = {
    "звезда (точечная)": (255, 0, 0),
    "звезда с дифракцией / кластер": (0, 255, 0),
    "галактика / крупный кластер": (0, 165, 255),
    "вспышка / спутник / мусор / артефакт": (0, 0, 255),
    "вспышка (вытянутый)": (0, 255, 255),
    "мусор / артефакт": (128, 0, 128),
    "неизвестно": (200, 200, 200)
}

LABEL_MAP = {
    "звезда (точечная)": "Star",
    "звезда с дифракцией / кластер": "Diffraction",
    "галактика / крупный кластер": "Galaxy",
    "вспышка / спутник / мусор / артефакт": "Flash",
    "вспышка (вытянутый)": "Elongated",
    "мусор / артефакт": "Artifact",
    "неизвестно": "Unknown"
}


class ImageProcessor:
    def __init__(self):
        self.background_model = None
        self.learning_rate = 0.01

    def load_and_preprocess(self, image_path):
        """Загрузка и предобработка"""
        img_bgr = cv2.imread(image_path)
        if img_bgr is None:
            raise FileNotFoundError(f"Не удалось загрузить: {image_path}")
        color_original = img_bgr.copy()

        if len(img_bgr.shape) == 3:
            gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
        else:
            gray = img_bgr.astype(np.float32)

        # Нормализация
        gray = (gray - gray.min()) / (gray.max() - gray.min() + 1e-8)

        return gray, color_original

    def detect_foreground(self, image, threshold=0.04):
        """Детекция переднего плана"""
        if self.background_model is None:
            self.background_model = image.copy()
            return np.zeros_like(image, dtype=bool), np.zeros_like(image)

        diff = np.abs(image - self.background_model)

        # Адаптивный порог
        mean_diff = np.mean(diff)
        std_diff = np.std(diff)
        adaptive_threshold = mean_diff + threshold * std_diff

        foreground_mask = diff > adaptive_threshold

        # Морфология
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        foreground_mask = cv2.morphologyEx(foreground_mask.astype(np.uint8), cv2.MORPH_OPEN, kernel)
        foreground_mask = cv2.morphologyEx(foreground_mask, cv2.MORPH_CLOSE, kernel)

        return foreground_mask.astype(bool), diff

    def update_background(self, image):
        """Обновление фона"""
        if self.background_model is None:
            self.background_model = image.copy()
        else:
            self.background_model = (1 - self.learning_rate) * self.background_model + self.learning_rate * image

    def find_flash_candidates(self, foreground_mask, min_size=2):
        """Поиск кандидатов"""
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
            foreground_mask.astype(np.uint8), connectivity=8
        )

        candidates = []
        for i in range(1, num_labels):
            area = stats[i, cv2.CC_STAT_AREA]
            if area < min_size:
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
                'bbox': (int(x), int(y), int(x + w), int(y + h)),
                'area': int(area),
                'centroid': (centroids[i][0], centroids[i][1]),
                'aspect_ratio': float(aspect_ratio),
                'type': obj_type
            })

        return candidates

    def process_image_sequence(self, image_paths, threshold=0.1):
        """Обработка последовательности"""
        all_detections = []
        self.background_model = None

        for i, image_path in enumerate(image_paths):
            try:
                processed, color_original = self.load_and_preprocess(image_path)
                foreground_mask, diff_map = self.detect_foreground(processed, threshold=threshold)
                candidates = self.find_flash_candidates(foreground_mask)
                self.update_background(processed)

                all_detections.append({
                    'frame': i,
                    'image_path': image_path,
                    'candidates': candidates,
                    'foreground_mask': foreground_mask,
                    'diff_map': diff_map,
                    'color_original': color_original
                })

                print(f"   Кадр {i + 1}: {len(candidates)} объектов")

            except Exception as e:
                print(f"   Ошибка кадра {i + 1}: {e}")
                continue

        return all_detections

    def draw_bounding_boxes(self, color_img, candidates):
        """Рисование рамок"""
        img = color_img.copy()

        for cand in candidates:
            x1, y1, x2, y2 = cand['bbox']
            obj_type = cand.get('type', 'неизвестно')

            color = OBJECT_COLORS.get(obj_type, (128, 128, 128))
            label = LABEL_MAP.get(obj_type, "Unknown")

            # Рисуем рамку
            thickness = 3 if "вспышка" in obj_type or "Flash" in label else 2
            cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)

            # Рисуем подпись
            label_y = max(y1 - 5, 20)
            cv2.putText(img, label, (x1, label_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)

            # Добавляем площадь для отладки
            area_text = str(cand['area'])
            cv2.putText(img, area_text, (x2 - 20, y2 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.3, color, 1)

        return img