import cv2
import numpy as np
from scipy import ndimage

# Цвета и короткие английские метки
OBJECT_COLORS = {
    "звезда (точечная)": (255, 0, 0),      # синий
    "звезда с дифракцией / кластер": (0, 255, 0),      # зелёный
    "галактика / крупный кластер": (0, 165, 255),      # оранжевый
    "вспышка / спутник / мусор / артефакт": (0, 0, 255),      # красный
    "вспышка (вытянутый)": (0, 255, 255),      # циан
    "мусор / артефакт": (128, 0, 128),      # фиолетовый
    "неизвестно": (200, 200, 200)      # светло-серый
}

LABEL_MAP = {
    "звезда (точечная)": "Point Star",
    "звезда с дифракцией / кластер": "Diffraction Star",
    "галактика / крупный кластер": "Galaxy/Cluster",
    "вспышка / спутник / мусор / артефакт": "Flash/Debris",
    "вспышка (вытянутый)": "Elongated Flash",
    "мусор / артефакт": "Artifact",
    "неизвестно": "Unknown"
}


class ImageProcessor:
    """
    Обработчик изображений для детектирования вспышек
    """

    def __init__(self):
        self.background_model = None
        self.learning_rate = 0.01

    def load_and_preprocess(self, image_path):
        """
        Загрузка и предобработка изображения
        """
        img_bgr = cv2.imread(image_path)
        if img_bgr is None:
            raise FileNotFoundError(f"Не удалось загрузить: {image_path}")
        color_original = img_bgr.copy()
        if len(img_bgr.shape) == 3:
            gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
        else:
            gray = img_bgr.astype(np.float32) / 255.0
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        return blurred, color_original

    def initialize_background(self, image):
        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
        self.background_model = image.copy()

    def update_background(self, image):
        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
        self.background_model = (1 - self.learning_rate) * self.background_model + self.learning_rate * image

    def detect_foreground(self, image, threshold=0.04):
        """
        image должен быть серым (2D, 0–1)
        """
        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0

        if self.background_model is None:
            self.initialize_background(image)
            return np.zeros_like(image, dtype=bool), np.zeros_like(image)

        diff = np.abs(image - self.background_model)
        foreground_mask = diff > (threshold * 0.8)

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        foreground_mask = cv2.morphologyEx(foreground_mask.astype(np.uint8), cv2.MORPH_OPEN, kernel)
        foreground_mask = cv2.morphologyEx(foreground_mask, cv2.MORPH_CLOSE, kernel)

        return foreground_mask.astype(bool), diff

    def find_flash_candidates(self, foreground_mask, min_size=3):
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
            foreground_mask.astype(np.uint8), connectivity=8
        )
        candidates = []
        for i in range(1, num_labels):
            area = stats[i, cv2.CC_STAT_AREA]
            if area < min_size:
                continue

            x, y, w, h = stats[i, cv2.CC_STAT_LEFT:cv2.CC_STAT_LEFT + 4]
            aspect_ratio = max(w, h) / min(w, h) if min(w, h) > 0 else 1.0

            if area <= 12:
                obj_type = "звезда (точечная)"
            elif area <= 80:
                obj_type = "звезда с дифракцией / кластер"
            elif area <= 400:
                obj_type = "галактика / крупный кластер"
            else:
                obj_type = "вспышка / спутник / мусор / артефакт"

            if aspect_ratio > 2.5:
                obj_type = "вспышка (вытянутый)"

            candidates.append({
                'bbox': (x, y, x + w, y + h),
                'area': area,
                'centroid': centroids[i],
                'aspect_ratio': aspect_ratio,
                'type': obj_type
            })

        return candidates

    def process_image_sequence(self, image_paths, threshold=0.1):
        all_detections = []
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
                print(f"Обработан кадр {i+1}/{len(image_paths)}: {len(candidates)} кандидатов")
            except Exception as e:
                print(f"Ошибка {image_path}: {e}")
                continue
        return all_detections

    def draw_bounding_boxes(self, color_img, candidates):
        """
        Рисует рамки разного цвета в зависимости от типа кандидата.
        """
        img = color_img.copy()

        for cand in candidates:
            x1, y1, x2, y2 = cand['bbox']
            obj_type = cand.get('type', 'неизвестно') or 'неизвестно'

            color = OBJECT_COLORS.get(obj_type, (128, 128, 128))
            display_label = LABEL_MAP.get(obj_type, "Unknown")

            thickness = 3 if "вспышка" in obj_type else 2

            cv2.rectangle(
                img,
                (int(x1), int(y1)),
                (int(x2), int(y2)),
                color,
                thickness=thickness
            )

            cv2.putText(
                img,
                display_label,
                (int(x1), int(y1) - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                1,
                cv2.LINE_AA
            )

        return img