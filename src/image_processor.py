import cv2
import numpy as np
from scipy import ndimage


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
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise FileNotFoundError(f"Не удалось загрузить изображение: {image_path}")

        # Нормализация
        img_normalized = img.astype(np.float32) / 255.0

        # Гауссово размытие для уменьшения шума
        img_blurred = cv2.GaussianBlur(img_normalized, (5, 5), 0)

        return img_blurred

    def initialize_background(self, image):
        """
        Инициализация модели фона
        """
        self.background_model = image.copy()

    def update_background(self, image):
        """
        Обновление модели фона (running average)
        """
        if self.background_model is None:
            self.initialize_background(image)
        else:
            self.background_model = (1 - self.learning_rate) * self.background_model + \
                                    self.learning_rate * image

    def detect_foreground(self, image, threshold=0.1):
        """
        Детектирование переднего плана (потенциальных вспышек)
        """
        if self.background_model is None:
            self.initialize_background(image)

        # Разность с фоном
        diff = np.abs(image - self.background_model)

        # Бинаризация
        foreground_mask = diff > threshold

        # Морфологические операции для очистки
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        foreground_mask = cv2.morphologyEx(foreground_mask.astype(np.uint8),
                                           cv2.MORPH_OPEN, kernel)
        foreground_mask = cv2.morphologyEx(foreground_mask,
                                           cv2.MORPH_CLOSE, kernel)

        return foreground_mask, diff

    def find_flash_candidates(self, foreground_mask, min_size=5):
        """
        Поиск кандидатов во вспышки
        """
        # Поиск связных компонентов
        labeled, num_features = ndimage.label(foreground_mask)

        candidates = []
        for i in range(1, num_features + 1):
            component_mask = labeled == i
            if np.sum(component_mask) >= min_size:
                # Вычисляем свойства компонента
                y_coords, x_coords = np.where(component_mask)
                center_y, center_x = np.mean(y_coords), np.mean(x_coords)
                area = np.sum(component_mask)
                intensity = np.sum(component_mask)

                candidates.append({
                    'center': (center_x, center_y),
                    'area': area,
                    'intensity': intensity,
                    'bbox': (np.min(x_coords), np.min(y_coords),
                             np.max(x_coords), np.max(y_coords))
                })

        return candidates

    def process_image_sequence(self, image_paths):
        """
        Обработка последовательности изображений для поиска вспышек
        """
        all_detections = []

        for i, image_path in enumerate(image_paths):
            try:
                # Загрузка и предобработка
                image = self.load_and_preprocess(image_path)

                # Детектирование переднего плана
                foreground_mask, diff_map = self.detect_foreground(image)

                # Поиск кандидатов
                candidates = self.find_flash_candidates(foreground_mask)

                # Обновление фона
                self.update_background(image)

                all_detections.append({
                    'frame': i,
                    'image_path': image_path,
                    'candidates': candidates,
                    'foreground_mask': foreground_mask,
                    'diff_map': diff_map
                })

                print(f"Обработан кадр {i + 1}/{len(image_paths)}: найдено {len(candidates)} кандидатов")

            except Exception as e:
                print(f"Ошибка обработки {image_path}: {e}")
                continue

        return all_detections