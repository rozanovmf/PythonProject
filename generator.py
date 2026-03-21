import numpy as np
import cv2
import os
from datetime import datetime, timedelta


class FlashImageGenerator:
    """
    Генератор синтетических изображений со вспышками
    """

    def __init__(self, img_size=(256, 256), background=1000.0, noise_level=10.0):
        self.img_size = img_size
        self.background = background
        self.noise_level = noise_level

    def generate_starfield(self, n_stars=50):
        """
        Генерирует звездное поле
        """
        img = np.ones(self.img_size) * self.background

        for _ in range(n_stars):
            x = np.random.randint(0, self.img_size[1])
            y = np.random.randint(0, self.img_size[0])
            flux = np.random.uniform(500, 2000)
            fwhm = np.random.uniform(1.5, 3.0)

            self._add_star(img, x, y, flux, fwhm)

        return img

    def _add_star(self, img, x, y, flux, fwhm=2.5):
        """Добавляет звезду на изображение"""
        sigma = fwhm / 2.355
        ys, xs = np.indices(img.shape)
        g = flux * np.exp(-0.5 * ((xs - x) ** 2 + (ys - y) ** 2) / sigma ** 2)
        img += g

    def add_flash(self, img, x, y, flash_intensity=5000, flash_duration=1):
        """
        Добавляет вспышку на изображение
        """
        # Создаем ядро вспышки (может быть разным для разных типов вспышек)
        flash_types = [
            lambda xx, yy: np.exp(-0.5 * ((xx - x) ** 2 + (yy - y) ** 2) / 4.0),  # Точечная
            lambda xx, yy: np.exp(-0.5 * ((xx - x) ** 2 + (yy - y) ** 2) / 8.0),  # Расширенная
            lambda xx, yy: np.exp(-0.5 * (np.abs(xx - x) + np.abs(yy - y)) / 3.0)  # Ромбовидная
        ]

        kernel = flash_types[np.random.randint(0, len(flash_types))]
        ys, xs = np.indices(img.shape)
        flash = flash_intensity * kernel(xs, ys)

        img += flash
        return img

    def generate_sequence(self, n_frames=100, flash_prob=0.05, out_dir=None):
        """
        Генерирует последовательность изображений со случайными вспышками
        """
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)

        base_image = self.generate_starfield()
        flash_events = []

        for i in range(n_frames):
            # Создаем копию базового изображения
            img = base_image.copy()

            # Добавляем шум
            img += np.random.normal(0, self.noise_level, self.img_size)

            flash_info = None

            # Случайно добавляем вспышку
            if np.random.random() < flash_prob:
                flash_x = np.random.randint(50, self.img_size[1] - 50)
                flash_y = np.random.randint(50, self.img_size[0] - 50)
                flash_intensity = np.random.uniform(3000, 10000)

                img = self.add_flash(img, flash_x, flash_y, flash_intensity)
                flash_info = {
                    'frame': i,
                    'x': flash_x,
                    'y': flash_y,
                    'intensity': flash_intensity
                }
                flash_events.append(flash_info)

            # Сохраняем изображение
            if out_dir:
                img_uint16 = np.clip(img, 0, 65535).astype(np.uint16)
                filename = os.path.join(out_dir, f"frame_{i:04d}.png")
                cv2.imwrite(filename, img_uint16)

            yield img, flash_info

    def generate_and_save_sequence(self, n_frames=60, flash_prob=0.12, out_dir="temp_frames"):
        """Генерирует кадры + сохраняет их + возвращает список путей"""
        import os
        os.makedirs(out_dir, exist_ok=True)

        image_paths = []
        flash_locations = []  # для отладки

        for i, (img, flash_info) in enumerate(self.generate_sequence(
                n_frames=n_frames, flash_prob=flash_prob, out_dir=out_dir)):

            filename = os.path.join(out_dir, f"frame_{i:04d}.png")
            image_paths.append(filename)
            if flash_info:
                flash_locations.append((i, flash_info))

        print(f" Сгенерировано {n_frames} кадров → {out_dir}")
        print(f" Вспышек найдено: {len(flash_locations)}")

        return image_paths, flash_locations




def demo_generator():
    """Демонстрация работы генератора"""
    generator = FlashImageGenerator(img_size=(200, 200))

    print("Генерация демонстрационной последовательности...")
    frames = list(generator.generate_sequence(
        n_frames=50,
        flash_prob=0.1,
        out_dir='demo_flash_frames'
    ))

    flash_count = sum(1 for _, flash in frames if flash is not None)
    print(f"Создано {len(frames)} кадров, {flash_count} со вспышками")

    return frames


if __name__ == '__main__':
    demo_generator()