import numpy as np
import cv2
import os
from datetime import datetime


class FlashImageGenerator:
    """
    Улучшенный генератор с реалистичными типами вспышек
    """

    def __init__(self, img_size=(256, 256), background=1000.0, noise_level=10.0):
        self.img_size = img_size
        self.background = background
        self.noise_level = noise_level
        self.active_events = []  # список активных движущихся вспышек

    def generate_starfield(self, n_stars=80):
        img = np.ones(self.img_size, dtype=np.float32) * self.background
        for _ in range(n_stars):
            x = np.random.randint(0, self.img_size[1])
            y = np.random.randint(0, self.img_size[0])
            flux = np.random.uniform(800, 2500)
            fwhm = np.random.uniform(1.8, 3.2)
            self._add_star(img, x, y, flux, fwhm)
        return img

    def _add_star(self, img, x, y, flux, fwhm=2.5):
        sigma = fwhm / 2.355
        ys, xs = np.indices(img.shape)
        g = flux * np.exp(-0.5 * ((xs - x) ** 2 + (ys - y) ** 2) / sigma ** 2)
        img += g

    # ==================== НОВЫЕ РЕАЛИСТИЧНЫЕ ВСПЫШКИ ====================
    def _add_meteor(self, img, x, y, intensity=8000, length=25):
        """Метеор: яркий след с затуханием"""
        angle = np.random.uniform(0, np.pi * 2)
        dx = length * np.cos(angle)
        dy = length * np.sin(angle)
        for i in range(length):
            px = int(x + i * dx / length)
            py = int(y + i * dy / length)
            if 0 <= px < img.shape[1] and 0 <= py < img.shape[0]:
                alpha = 1 - i / length
                img[py, px] += intensity * alpha * np.random.uniform(0.9, 1.1)

    def _add_satellite(self, img, x, y, intensity=3500):
        """Спутник: медленное движение, лёгкое мерцание"""
        img[int(y), int(x)] += intensity
        # небольшой "хвост" от экспозиции
        img[int(y), int(x) + 1] += intensity * 0.6

    def _add_airplane(self, img, x, y, intensity=4500):
        """Самолёт: красные/зелёные огни + след"""
        img[int(y), int(x)] += intensity
        # навигационные огни
        img[int(y) + 2, int(x) + 3] += intensity * 0.8  # зелёный
        img[int(y) - 2, int(x) - 3] += intensity * 0.8  # красный

    def _add_lightning(self, img, x, y, intensity=12000):
        """Молния: ветвистая, очень яркая и короткая"""
        for _ in range(8):  # несколько ветвей
            cx, cy = x, y
            for _ in range(12):
                cx += np.random.randint(-4, 5)
                cy += np.random.randint(-8, 3)
                if 0 <= cx < img.shape[1] and 0 <= cy < img.shape[0]:
                    img[cy, cx] += intensity * np.random.uniform(0.6, 1.0)

    # ==================== ОСНОВНОЙ ГЕНЕРАТОР ====================
    def generate_and_save_sequence(self, n_frames=60, flash_prob=0.12,
                                   flash_type="random", out_dir="temp_frames"):
        os.makedirs(out_dir, exist_ok=True)
        base_image = self.generate_starfield()
        image_paths = []
        flash_locations = []

        for i in range(n_frames):
            img = base_image.copy()
            img += np.random.normal(0, self.noise_level, self.img_size)

            # Обновляем активные события (движение)
            for event in self.active_events[:]:
                event['life'] -= 1
                if event['life'] <= 0:
                    self.active_events.remove(event)
                    continue
                # двигаем
                event['x'] += event.get('vx', 0)
                event['y'] += event.get('vy', 0)

            # Новая вспышка
            if np.random.random() < flash_prob or flash_type != "random":
                typ = flash_type if flash_type != "random" else np.random.choice(
                    ['meteor', 'satellite', 'airplane', 'lightning'])

                x = np.random.randint(30, self.img_size[1] - 30)
                y = np.random.randint(30, self.img_size[0] - 30)

                if typ == 'meteor':
                    self._add_meteor(img, x, y)
                    # добавляем движение на 3-6 кадров
                    self.active_events.append({
                        'x': x, 'y': y, 'life': np.random.randint(3, 7),
                        'vx': np.random.uniform(-4, -1), 'vy': np.random.uniform(-1, 1)
                    })
                elif typ == 'satellite':
                    self._add_satellite(img, x, y)
                    self.active_events.append({
                        'x': x, 'y': y, 'life': np.random.randint(8, 15),
                        'vx': np.random.uniform(-1.5, -0.5)
                    })
                elif typ == 'airplane':
                    self._add_airplane(img, x, y)
                else:  # lightning
                    self._add_lightning(img, x, y)

                flash_locations.append((i, typ))

            # Сохранение
            img_uint16 = np.clip(img, 0, 65535).astype(np.uint16)
            filename = os.path.join(out_dir, f"frame_{i:04d}.png")
            cv2.imwrite(filename, img_uint16)
            image_paths.append(filename)

        print(f"Сгенерировано {n_frames} кадров ({flash_type}) → {out_dir}")
        print(f"Вспышек: {len(flash_locations)}")
        return image_paths, flash_locations

if __name__ == '__main__':
    demo_generator()