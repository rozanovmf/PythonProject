import cv2
import numpy as np
from astropy.io import fits
import os
from pathlib import Path


class RealSequenceLoader:
    """
    Загрузчик реальных последовательностей: папка изображений / видео / FITS
    Возвращает список путей + готовые к использованию массивы
    """

    def load_sequence(self, source, max_frames=200):
        """
        source может быть:
        - путь к папке → все изображения/файлы FITS в ней
        - путь к видео (.mp4, .avi и т.д.)
        - путь к одному файлу (.jpg, .png, .tif, .fits, .fit)
        - путь к FITS-кубу (3D)
        """
        source = Path(source).resolve()  # делаем абсолютный путь — помогает с ошибками

        if source.is_dir():
            return self._from_folder(source, max_frames)

        elif source.is_file():
            ext = source.suffix.lower()
            if ext in {'.mp4', '.avi', '.mov', '.mkv'}:
                return self._from_video(str(source), max_frames)

            elif ext in {'.fits', '.fit'}:
                return self._from_fits(source, max_frames)

            elif ext in {'.png', '.jpg', '.jpeg', '.tif', '.tiff'}:
                # Одиночное изображение → последовательность из 1 кадра
                print(f"Обнаружен одиночный файл изображения: {source.name}")
                frame = self._load_single(source)
                temp_path = f"temp_single_frame_{source.stem}.png"
                cv2.imwrite(temp_path, (frame * 65535).astype(np.uint16))
                return [temp_path], [frame]

            else:
                raise ValueError(f"Неизвестный формат файла: {ext}")

        else:
            raise ValueError(f"Путь не существует или недоступен: {source}")

    def _from_folder(self, folder, max_frames):
        allowed = {'.png', '.jpg', '.jpeg', '.tif', '.tiff', '.fits', '.fit'}

        files = sorted(
            f for f in folder.iterdir()
            if f.is_file() and f.suffix.lower() in allowed
        )[:max_frames]

        paths = [str(f) for f in files]
        frames = []

        for f in files:
            try:
                frame = self._load_single(f)
                frames.append(frame)
            except Exception as e:
                print(f"Пропущен файл {f.name}: {e}")
                continue

        return paths, frames
    def _from_video(self, video_path, max_frames):
        cap = cv2.VideoCapture(video_path)
        frames = []
        paths = []  # для совместимости сохраним временные пути
        temp_dir = "temp_real_video_frames"
        os.makedirs(temp_dir, exist_ok=True)

        i = 0
        while len(frames) < max_frames:
            ret, frame = cap.read()
            if not ret: break
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
            frames.append(gray)
            path = os.path.join(temp_dir, f"real_frame_{i:04d}.png")
            cv2.imwrite(path, (gray * 65535).astype(np.uint16))
            paths.append(path)
            i += 1
        cap.release()
        return paths, frames

    def _from_fits(self, fits_path, max_frames):
        with fits.open(fits_path) as hdul:
            data = hdul[0].data
            if data.ndim == 2:  # один кадр
                data = data[None, :, :]
            # если 3D — это куб
            frames = []
            paths = []
            for i in range(min(data.shape[0], max_frames)):
                img = data[i].astype(np.float32)
                img = (img - img.min()) / (img.max() - img.min() + 1e-8)
                frames.append(img)
                path = f"temp_fits_frame_{i:04d}.png"
                cv2.imwrite(path, (img * 65535).astype(np.uint16))
                paths.append(path)
        return paths, frames

    def _load_single(self, path):
        ext = path.suffix.lower()
        try:
            if ext in {'.fits', '.fit'}:
                with fits.open(path) as hdul:
                    img = hdul[0].data.astype(np.float32)
                    if img.ndim == 3:  # иногда приходит с каналом
                        img = img[0] if img.shape[0] == 1 else np.mean(img, axis=0)
                    img = (img - np.min(img)) / (np.max(img) - np.min(img) + 1e-8)
                    return img
            else:  # png, jpg, tif и т.д.
                img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
                if img is None:
                    raise ValueError(f"Не удалось прочитать изображение: {path}")
                return img.astype(np.float32) / 255.0
        except Exception as e:
            raise ValueError(f"Ошибка при загрузке {path}: {str(e)}")