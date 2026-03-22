import numpy as np
from scipy import signal
from scipy.fft import fft, fftfreq
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


class SignalDetector:
    """
    Детектор сигналов во временных рядах
    """

    def __init__(self):
        self.detection_methods = ['threshold', 'derivative', 'wavelet']

    def find_peaks_threshold(self, data, threshold_std=3.0, min_distance=5):
        """
        Поиск пиков по порогу
        """
        data = np.array(data)
        mean_val = np.mean(data)
        std_val = np.std(data)

        threshold = mean_val + threshold_std * std_val

        peaks = []
        i = 0
        while i < len(data):
            if data[i] > threshold:
                # Находим локальный максимум в окрестности
                start = max(0, i - min_distance)
                end = min(len(data), i + min_distance + 1)
                local_max_idx = start + np.argmax(data[start:end])

                peaks.append(local_max_idx)
                i = local_max_idx + min_distance
            else:
                i += 1

        return peaks, threshold

    def find_peaks_derivative(self, data, deriv_threshold=0.1):
        """
        Поиск пиков по производной
        """
        data = np.array(data)

        # Вычисляем производную
        derivative = np.diff(data)

        # Находим резкие изменения
        sharp_changes = np.where(np.abs(derivative) > deriv_threshold)[0]

        peaks = []
        for idx in sharp_changes:
            if idx > 0 and idx < len(data) - 1:
                # Проверяем, что это локальный максимум
                if data[idx] > data[idx - 1] and data[idx] > data[idx + 1]:
                    peaks.append(idx)

        return peaks

    def analyze_frequency_domain(self, data, sampling_rate=1.0):
        """
        Анализ в частотной области
        """
        n = len(data)

        # Быстрое преобразование Фурье
        yf = fft(data - np.mean(data))
        xf = fftfreq(n, 1 / sampling_rate)

        # Только положительные частоты
        pos_freq_mask = xf > 0
        xf = xf[pos_freq_mask]
        yf = np.abs(yf[pos_freq_mask])

        # Находим доминирующие частоты
        dominant_freq_idx = np.argmax(yf[1:]) + 1  # Пропускаем постоянную составляющую
        dominant_freq = xf[dominant_freq_idx]
        dominant_power = yf[dominant_freq_idx]

        return xf, yf, dominant_freq, dominant_power

    def plot_signal_analysis(self, data, times=None, sampling_rate=1.0):
        """
        Анализ сигнала БЕЗ отрисовки графиков — только расчёт и вывод в консоль
        """
        data = np.asarray(data)
        n = len(data)

        if times is None:
            times = np.arange(n)

        # Расчёт пиков (это остаётся)
        peaks_thresh, threshold = self.find_peaks_threshold(data)
        peaks_deriv = self.find_peaks_derivative(data)

        # Выводим только текстовые результаты
        print(f"Обнаружено пиков (порог): {len(peaks_thresh)}")
        print(f"Обнаружено пиков (производная): {len(peaks_deriv)}")

        if n >= 8:
            xf, yf, dom_freq, dom_power = self.analyze_frequency_domain(data, sampling_rate)
            print(f"Доминирующая частота: {dom_freq:.3f} Гц")
        else:
            print("Частотный анализ пропущен (недостаточно данных)")

        # Возвращаем то же, что и раньше — чтобы остальной код не сломался
        return peaks_thresh, peaks_deriv