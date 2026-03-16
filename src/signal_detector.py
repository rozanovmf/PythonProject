import numpy as np
from scipy import signal
from scipy.fft import fft, fftfreq
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
        Полный анализ сигнала с графиками
        """
        if times is None:
            times = np.arange(len(data))

        fig, axes = plt.subplots(2, 2, figsize=(15, 10))

        # 1. Исходный сигнал с детектированными пиками
        peaks_thresh, threshold = self.find_peaks_threshold(data)
        peaks_deriv = self.find_peaks_derivative(data)

        axes[0, 0].plot(times, data, 'b-', alpha=0.7, label='Сигнал')
        axes[0, 0].axhline(threshold, color='r', linestyle='--',
                           label=f'Порог ({threshold:.2f})')
        axes[0, 0].scatter([times[p] for p in peaks_thresh], [data[p] for p in peaks_thresh],
                           color='red', s=50, zorder=5, label='Пики (порог)')
        axes[0, 0].scatter([times[p] for p in peaks_deriv], [data[p] for p in peaks_deriv],
                           color='orange', s=30, zorder=5, label='Пики (производная)')
        axes[0, 0].set_title('Детектирование пиков')
        axes[0, 0].set_xlabel('Время')
        axes[0, 0].set_ylabel('Интенсивность')
        axes[0, 0].legend()
        axes[0, 0].grid(alpha=0.3)

        # 2. Производная
        derivative = np.diff(data)
        axes[0, 1].plot(times[1:], derivative, 'g-', alpha=0.7)
        axes[0, 1].axhline(0.1, color='r', linestyle='--', label='Порог производной')
        axes[0, 1].axhline(-0.1, color='r', linestyle='--')
        axes[0, 1].set_title('Производная сигнала')
        axes[0, 1].set_xlabel('Время')
        axes[0, 1].set_ylabel('Производная')
        axes[0, 1].legend()
        axes[0, 1].grid(alpha=0.3)

        # 3. Частотный анализ
        xf, yf, dom_freq, dom_power = self.analyze_frequency_domain(data, sampling_rate)
        axes[1, 0].plot(xf, yf, 'purple', alpha=0.7)
        axes[1, 0].axvline(dom_freq, color='red', linestyle='--',
                           label=f'Доминирующая: {dom_freq:.3f} Гц')
        axes[1, 0].set_title('Частотный спектр')
        axes[1, 0].set_xlabel('Частота (Гц)')
        axes[1, 0].set_ylabel('Мощность')
        axes[1, 0].legend()
        axes[1, 0].grid(alpha=0.3)
        axes[1, 0].set_xlim(0, min(1.0, xf[-1]))

        # 4. Статистика пиков
        methods = ['Пороговый', 'Производная']
        counts = [len(peaks_thresh), len(peaks_deriv)]

        axes[1, 1].bar(methods, counts, color=['red', 'orange'], alpha=0.7)
        axes[1, 1].set_title('Количество обнаруженных пиков')
        axes[1, 1].set_ylabel('Количество')

        for i, count in enumerate(counts):
            axes[1, 1].text(i, count + 0.1, str(count), ha='center')

        plt.tight_layout()
        plt.show()

        print(f"Обнаружено пиков (порог): {len(peaks_thresh)}")
        print(f"Обнаружено пиков (производная): {len(peaks_deriv)}")
        print(f"Доминирующая частота: {dom_freq:.3f} Гц")

        return peaks_thresh, peaks_deriv