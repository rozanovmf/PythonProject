import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'          # 0 = все логи, 1 = INFO отключены, 2 = WARNING отключены, 3 = ERROR только
from tensorflow.keras import layers, models


class FlashDetectorModel:
    """
    Модель ML для классификации вспышек
    """

    def __init__(self):
        self.model = None

    def build_cnn_model(self, input_shape=(100, 1), num_classes=3):
        """
        Строит CNN модель для классификации временных рядов
        """
        model = models.Sequential([
            layers.Input(shape=input_shape),

            # Первый сверточный блок
            layers.Conv1D(32, kernel_size=5, activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.MaxPooling1D(2),
            layers.Dropout(0.2),

            # Второй сверточный блок
            layers.Conv1D(64, kernel_size=3, activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.MaxPooling1D(2),
            layers.Dropout(0.2),

            # Третий сверточный блок
            layers.Conv1D(128, kernel_size=3, activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.GlobalAveragePooling1D(),
            layers.Dropout(0.3),

            # Полносвязные слои
            layers.Dense(64, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.3),

            layers.Dense(32, activation='relu'),
            layers.Dropout(0.2),

            layers.Dense(num_classes, activation='softmax')
        ])

        model.compile(
            optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )

        self.model = model
        return model

    def create_synthetic_data(self, n_samples=1000, sequence_length=100):
        """
        Создает синтетические данные для обучения
        """
        X = []
        y = []

        for i in range(n_samples):
            # Базовый сигнал с шумом
            base_signal = np.random.normal(0, 0.1, sequence_length)

            # Случайно выбираем тип сигнала
            signal_type = np.random.randint(0, 3)

            if signal_type == 0:  # Нет вспышки
                signal = base_signal + 1.0  # Базовая линия
                label = 0

            elif signal_type == 1:  # Короткая вспышка
                signal = base_signal + 1.0
                # Добавляем короткий пик
                peak_pos = np.random.randint(20, 80)
                peak_width = np.random.randint(3, 8)
                peak_height = np.random.uniform(2.0, 5.0)

                for j in range(peak_width):
                    pos = peak_pos + j - peak_width // 2
                    if 0 <= pos < sequence_length:
                        gaussian = np.exp(-0.5 * ((j - peak_width // 2) / (peak_width / 4)) ** 2)
                        signal[pos] += peak_height * gaussian
                label = 1

            else:  # Длинная вспышка
                signal = base_signal + 1.0
                # Добавляем широкий пик
                peak_pos = np.random.randint(30, 70)
                peak_width = np.random.randint(10, 25)
                peak_height = np.random.uniform(1.5, 3.0)

                for j in range(peak_width):
                    pos = peak_pos + j - peak_width // 2
                    if 0 <= pos < sequence_length:
                        gaussian = np.exp(-0.5 * ((j - peak_width // 2) / (peak_width / 3)) ** 2)
                        signal[pos] += peak_height * gaussian
                label = 2

            X.append(signal)
            y.append(label)

        X = np.array(X).reshape(-1, sequence_length, 1)
        y = np.array(y)

        return X, y

    def train_model(self, epochs=50, batch_size=32):
        """
        Обучает модель на синтетических данных
        """
        if self.model is None:
            self.build_cnn_model()

        X_train, y_train = self.create_synthetic_data(2000)
        X_val, y_val = self.create_synthetic_data(500)

        history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            verbose=1
        )

        return history

    def predict_flash_type(self, signal_sequence):
        """
        Предсказывает тип вспышки
        """
        if self.model is None:
            self.build_cnn_model()
            # В реальном приложении здесь должна быть загрузка предобученной модели

        # Подготовка входных данных
        signal_array = np.array(signal_sequence).reshape(1, -1, 1)

        # Если длина не 100, интерполируем
        if signal_array.shape[1] != 100:
            from scipy import interpolate
            x_old = np.linspace(0, 1, len(signal_sequence))
            x_new = np.linspace(0, 1, 100)
            f = interpolate.interp1d(x_old, signal_sequence, kind='linear')
            signal_array = f(x_new).reshape(1, 100, 1)

        prediction = self.model.predict(signal_array, verbose=0)
        class_idx = np.argmax(prediction[0])

        class_names = {
            0: 'Фон (без вспышки)',
            1: 'Короткая вспышка',
            2: 'Длинная вспышка'
        }

        return class_names[class_idx], prediction[0]


def save_model(model, path='flash_model.h5'):
    """Сохраняет модель"""
    model.model.save(path)


def load_model(path='flash_model.h5'):
    """Загружает модель"""
    from tensorflow.keras.models import load_model
    try:
        detector = FlashDetectorModel()
        detector.model = load_model(path)
        return detector
    except:
        print(f"Модель {path} не найдена. Создаем новую.")
        detector = FlashDetectorModel()
        detector.build_cnn_model()
        return detector