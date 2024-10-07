import os
import numpy as np
from glob import glob
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, CSVLogger, TensorBoard
from tensorflow.keras.metrics import Recall, Precision
from data import load_data, tf_dataset
from model import build_model
from config.config import load_data_folder_path

# Реализация функции iou (intersection / union)
def iou(y_true, y_pred):
    def f(y_true, y_pred):
        intersection = (y_true * y_pred).sum()
        union = y_true.sum() + y_pred.sum() - intersection
        x = (intersection + 1e-15) / (union + 1e-15)
        x = x.astype(np.float32)
        return x
    return tf.numpy_function(f, [y_true, y_pred], tf.float32)


if __name__ == "__main__":
    np.random.seed(42)
    tf.random.set_seed(42)


    path = load_data_folder_path() # Папка с данными
    (train_x, train_y), (valid_x, valid_y), (test_x, test_y) = load_data(path) # Загрузка данных

    assert len(train_x) == len(train_y)
    assert len(valid_x) == len(valid_y)
    assert len(test_x) == len(test_y)

    ## Гиперпараметры
    batch = 16
    lr = 1e-3
    epochs = 40

    train_dataset = tf_dataset(train_x, train_y, batch=batch) # Создание Тензоров для обучения
    valid_dataset = tf_dataset(valid_x, valid_y, batch=batch) # Создание Тензоров для валидации

    model = build_model()

    opt = tf.keras.optimizers.Adam(lr) # Оптимизатор функции
    metrics = ["acc", Recall(), Precision(), iou] # Отслеживаемые метрики
    model.compile(loss="binary_crossentropy", optimizer=opt, metrics=metrics) # Сборка модели с параметрами

    callbacks = [
        ModelCheckpoint("files/model.h5"), # Файл для сохранения обученной модели
        ReduceLROnPlateau(monitor="val_loss", factor=0.1, patience=3), # Снижение скорости обучения при отсутствии улучшений
        CSVLogger("files/data.csv"), # Файл для сохранения метрик на каждой эпохе
        TensorBoard(), # Запись метрик
        EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True) # Остановка обучения при отсутсвии улучшений
    ]

    # Кол-во шагов (пакетов) для обучения в одной эпохе
    train_steps = len(train_x) // batch 
    valid_steps = len(valid_x) // batch

    if len(train_x) % batch != 0:
        train_steps += 1
    if len(valid_x) % batch != 0:
        valid_steps += 1

    model.fit(
        train_dataset,
        validation_data=valid_dataset,
        epochs=epochs,
        steps_per_epoch=train_steps,
        validation_steps=valid_steps,
        callbacks=callbacks
    ) # Запуск обучения