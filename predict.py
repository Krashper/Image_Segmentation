import os
import numpy as np
import cv2
import tensorflow as tf
from tqdm import tqdm
from tensorflow.keras.utils import CustomObjectScope
from data import load_data, tf_dataset
from train import iou
from config.config import load_data_folder_path


def read_image(path: str) -> np.ndarray:
    # Преобразование картинки
    x = cv2.imread(path,cv2.IMREAD_COLOR)
    x = x/255.0
    # (256, 256, 3)
    return x


def read_mask(path: str) -> np.ndarray:
    # Преобразование маски
    x = cv2.imread(path,cv2.IMREAD_GRAYSCALE)
    # (256, 256)
    x = np.expand_dims(x, axis=-1)
    # (256, 256, 1)
    return x


def mask_parse(mask):
    # Преобразование маски для визуализации в matplotlib
    mask = np.squeeze(mask)
    mask = [mask, mask, mask]
    mask = np.transpose(mask, (1, 2, 0))
    return mask


if __name__ == "__main__":
    print("")
    path = load_data_folder_path() # Папка с данными
    batch = 8 # Размер пакета

    (train_x, train_y), (valid_x, valid_y), (test_x, test_y) = load_data(path) # Загрузка данных
    print(len(train_x), len(valid_x), len(test_x))

    test_dataset = tf_dataset(test_x, test_y, batch=batch) # Преобразование в Тензоры для тестирования
    test_steps = len(test_x)//batch # Кол-во шагов (пакетов) в одной эпохе
    if len(test_x) % batch != 0:
        test_steps += 1

    with CustomObjectScope({"iou": iou}): # Опираемся на iou при загрузке модели (чтобы избежать ошибку)
        model = tf.keras.models.load_model("files/model.h5")

    model.evaluate(test_dataset, steps=test_steps) # Оценка модели

    for i, (x, y) in tqdm(enumerate(zip(test_x, test_y)), total=len(test_x)):
        x = read_image(x)
        y = read_mask(y)
        y_pred = model.predict(np.expand_dims(x, axis=0)) # Предсказание модели для пикселя
        y_pred = y_pred[0] > 0.6 # 1, если > 0.6, иначе - 0
        h, w, _ = x.shape
        
        white_line = np.ones((h, 10, 3)) * 255.0 # Разделитель для изображений

        all_images = [
            x * 255.0, white_line,
            mask_parse(y), white_line,
            mask_parse(y_pred) * 255.0
        ] # Исходное изображение | Маска оригинальная | Маска предсказанная

        image = np.concatenate(all_images, axis=1)
        cv2.imwrite(f"results/{i}.png", image) # Сохранение изображения