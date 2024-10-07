import os
import numpy as np
from glob import glob
import tensorflow as tf
from sklearn.model_selection import train_test_split
from typing import Tuple
import cv2
from config.config import load_data_folder_path


def load_data(path: str = "", split: float = 0.1) -> Tuple[
        Tuple[str, str],
        Tuple[str, str],
        Tuple[str, str]
    ]:
    # Загрузка данных

    images = sorted(glob(os.path.join(path, "images/*"))) # Список названий картинок
    masks = sorted(glob(os.path.join(path, "masks/*"))) # Список названий масок

    train_x, valid_x = train_test_split(images, test_size=split, random_state=42) # Разбиение на train и validation данные (картинки)
    train_y, valid_y = train_test_split(masks, test_size=split, random_state=42) # Разбиение на train и validation данные (маски)

    train_x, test_x = train_test_split(train_x, test_size=split, random_state=42) # Разбиение на train и test данные (картинки)
    train_y, test_y = train_test_split(train_y, test_size=split, random_state=42) # Разбиение на train и test данные (маски)

    return (train_x, train_y), (valid_x, valid_y), (test_x, test_y)


def read_image(path: str) -> np.ndarray:
    # Чтение и преобразование картинки в нормированную матрицу

    path = path.decode() # Название в "понятный вид"
    x = cv2.imread(path,cv2.IMREAD_COLOR) # Чтение картинки с настоящими цветами
    x = x/255.0 # Нормирование каждого пикселя
    # (256, 256, 3)
    return x


def read_mask(path: str) -> np.ndarray:
    # Чтение и преобразование маски в нормированную матрицу

    path = path.decode() # Название в "понятный вид"
    x = cv2.imread(path,cv2.IMREAD_GRAYSCALE) # Чтение картинки с чёрно-белыми цветами
    x = x/255.0 # Нормирование каждого пикселя
    # (256, 256)
    x = np.expand_dims(x, axis=-1) 
    # (256, 256, 1)
    return x


def tf_parse(x, y):
    # Преоразование данных в вид (Картинка, Маска)
    def _parse(x, y):
        # Получение нормированных матриц
        x = read_image(x)
        y = read_mask(y)
        return x, y
    
    x, y = tf.numpy_function(_parse, [x, y], [np.float64, np.float64]) # Использование numpy функции в качестве слоя Tensorflow
    x.set_shape((256, 256, 3)) # Преобразуем на всякий случай в нужный вид
    y.set_shape((256, 256, 1))

    return x, y


def tf_dataset(x, y, batch=8):
    # Преобразование в Tensor'ы по пакетам
    dataset = tf.data.Dataset.from_tensor_slices((x, y)) # Преобразуем данные в Тензоры
    dataset = dataset.map(tf_parse) # Преобразование данных в вид (Картинка, Маска)
    dataset = dataset.batch(batch) # Список данных по размеру batch
    dataset = dataset.repeat() # Бесконечный датасет (в виде последний элемент -> первый)
    return dataset


if __name__ == "__main__":
    path = load_data_folder_path() # Путь до папки с данными
    (train_x, train_y), (valid_x, valid_y), (test_x, test_y) = load_data(path)

    assert len(train_x) == len(train_y)
    assert len(valid_x) == len(valid_y)
    assert len(test_x) == len(test_y)

    ds = tf_dataset(train_x, train_y)
    for x, y in ds:
        print(x.shape, y.shape)
        break