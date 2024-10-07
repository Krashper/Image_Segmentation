import tensorflow as tf
from tensorflow.keras.layers import Conv2D, BatchNormalization, Activation, UpSampling2D, Input, MaxPool2D, Concatenate
from tensorflow.keras.models import Model


# Здесь приводится реализация модели U-Net

def conv_block(x, num_filters): # Блок с извлечением признаков
    x = Conv2D(num_filters, (3, 3), padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)

    x = Conv2D(num_filters, (3, 3), padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)

    return x


def build_model():
    size = 256 # Размер картинки
    num_filters = [32, 48, 64, 80, 96] # Кол-во фильтров на каждом блоке с извлечением признаков
    inputs = Input(shape=(size, size, 3)) # Входной слой

    skip_x = [] # Результат на выходе каждого блока
    x = inputs 

    ## Encoder
    for f in num_filters:
        x = conv_block(x, f)
        skip_x.append(x)
        x = MaxPool2D((2, 2))(x) # Уменьшение размерности

    ## Мост
    x = conv_block(x, num_filters[-1]) # Бутылочное горлышко (Encoder -> Decoder)

    num_filters.reverse() # В обратном порядке кол-во фильтров (для Decoder)
    skip_x.reverse() # В обратном порядке выходы блоков (для соединения)
    
    ## Decoder
    for i, f in enumerate(num_filters):
        x = UpSampling2D((2, 2))(x) # Увеличение размерность, Ex: (1, 2, 2, 1) -> (1, 4, 4, 1) 
        xs = skip_x[i] # Выход блока для соединения
        x = Concatenate()([x, xs]) # Соединение выхода блока с Encoder + Decoder
        x = conv_block(x, f) # Применение блока извлечения признаков
    
    ## Выход
    x = Conv2D(1, (1, 1), padding="same")(x)
    x = Activation("sigmoid")(x) # Выходной слой с функцией активации sigmoid (0 - пиксель не закрашен, 1 - закрашен)

    return Model(inputs, x)


if __name__ == "__main__":
    model = build_model()
    model.summary()
