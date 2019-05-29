# Набор данных и премер, по распознаванию рукописных чисел MNIST, предложенный автором KERAS Ф. Шолле
# Наборы данных оцифрованных рукописных цифр, каждая цифра представляла собой матрицу чисел 28*28
# каждая ячейка это число от 0 до 255 в зависимости от контрастности(все числа были черно белые).
# Таких матриц для тренировочной выборки было 60000, а также было 60000 чисел,
# каждому из которых соответствовала отдельная матрица.
# По этим соотношениям находились связи.
# Для тренировочной выборки подобраны 10000 матриц (28*28) и 10000 чисел

# лучше устанавливать библиотеки последних версий через github с помощью devtools
install.packages('devtools')
library('devtools')
devtools::install_github("rstudio/tensorflow")
devtools::install_github("rstudio/keras")

# или с CRAN, но на CRAN может быть не самая последняя версия
install.packages('tensorflow')
install.packages('keras')

# запускаем библиотеки

library('keras')
library('tensorflow')

# для решения некоторых ошибок порой помогает выполнение этой команды,
# если ошибок не было, выполнть не обязательно
use_condaenv("r-tensorflow")

# Устанавливаем Keras
install_keras()

# если вдруг возникает ошибка '' то лучше откатить версию tensorflow до предыдущей
# если ошибок не было, выполнть не обязательно 
install_tensorflow(version = '1.12')

# загрузка данных
mnist <- dataset_mnist()

##################
# Внутри ограничений действия, если dataset mnist не был найден
# если ошибок не было, выполнть не обязательно 

# Проблема, возникшая на Windows 10, dataset mnist не был найден
# при установке KERAS единственное решение, котрое сработало - это 
# загрузка Rstudio из Anaconda Navigator

# на mac в терминале прописыаем anaconda-navigator, далее устанавливаем Rstudio
# после установки запускаем 
library('keras')
library('tensorflow')
install_keras()

# после этих действий mnist можно использовать
mnist <- dataset_mnist() # загружаем данные для распознавания чисел
################


# для удобства разиваем их на 4 объекта
train_images <- mnist$train$x
train_labels <- mnist$train$y
test_images <- mnist$test$x
test_labels <- mnist$test$y

# строим архитектуру нейронной сети

network <- keras_model_sequential() %>%
    layer_dense(units = 512, activation = 'relu', input_shape = c(28*28)) %>%
    layer_dense(units = 10, activation = 'softmax')

# Добавляем для нейронной сети оптимизатор, функцию потерь, какие метрики выводить на экран (в примере выводится только точность)
network %>% compile(
    optimizer = 'rmsprop',
    loss = 'categorical_crossentropy',
    metrics = c('accuracy')
)

# Изначально массивы имеют размерность 60000, 28, 28, сами значения изменяются в пределах от 0 до 255
# для обучения нейронной сети потребуется преобразовать форму 60000, 28*28, а значения перевести в размерность от 0 до 1

train_images <- array_reshape(train_images, c(60000, 28*28)) # меняем размерность к матрице
train_images <- train_images/255 # меняем область значений

test_images <- array_reshape(test_images, c(10000, 28*28))
test_images <- test_images/255

# создаем категории для ярлыков

train_labels <- to_categorical(train_labels)
test_labels <- to_categorical(test_labels)

# поле подготовки данных тренируем нейронную сеть

network %>% fit(train_images, train_labels, epochs = 5, batch_size = 128)

# точность модели составила 98,9%

metric <- network %>% evaluate(test_images, test_labels)
metric

# по тестовой выборке 97,8
# расхождения в точности связаны с таким эффектом как переобучение или overfitting
# Под переобучением понимается ситуация, при которой точность или прогнозов по тренировочной выборке 
# высокая, а по тестовой выборке результаты ниже. Это связано с возникновением случайных взаимосвязей
# между случайно собранных наблюдений для тренировочной выборки.

# Предсказание определенных значений можно выполнить довольно просто:
    
network %>% predict_classes(test_images[1:15,])
