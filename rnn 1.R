# Базовые примеры по RNN
# пример 1
setwd("/Users/mishkail/r/NeuralNetR/Course/rnn")
library('keras')
library('tensorflow')

###################################
пример rnn
###################################


timesteps <- 100 # количество временных шагов
input_features <- 32 # число входных переменных
output_features <- 64 # число исходящих переменных

# пишем свою функцию по заполнению массива с помощью равномерного распределения
# array- создание массива
# runif - функция равномерного рапределения (по умолчанию от 0 до 1)
# prod - произведение всех элементов

random_array <- function(dim) {
    array(runif(prod(dim)), dim = dim)
}

# Создаем массив размерностью 100*32 со случайными числами
input <- random_array(dim = c(timesteps, input_features))

str(input) # структура объекта
hist(input[,1]) # гистограмма первого столбца равномерного распределения

# Создаем обьект начального или нулевого состояния нашей сети
state_t <- rep_len(0, length = c(output_features))

# Создаем вектор весов
W <- random_array(dim = c(output_features, input_features))
# Вектор выходных данных предыдущего шага
U <- random_array(dim = c(output_features, output_features))
# вектор констант(смещений)
b <- random_array(dim = c(output_features, 1))

# Функция обновления слоя нейронов на предыдущие состояния выходных данных
# y = b + w*input + u*output_t-1

# пустая выходная матрица
output_seq <- array(0, dim = c(timesteps, output_features))

# запускаем цикл обновления значений предыдущего выходного массива
# на вектор весов и входных переменных
for (i in 1:nrow(input)) {
    input_t <- input[i,]
    output_t <- tanh(as.numeric((W %*% input_t) + (U %*% state_t) + b))
    output_seq[i,] <- as.numeric(output_t)
    state_t <- output_t
}

output_seq
str(output_seq)
table(state_t)
plot(state_t)

# Добавить пример с rnorm

###################################
# пример вывода результатов в Keras
###################################

# двумерный массив
model <- keras_model_sequential() %>%
    layer_embedding(input_dim = 10000, output_dim = 32) %>%
    layer_simple_rnn(units = 32)
summary(model)
# трехмерный массив
model1 <- keras_model_sequential() %>%
    layer_embedding(input_dim = 10000, output_dim = 32) %>%
    layer_simple_rnn(units = 32, return_sequences = TRUE)
summary(model1)

# двумерны массив на выходе, но в промежуточных слоях реккурентная сеть учитывает
# все состояния сети
model2 <- keras_model_sequential() %>%
    layer_embedding(input_dim = 10000, output_dim = 32) %>%
    layer_simple_rnn(units = 32, return_sequences = TRUE) %>%
    layer_simple_rnn(units = 32, return_sequences = TRUE) %>%
    layer_simple_rnn(units = 32, return_sequences = TRUE) %>%
    layer_simple_rnn(units = 32)
summary(model2)

###################################
# Пример 3 Keras
###################################

###################################
В случае появления ошибки
Ошибка в py_call_impl(callable, dots$args, dots$keywords) :
    ValueError: Object arrays cannot be loaded when allow_pickle=False
либо
install_tensorflow(version = "nightly")

либо откатить numpy через терминал

pip uninstall numpy
pip install --upgrade numpy==1.16.1

либо 

devtools::install_github("rstudio/keras")
###################################

library(keras)
max_features <- 10000
maxlen <- 500
batch_size <- 32

# загружаем данные imdb
imdb <- dataset_imdb(num_words = max_features)
imdb$train$y
str(imdb) # посмотрим загруженный массив

# Набор данных из 25 000 обзоров фильмов из IMDB, 
# помеченных настроением (положительный / отрицательный). 
# Обзоры были предварительно обработаны, и каждый отзыв закодирован
# как последовательность индексов слов (целых чисел). 
# Для удобства слова индексируются по общей частоте в наборе данных, 
# так что, например, целое число «3» кодирует третье наиболее часто встречающееся
# слово в данных.

# Данные x включают целочисленные последовательности. 
# Если аргумент num_words был определен, 
# максимально возможное значение индекса равно num_words-1. 
# Если указан аргумент maxlen`, наибольшая возможная длина последовательности - maxlen '.
# Данные y включают в себя набор целочисленных меток (0 или 1).



# подготовка данных. разделим наборы на тестовые и тренирововчные
c(c(input_train, y_train), c(input_test, y_test)) %<-% imdb


# Добавляем число шагов обучения 
input_train <- pad_sequences(input_train, maxlen = maxlen)
input_test <- pad_sequences(input_test, maxlen = maxlen)
25000*500
dim(input_train)
dim(input_test)


# тренировка реккурентной сети simple RNN

model <- keras_model_sequential() %>%
    layer_embedding(input_dim = max_features, output_dim = 32) %>%
    layer_simple_rnn(units = 32) %>%
    layer_dense(units = 1, activation = "sigmoid")
model %>% compile(
    optimizer = "rmsprop",
    loss = "binary_crossentropy",
    metrics = c("acc")
)

history <- model %>% fit(
    input_train, y_train,
    epochs = 5,
    batch_size = 128,
    validation_split = 0.2
)



plot(history)

?keras::fit()
# это уже дополнительно
max(-1:5, 0)
pmax(-1:5, 0)

# ниже пример для скалярного произведения
x <- 1:5
y <- 5:1

x*y # произведение векторов
x%*%y # скалярное произведение векторов



