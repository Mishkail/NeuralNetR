# Базовые примеры по RNN
# пример 1
setwd("/Users/mishkail/r/NeuralNetR/Course/rnn")
library('keras')
library('tensorflow')


timesteps <- 100
input_features <- 32
output_features <- 64
random_array <- function(dim) {
    array(runif(prod(dim)), dim = dim)
}
inputs <- random_array(dim = c(timesteps, input_features))
state_t <- rep_len(0, length = c(output_features))
W <- random_array(dim = c(output_features, input_features))
U <- random_array(dim = c(output_features, output_features))
b <- random_array(dim = c(output_features, 1))

output_sequence <- array(0, dim = c(timesteps, output_features))
for (i in 1:nrow(inputs)) {
    input_t <- inputs[i,]
    output_t <- tanh(as.numeric((W %*% input_t) + (U %*% state_t) + b))
    output_sequence[i,] <- as.numeric(output_t)
    state_t <- output_t
}

table(state_t)


# пример 2 Keras
model <- keras_model_sequential() %>%
    layer_embedding(input_dim = 10000, output_dim = 32) %>%
    layer_simple_rnn(units = 32)
summary(model)

model1 <- keras_model_sequential() %>%
    layer_embedding(input_dim = 10000, output_dim = 32) %>%
    layer_simple_rnn(units = 32, return_sequences = TRUE)
summary(model1)

model2 <- keras_model_sequential() %>%
    layer_embedding(input_dim = 10000, output_dim = 32) %>%
    layer_simple_rnn(units = 32, return_sequences = TRUE) %>%
    layer_simple_rnn(units = 32, return_sequences = TRUE) %>%
    layer_simple_rnn(units = 32, return_sequences = TRUE) %>%
    layer_simple_rnn(units = 32)
summary(model2)

# Пример 3 Keras

############
В случае появления ошибки
Ошибка в py_call_impl(callable, dots$args, dots$keywords) :
    ValueError: Object arrays cannot be loaded when allow_pickle=False
либо
install_tensorflow(version="nightly")

либо откатить numpy через терминал

pip uninstall numpy
pip install --upgrade numpy==1.16.1
###########

library(keras)
max_features <- 10000
maxlen <- 500
batch_size <- 32
cat("Loading data...\n")
imdb <- dataset_imdb(num_words = max_features)
str(imdb) # посмотрим загруженный массив

# подготовка данных
c(c(input_train, y_train), c(input_test, y_test)) %<-% imdb
cat(length(input_train), "train sequences")
cat(length(input_test), "test sequences")

cat("Pad sequences (samples x time)\n")
input_train <- pad_sequences(input_train, maxlen = maxlen)
input_test <- pad_sequences(input_test, maxlen = maxlen)
cat("input_train shape:", dim(input_train), "\n")
cat("input_test shape:", dim(input_test), "\n")

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
    epochs = 10,
    batch_size = 128,
    validation_split = 0.2
)

?keras::fit()
# это уже дополнительно
max(-1:5, 0)
pmax(-1:5, 0)

# ниже пример для скалярного произведения
x <- 1:5
y <- 5:1

x*y # произведение векторов
x%*%y # скалярное произведение векторов



