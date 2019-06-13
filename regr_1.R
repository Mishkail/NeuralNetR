# Regression with Keras example 1
setwd("/Users/mishkail/r/NeuralNetR/course/regression")

#В основе Keras лежит работа c тензорами - это обобщенное наименование для векторов, матриц и тд
#Поэтому при работе в R с реальными данными необходимо dataframe и datatable преобразовать в матрицу
library(keras)
dataset <- dataset_boston_housing()
c(c(train_data, train_targets), c(test_data, test_targets)) %<-% dataset

# %<-% оператор мультиприсвоения, в одну строку заменяет написание нескольких строк, доступен в пакете Keras

Так как исходные данные имеют различные единицы измерения мы стандартизируем их относительно
математического ожидания по отношению к стандартному отклонению.
Для шкалирования потребуется строго шкалирование по тренировочным данным. 
Шкалирование тестовых данных осуществляется на основе оценок тренировочной выборки.


mean <- apply(train_data, 2, mean)
std <- apply(train_data, 2, sd)

Стандартизиорванные данные:
Стандартизация осуществляется с помощью оценок z-score

train_data <- scale(train_data, center = mean, scale = std)
test_data <- scale(test_data, center = mean, scale = std)

# Построение модели нейронной сети

build_model <- function() {
    model <- keras_model_sequential() %>%
        layer_dense(units = 64, activation = "relu",
                    input_shape = dim(train_data)[[2]]) %>%
        layer_dense(units = 64, activation = "relu") %>%
        layer_dense(units = 1)
    model %>% compile(
        optimizer = "rmsprop",
        loss = "mse",
        metrics = c("mae")
    ) }

sample()

# Кросс-валидация

k <- 4
indices <- sample(1:nrow(train_data)) # перемешиваем значения случайно
folds <- cut(indices, breaks = k, labels = F) # разбиваем выборку на 4 группы
folds

num_epochs <- 100
all_scores <- NULL
for (i in 1:k) {
    cat("processing fold #", i, "\n")
    
    val_indices <- which(folds == i, arr.ind = TRUE)
    val_data <- train_data[val_indices,]
    val_targets <- train_targets[val_indices]
    
    
    partial_train_data <- train_data[-val_indices,]
    partial_train_targets <- train_targets[-val_indices]
    
    model <- build_model()
    
    model %>% fit(partial_train_data, partial_train_targets,
                  epochs = num_epochs, batch_size = 1, verbose = 0)
    
    results <- model %>% evaluate(val_data, val_targets, verbose = 0)
    all_scores <- c(all_scores, results$mean_absolute_error)
}

all_scores
mean(all_scores)

# с целью уменьшения ошибки число эпох увеличиваем до 500

num_epochs <- 500
all_mae_histories <- NULL

for (i in 1:k) {
    cat("processing fold #", i, "\n")
    val_indices <- which(folds == i, arr.ind = TRUE)
    val_data <- train_data[val_indices,]
    val_targets <- train_targets[val_indices]
    
    partial_train_data <- train_data[-val_indices,]
    partial_train_targets <- train_targets[-val_indices]
    
    model <- build_model()
    
    history <- model %>% fit(
        partial_train_data, partial_train_targets,
        validation_data = list(val_data, val_targets),
        epochs = num_epochs, batch_size = 1, verbose = 0
    )
    mae_history <- history$metrics$val_mean_absolute_error
    all_mae_histories <- rbind(all_mae_histories, mae_history)
}

# результаты кросс валидации
average_mae_history <- data.frame(
    epoch = seq(1:ncol(all_mae_histories)),
    validation_mae = apply(all_mae_histories, 2, mean)
)

library(ggplot2)

# визуализация ошибки MAE в течение обучения нейронной сети 
ggplot(average_mae_history, aes(x = epoch, y = validation_mae)) + geom_line()

ggplot(average_mae_history, aes(x = epoch, y = validation_mae)) + geom_smooth()

model <- build_model()
model %>% fit(train_data, train_targets,
              epochs = 80, batch_size = 16, verbose = 0)
result <- model %>% evaluate(test_data, test_targets)

result$loss
result$mean_absolute_error

# В качестве функции потерь (loss function) удобно использовать MSE
# В качестве качества оценки регрессии (metrics of accuracy) удобно использовать MAE




