library('lubridate')

library('lmtest')
library('sandwich')
library('car')
library('bstats')
library('zoo')
library('xts')
library('dplyr')
library('broom')
library('ggplot2')

library('quantmod')
library('rusquant')
library('sophisthse')
library('Quandl')
library('keras')
getwd()
setwd("C:/Users/user/Documents/R/NN")
dir.create('NN')
list.files()
getSymbols(Symbols = c('AAPL','AXP','HD', 'CVX', 'MCD', 'MSFT',
                       'KO', 'XOM', 'GS', 'CAT', 'JNJ', 'MRK',
                       'CSCO', 'UNH', 'TRV', 'PG', 'WMT', 'JPM',
                       'PFE', 'VZ', 'V', 'UTX', 'MMM',
                       'WBA', 'IBM', 'DIS', 'DWDP', 'BA', 'INTC'),
           from = '2012-12-06', to = '2017-12-06', src = 'yahoo')
# акции Nike не загрузились, поэтому в данном исследовании убираем данный актив из выборки

# для случайного генератора также лучше добавить опцию set.seed для проверки
set.seed(5)

# обьединяем акции в одной таблице
work_data <- cbind.xts(AAPL,AXP, HD, CVX, MCD, MSFT,
                       KO, XOM, GS, CAT, JNJ, MRK,
                       CSCO, UNH, TRV, PG, WMT, JPM,
                       PFE, VZ, V, UTX, MMM,
                       WBA, IBM, DIS, DWDP, BA, INTC)
dim(work_data)
write.csv2(work_data, file = 'work_data.csv') # сохранили таблицу данных в исходном формате
# убираем из совокупности все столбцы, в которых содержатся adjusted

n <- names(work_data)
grep('Adjusted', n) 
work_data1 <- work_data[,-grep('Adjusted', n)] 
names(work_data1) # проверяем сработал или нет

write.csv2(work_data1, file = 'work_data_good.csv') # сохранили таблицу данных в формате для анализа

library(keras)

# разбиваем таблицу на результат и факторы
y <- work_data1$AAPL.Close 
work_data1 <- work_data1[,-4]
head(work_data1)

# разбиваем совокупность на тренировочную и тестовую выборки
train <- sample(nrow(work_data1), 0.8*nrow(work_data1))

# шкалируем по train 
X_train <- scale(work_data1[train,])

X_test <- scale(work_data1[-train,],
                center = attr(X_train, "scaled:center"),
                scale = attr(X_train, "scaled:center"))

y_train <- scale(y[train])
y_test <- scale(y[-train],
                center = attr(y_train, "scaled:center"),
                scale = attr(y_train, "scaled:center"))
# приводим к матричному виду
y1 <- as.matrix(y_train)
x1 <- as.matrix(X_train)
y2 <- as.matrix(y_test)
x2 <- as.matrix(X_test)
dim(x1)
dim(y1)
dim(x2)
plot(y)

dim(X_train)
# база для нейронной сети
model <- keras_model_sequential() %>%
  layer_dense(units = 64, activation = "relu",
              input_shape = dim(x1)[[2]]) %>%
  layer_dense(units = 64, activation = "relu") %>%
  layer_dense(units = 1)
model %>% compile(
  optimizer = "rmsprop",
  loss = "mse",
  metrics = c("mae")
) 

# обучаем нейронную сеть

history <- model %>% fit(
  x1, y1,
  epochs = 100,
  batch_size = 16,
  validation_split = 0.2
  )

plot(history)  

# Проверяем результаты тестовой выборки
result <- model %>% evaluate(x2, y2)

test_predictions <- model %>% predict(X_test) # строим предсказанные значения
predictions <- data.frame(test_predictions, y_test) 
head(predictions)
# график предсказанных и фактических значений
ggplot(predictions, aes(x = 1:length(y_test))) +                    # basic graphical object
  geom_line(aes(y=test_predictions,linetype ="test_predictions"), colour="black") +  # first layer
  geom_line(aes(y=y_test, linetype ="y_test"), colour="red")+
  xlab("Наблюдения")+
  ylab("Фактические/прогнозные значения")+
  scale_linetype_manual(values =  c(2,1))+
  labs(title="Targets/Predictions")
  
  

result$loss
result$mean_absolute_error
summary(model)

#################################
# трехслойная нейронная сеть

model2 <- keras_model_sequential() %>%
  layer_dense(units = 64, activation = "relu",
              input_shape = dim(x1)[[2]]) %>%
  layer_dense(units = 64, activation = "relu") %>%
  layer_dense(units = 64, activation = "relu") %>%
  layer_dense(units = 1)

model2 %>% compile(
  optimizer = "rmsprop",
  loss = "mse",
  metrics = c("mae")
) 
history$metrics
# обучаем нейронную сеть

history2 <- model2 %>% fit(
  x1, y1,
  epochs = 100,
  batch_size = 16,
  validation_split = 0.2
)
# ниже команда только для замера времени выполнения
system.time(model2 %>% fit(
  x1, y1,
  epochs = 100,
  batch_size = 16,
  validation_split = 0.2
))
plot(history2)  

# Проверяем результаты тестовой выборки
result2 <- model2 %>% evaluate(x2, y2)

result2$loss
result2$mean_absolute_error
summary(model2)

####################

######################
инс 2 для замера времени
model3 <- keras_model_sequential() %>%
  layer_dense(units = 64, activation = "relu",
              input_shape = dim(x1)[[2]]) %>%
  layer_dense(units = 64, activation = "relu") %>%
  layer_dense(units = 1)
model3 %>% compile(
  optimizer = "rmsprop",
  loss = "mse",
  metrics = c("mae")
) 

# обучаем нейронную сеть

history3 <- model3 %>% fit(
  x1, y1,
  epochs = 100,
  batch_size = 16,
  validation_split = 0.2
)

system.time(model3 %>% fit(
  x1, y1,
  epochs = 100,
  batch_size = 16,
  validation_split = 0.2
))


plot(history)  
#################
однослойный персептрон

model4 <- keras_model_sequential() %>%
  layer_dense(units = 64, activation = "relu",
              input_shape = dim(x1)[[2]]) %>%
  layer_dense(units = 1)

model4 %>% compile(
  optimizer = "rmsprop",
  loss = "mse",
  metrics = c("mae")
) 
history$metrics
# обучаем нейронную сеть

history4 <- model4 %>% fit(
  x1, y1,
  epochs = 100,
  batch_size = 16,
  validation_split = 0.2
)
system.time(model4 %>% fit(
  x1, y1,
  epochs = 100,
  batch_size = 16,
  validation_split = 0.2
))
plot(history4)  

# Проверяем результаты тестовой выборки
result4 <- model4 %>% evaluate(x2, y2)

result4$loss
result4$mean_absolute_error
summary(model4)



###############
# Проверяем результаты тестовой выборки
result <- model %>% evaluate(x2, y2)

result$loss
result$mean_absolute_error
summary(model)



#####################


# Все, что ниже не работает

dim(x1r)
shape(x1r)
# rnn 
timestep <- 1
x1r <- array_reshape(x1, c(dim(x1)[[1]], 1, dim(x1)[[2]]))
y1r <- array_reshape(y1, c(dim(y1)[[1]],1, dim(y1)[[2]]))
shape(x1r)
model2 <- keras_model_sequential() %>%
  layer_simple_rnn(units = 64,input_shape = c(1, dim(x1r)[[3]])) %>%
  layer_dense(units = 1, activation = "relu")
model2 %>% compile(
  optimizer = "rmsprop",
  loss = "mse",
  metrics = c("mae")
) 


# обучаем нейронную сеть

history <- model2 %>% fit(
  x1r, y1r,
  epochs = 100,
  validation_split = 0.2,
)

plot(history)  

####################
# пример rnn
install.packages('rnn')

library('rnn')

X1 = sample(0:127, 10000, replace=TRUE)
X2 = sample(0:127, 10000, replace=TRUE)

Y <- X1 + X2

X1 <- int2bin(X1, length=8)
X2 <- int2bin(X2, length=8)
Y  <- int2bin(Y,  length=8)

X <- array( c(X1,X2), dim=c(dim(X1),2) )

model <- trainr(Y=Y,
                X=X,
                learningrate   =  1,
                hidden_dim     = 16  )
###########################

# визуализация нейронной сети

devtools::install_github("andrie/deepviz")

library(deepviz)
library(magrittr)

c(13, 10, 1) %>% 
  plot_deepviz()

c(13, 10, 10, 1) %>% 
  plot_deepviz()

c(10, 10, 10, 5, 1) %>% 
  plot_deepviz()
