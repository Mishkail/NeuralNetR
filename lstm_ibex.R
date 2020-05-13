# Пример по lstm для испанский индекс

install.packages('BatchGetSymbols')

library(plotly)
library(BatchGetSymbols)
library('keras')
library('tensorflow')




tickers <- c('%5EIBEX')
first.date <- Sys.Date() - 360*15
last.date <- Sys.Date()

# Загрузка данных
yts <- BatchGetSymbols(tickers = tickers,
                        first.date = first.date,
                        last.date = last.date,
                        cache.folder = file.path(tempdir(),
                        'BGS_Cache') )
# в случае ошибки функции BatchGetSymbols перезапусть Rstudio 


yts$df.control

str(yts$df.tickers)
str(yts$df.tickers$price.close)


# подготовка данных
y <-  yts$df.tickers$price.close
myts <-  data.frame(index = yts$df.tickers$ref.date, price = y, vol = yts$df.tickers$volume)
myts <-  myts[complete.cases(myts), ]
myts <-  myts[-seq(nrow(myts) - 3000), ]
myts$index <-  seq(nrow(myts))


# графмк курса
plot_ly(myts, x = ~index, y = ~price, type = "scatter", mode = "markers", color = ~vol)

# автокорреляция
acf(myts$price, lag.max = 3000)

# стандартизация данных
msd.price <-  c(mean(myts$price), sd(myts$price))
msd.vol <-  c(mean(myts$vol), sd(myts$vol))
myts$price <-  (myts$price - msd.price[1])/msd.price[2]
myts$vol <-  (myts$vol - msd.vol[1])/msd.vol[2]
summary(myts)

# разбиваем совокупность на тестовую и тренировочную 2/3 - тренировка и 1/3- тест
datalags = 10
train <-  myts[seq(2000 + datalags), ]
test <-  myts[2000 + datalags + seq(1000 + datalags), ]
batch.size <-  50

# размерность массива для lstm
# по X размерность (число наблюдений тренировочной/тестовой выборки, лаг, число переменных)
# по Y размерность (число наблюдений, число переменных на выходе)

x.train <-  array(data = lag(cbind(train$price, train$vol), datalags)[-(1:datalags), ], dim = c(nrow(train) - datalags, datalags, 2))
y.train = array(data = train$price[-(1:datalags)], dim = c(nrow(train)-datalags, 1))

dim(x.train)
dim(y.train)

x.test <-  array(data = lag(cbind(test$vol, test$price), datalags)[-(1:datalags), ], dim = c(nrow(test) - datalags, datalags, 2))
y.test <-  array(data = test$price[-(1:datalags)], dim = c(nrow(test) - datalags, 1))

# обучаем сеть
model <- keras_model_sequential()  %>%
    layer_lstm(units = 100,
               input_shape = c(datalags, 2),
               batch_size = batch.size,
               return_sequences = TRUE,
               stateful = TRUE) %>%
    layer_dropout(rate = 0.5) %>%
    layer_lstm(units = 50,
               return_sequences = FALSE,
               stateful = TRUE) %>%
    layer_dropout(rate = 0.5) %>%
    layer_dense(units = 1)

model

model %>%
    compile(loss = 'mse', optimizer = 'adam')


model %>% fit(x.train, y.train, epochs = 20, batch_size = batch.size)


# Возможно также построить цикл для обучения модели
# где можно задать обнуление показателей слоя после каждого обучения

for(i in 1:100){
    model %>% fit(x = x.train,
                  y = y.train,
                  batch_size = batch.size,
                  epochs = 5,
                  verbose = 0,
                  shuffle = FALSE)
    model %>% reset_states()
}



# предсказания )
pred_out <- model %>% predict(x.test, batch_size = batch.size) %>% .[,1]

# визуализация прогнозов модели
plot_ly(myts, x = ~index, y = ~price, type = "scatter", mode = "markers", color = ~vol) %>%
    add_trace(y = c(rep(NA, 2000), pred_out), x = myts$index, name = "LSTM prediction", color = 'black')

# график отклонений
plot(y.test - pred_out, type = 'line')

plot(x = y.test, y = pred_out)

###############################
1 - индекc DJ за 5 лет торги дневные
2- построить модель регресси, и lstm, валидация 80/20
3 - составить таблицу со значениями по тестовой выборке, и прогнозам регрессии и lstm
4 - построить графики прогнозов
5 - сравнить прогнозы моделей по MSE, MAPE
###############################
