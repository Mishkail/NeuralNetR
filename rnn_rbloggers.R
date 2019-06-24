# rnn Keras

getwd()
setwd("C:/Users/user/Documents/R/NN/rnn-lstm")

library(keras)

install.packages('BatchGetSymbols')
install.packages('plotly')

library(BatchGetSymbols)
library(plotly)

tickers <- c('%5EIBEX')
first.date <- Sys.Date() - 360*15
last.date <- Sys.Date()

myts <- BatchGetSymbols(tickers = tickers,
                        first.date = first.date,
                        last.date = last.date,
                        cache.folder = file.path(tempdir(),
                                                 'BGS_Cache') ) # cache in tempdir()
myts$df.control
head(myts$df.tickers)

y <-  myts$df.tickers$price.close # результат - цена закрытия акции
head(y)

myts <-  data.frame(index = myts$df.tickers$ref.date, price = y, vol = myts$df.tickers$volume)
head(myts)

myts <-  myts[complete.cases(myts), ] # убираем все пропущенные строки
nrow(myts)
myts <-  myts[-seq(nrow(myts) - 3000), ]
myts$index <-  seq(nrow(myts))

plot_ly(myts, x = ~index, y = ~price, type = "scatter", mode = "markers", color = ~vol) 

acf(myts$price, lag.max = 3000) # Auto- and Cross- Covariance and -Correlation Function Estimation

datalags <-  10
train <-  myts[seq(2000 + datalags), ]
test <-  myts[2000 + datalags + seq(1000 + datalags), ]
batch.size <-  50

x.train <-  array(data = lag(cbind(train$price, train$vol), datalags)[-(1:datalags), ], dim = c(nrow(train) - datalags, datalags, 2))
y.train = array(data = train$price[-(1:datalags)], dim = c(nrow(train)-datalags, 1))

x.test <-  array(data = lag(cbind(test$vol, test$price), datalags)[-(1:datalags), ], dim = c(nrow(test) - datalags, datalags, 2))
y.test = array(data = test$price[-(1:datalags)], dim = c(nrow(test) - datalags, 1))


model <- keras_model_sequential()

model %>%
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


model %>%
  compile(loss = 'mae', optimizer = 'adam')

model

for(i in 1:2000){
  model %>% fit(x = x.train,
                y = y.train,
                batch_size = batch.size,
                epochs = 1,
                verbose = 0,
                shuffle = FALSE)
  model %>% reset_states()
}
