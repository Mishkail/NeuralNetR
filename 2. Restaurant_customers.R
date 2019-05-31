# загружаем необходимые библиотеки
library(NeuralNetTools)
library(nnet)

# после обновления ПО R и RStudio  проверяем работоспособность пакета RWeka
# В прошлый раз (до обновления) пакет капризничал из за java 
library(RWeka) # для другой задачи

# устанавливаем рабочую директорию для сохранения
setwd('~/r/NeuralNetR')
getwd()

# загружаем данные из рабочей дирректории

mydata = read.csv('RestaurantTips.csv', sep = ';', header = T)
mydata

# Обращаемся к нашей таблице напрямую
attach(mydata)
names(mydata)

# создаем модель нейронной сети

model = nnet(CustomerWillTip ~ Service + Ambience + Food,
             data = mydata,
             size = c(3, 5),
             rang = 0.1,
             decay = 5e-2,
             maxit = 5000 )

# А сейчас взглянем на результаты нашей модели!

print(model)
plotnet(model)
garson(model)

# С помощью пакета NeuralNetTools мы моем лицезреть 
# новый вид нейронной сети и гистограмму влияния факторов 
# на результат нейронной сети 

model1 = nnet(CustomerWillTip ~ Service + Ambience + Food,
             data = mydata,
             size = 1,
             rang = 0.1,
             decay = 5e-2,
             maxit = 5000 )
plotnet(model1)

# прогноз

Service <- c(9, 5, 4)
Ambience <- c(7, 6, 8)
Food <- c(4, 6, 7)
test <- data.frame(Service, Ambience, Food)

predict(model1, newdata = test)
