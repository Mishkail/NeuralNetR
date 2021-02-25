# Example 1 Squares
getwd()
install.packages('neuralnet')
# подключаем пакет нейронных сетей
setwd("/Users/mishkail/r/NeuralNetR")
library('neuralnet')
list.files()
# обращаемся к нашему файлу с данными
mydata <-  read.csv('Squares.csv', sep=';', header = TRUE)
mydata
attach(mydata) 
names(mydata)

# попробуем запустить нейронную сеть, зависимости output от input

model <-  neuralnet(formula = Output ~ Input,
                  data = mydata,
                  hidden = c(10),
                  threshold = 0.01)
model2 <-  neuralnet(formula = Output ~ Input,
                    data = mydata,
                    hidden = c(5, 4),
                    threshold = 0.01)
# посмотрим на результаты полученной модели
print(model)
# посмотрим на график
plot(model)
plot(model2)
# Можем посмотреть реальные и предсказанные значения
model$net.result

final_output = cbind(Input, Output,
                     as.data.frame(model$net.result))
final_output
colnames(final_output) = c('Input', 'Expected_output',
                           'Neuralnet_output')
print(final_output)
dif <- (final_output$Expected_output- final_output$Neuralnet_output)^2
dif
MSE <- sum(dif)/length(dif) 

# самостоятельно построить модель нн с 2 скрытыми слоями, в первом 9 нйронов, во втором 5 нейронов
# Выбрать лучшую модель Из следующих
# (3,5,7), (2,6,3), (8), (2,7)
# set.seed(5)

# обучить нейронную сеть определять bacteria(MASS) (logistic_reg)
# обучить нейронную сеть определять mpg (mtcars)

set.seed(5)

install.packages('MASS')
library(MASS)

