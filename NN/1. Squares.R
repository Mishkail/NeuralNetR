# Example 1 Squares
getwd()
# подключаем пакет нейронных сетей
setwd("/Users/mishkail/r/NeuralNetR")
library('neuralnet')
# обращаемся к нашему файлу с данными
mydata = read.csv('Squares.csv', sep=';', header = TRUE)
mydata
attach(mydata) 
names(mydata)

# попробуем запустить нейронную сеть, зависимости output от input

model = neuralnet(formula = Output ~ Input,
                  data = mydata,
                  hidden = 10,
                  threshold = 0.01)
# посмотрим на результаты полученной модели
print(model)
# посмотрим на график
plot(model)
# Можем посмотреть реальные и предсказанные значения
final_output = cbind(Input, Output,
                     as.data.frame(model$net.result))
colnames(final_output) = c('Input', 'Expected_output',
                           'Neuralnet_output')
print(final_output)

# самостоятельно построить модель нн с 2 скрытыми слоями, в первом 9 нйронов, во втором 5 нейронов

# обучить нейронную сеть определять bacteria (logistic_reg)
# обучить нейронную сеть определять mpg
set.seed(5)


