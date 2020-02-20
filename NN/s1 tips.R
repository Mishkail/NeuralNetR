
install.packages('nnet')
install.packages('NeuralNetTools')

library('nnet')
library('NeuralNetTools')

getwd()
list.files()
dir.create('NN')

setwd("C:/Users/mag/Documents/NN")

rest_tips <- read.csv("Rest_Tips.csv", sep = ';', header = T)

View(rest_tips)
attach(rest_tips)

model <- nnet(CustomerWillTip ~ Service + Ambience + Food,
              size = 5,
              rang = 0.1,
              decay = 0.01,
              maxit = 5000
              )
print(model)

plotnet(model)
garson(model)
olden(model)

model1 <- nnet(CustomerWillTip ~ Service + Ambience + Food,
              skip = 1,
              size = 0,
              rang = 0.1,
              decay = 0.01,
              maxit = 5000
)

plotnet(model1)
garson(model1)
olden(model1)

model3 <- nnet(CustomerWillTip ~ Service + Ambience + Food,
              size = 1,
              rang = 0.1,
              decay = 0.01,
              maxit = 5000
)
garson(model3)
olden(model3)


model3$fitted.values
model3$residuals

t1 <- data.frame(model3$fitted.values, rest_tips$CustomerWillTip, 
                 model3$residuals)
View(t1)
t1

t2 <- data.frame(model$fitted.values,rest_tips$CustomerWillTip, 
                 model$residuals)

t2
MSE <- sqrt(sum((model$residuals)^2/length(model$residuals)))
MSE

MSE1 <- sqrt(sum((model3$residuals)^2/length(model3$residuals)))
MSE1
