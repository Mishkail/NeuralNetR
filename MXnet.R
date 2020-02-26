# Установка библиотеки MXnet для глубокого обучения
getwd()

list.files()
dir.create('NN')
setwd("C:/Users/user/Documents/NN")

install.packages('MXnet')
install.packages("https://s3.ca-central-1.amazonaws.com/jeremiedb/share/mxnet/CPU/3.6/mxnet.zip", repos = NULL)

# дополнительные библиотеки для MXnet
install.packages('Rcpp')
install.packages('DiagrammeR')
library('Rcpp')
library('mxnet')


