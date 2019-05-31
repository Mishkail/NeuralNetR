# Задача на класификацию с помощью нейронной сети SOM Kohonen
# Самоорганизующаяся карта Кохонена ( Если попробовать перевести на русский)
# С помощью данной сети решаются задачи кластеризации
install.packages("kohonen")
library('kohonen')
data('wines')
str(wines)
head(wines)
View(wines)

set.seed(1)

som.wines <- som(scale(wines), grid = somgrid(5, 5, 'hexagonal'))
som.wines
dim(getCodes(som.wines))

plot(som.wines, main = 'Wine data Kohonen SOM')
# Взглянем на красивое отображение карт Кохонена
# Которые получились в ходе команды som для обьекта som.wines

# Избавимся от нашего красивого графика, чтобы не мучать память
graphics.off()
par(mfrow = c(1,1))
plot(som.wines, type = 'changes', main = 'Wine data SOM')
# На графике изображено среднее расстояние до 
# ближайшей единицы по сравнению с количеством итераций

# Далее зададим обучающую выборку 150 строк
# а оставшися 27 будут использоваться в качестве тестовой выборки

train <- sample(nrow(wines), 150)
X_train <- scale(wines[train,])
X_test <- scale(wines[-train,],
                center = attr(X_train, "scaled:center"),
                scale = attr(X_train, "scaled:center"))
train_data <- list(measurements = X_train,
                   vintages = vintages[train])
test_data <- list(measurements = X_test,
                  vintages = vintages[-train])

mygrid <- somgrid(5, 5, 'hexagonal')

som.wines <- supersom(train_data, grid = mygrid)                

som.predict <- predict(som.wines, newdata = test_data)
table(vintages[-train], som.predict$predictions[['vintages']])
map(som.wines)

график для обученно нейронной сети
plot(som.wines, main = 'Wine data Kohonen SOM')