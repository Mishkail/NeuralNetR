# 1. Подобрать котировки курса индекса (индекс каждый индивидуально)
# 2. Число наблюдений оставить 1200, где 1000 -тренировочная, 200 - тестовая 
# 3.
# обучить нейронные сети c помощью KERAS, TF: sequential model (SM), RNN, LSTM.
# Нормализация данных к-средних ?
# 4 Для metrics 'acc',
    # activation: 'sigmoid', 'relu',
    # optimizer: 'rmsprop', 'adam'
    # loss = 'MSE', 'MAE', 'MAPE'
# 5. Число скрытых слоев от 2
    #  batch_size = 125,
    # validation_split = от 0.12 до 0,25 
    # epochs = от 5
# 6. по прогнозам тестовой выборки найти 'MSE' и 'MAPE' для каждой нейронной сети
# 7. составить сранительную таблицу(data.frame), где по строкам архитектура сети:
    # По строкам SM(rmsprop), SM(adam), RNN (rmsprop), RNN(adam), lstm(rmsprop), lstm(adam)
    
    # По столбцам: 
    # Наименование: NN (SM, RNN, LSTM)
    # optimizer: rmsprop, adam 
    # 3 столбец: MSE
    # 4 столбец: MAPE
# 8. Подведение итогов по выбору нейронной сети и вывод графиков
# 9. Сохранить скрипт работы на github и отправить ссылку
# !!!!!!!!! Везде оставляйте комментарии о ходе работы :)

##############
# пример сравнительной таблицы
b <- data.frame('NN' = c(rep('SM', 2), rep('RNN', 2), rep('lstm', 2)), 
           'optimizer' = rep(c('rmsprop', 'adam'), 3),
           'MSE' = NA,
           'MAPE' = NA)
b
##############
# Пример группровки по таблице
aggregate(data = b, b$MSE ~ b$optimizer, max)

##############
