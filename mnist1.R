
install.packages('tensorflow')
install.packages('keras')
install.packages('devtools')
library('devtools')

devtools::install_github("rstudio/tensorflow")
devtools::install_github("rstudio/keras")

tensorflow::install_tensorflow()

library('keras')
library('tensorflow')
use_condaenv("r-tensorflow")
Sys.setenv(KERAS_BACKEND="theano")
install_keras()
install_tensorflow()

mnist <- dataset_mnist()

# при установке KERAS единственное решение, котрое сработало - это 
# загрузка Rstudio из Anaconda Navigator

# на mac в терминале прописыаем anaconda-navigator, далее устанавливаем Rstudio
# после установки запускаем 
library('keras')
library('tensorflow')
install_keras()

# после этих действий mnist можно использовать
mnist <- dataset_mnist() # загружаем данные для распознавания чисел

# для удобства разиваем их на 4 оъекта
train_images <- mnist$train$x
train_labels <- mnist$train$y
test_images <- mnist$test$x
test_labels <- mnist$test$y

# строим архитектуру нейронной сети

network <- keras_model_sequential() %>%
  layer_dense(units = 512, activation = 'relu', input_shape = c(28*28)) %>%
  layer_dense(units = 10, activation = 'softmax')

# Добавляем для нейронной сети оптимизатор, функцию потерь, какие метрики выводить на экран (в примере выводится только точность)
network %>% compile(
  optimizer = 'rmsprop',
  loss = 'categorical_crossentropy',
  metrix = c('accuracy')
)

# Изначально масивы имеют размерность 60000, 28, 28, сами значения изменяются в пределах от 0 до 255
# для обучения нейронной сети потребуется преобразовать форму 60000, 28*28, а значения перевести в размерность от 0 до 1

train_images <- array_reshape(train_images, c(60000, 28*28)) # меняем размерность к матрице
train_images <- train_images/255 # меняем область значений

test_images <- array_reshape(test_images, c(10000, 28*28))
test_images <- test_images/255

# создаем категории для ярлыков

train_labels <- to_categorical(train_labels)
test_labels <- to_categorical(test_labels)

# поле подготовки данных тренируем нейронную сеть

network %>% fit(train_images, train_labels, epochs = 5, batch_size = 128)
