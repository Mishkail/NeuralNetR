install.packages('neuralnet')

library('neuralnet')
getwd()
a <- 1:10
sq <- data.frame(a,a^2 )
sq

names(sq) <- c('input', 'output')
sq
attach(sq)

set.seed(1)

w <- rnorm(16, mean = 0, sd = 0.5)

w

m <- neuralnet(output ~ input,
               data = sq,
               hidden = 5,
               startweights = w,
               threshold = 0.05,
               learningrate = 0.01)

plot(m)
result <- as.data.frame(m$net.result)

t3 <- data.frame(result,
                 output = sq$output,
                 residual = sq$output - result)

names(t3) <- c('res', 'output', 'residuals')
t3

detach(sq)

test <- data.frame(input=c(11,12))

predict(m,test)


mm <- neuralnet(output ~ input,
               data = sq,
               hidden = c(3, 5, 5),
               threshold = 0.05,
               learningrate = 0.01)

plot(mm)

