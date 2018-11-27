library(tidyverse)
a <- read_csv("fast-813281894-nn-prune-greedytrapezoid.csv")
ops <- a %>% group_by(epoch) %>% summarise(mean=mean(ops),sd=sd(ops))

ggplot(ops, aes(x=epoch, y=mean)) + geom_point()