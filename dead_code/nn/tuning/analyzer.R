library(tidyverse)
a <-read_csv("tuning/tuner.csv")
a %>% select(-experiment) %>% select(-epoch) %>%filter(score > .8)
a %>% select(-experiment) %>% select(-epoch) %>%filter(score > .8) %>% select(-hidden_size)

