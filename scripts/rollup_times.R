library(tidyverse)
a <- read_csv("bottom_up_clean/all_times.csv")
a$time = log(a$time)
b <- a %>% group_by(method) %>% summarise(mean(time))

b$`mean(time)` = exp(b$`mean(time)`)

# A tibble: 5 x 2
#  method      `mean(time)`
#  <chr>              <dbl>
#1 ablated         0.00437
#2 additive        0.00480
#3 additive_nn     4.84
#4 ilp             0.0567
#5 random          0.000736


write_csv(b, "bottom_up_clean/all_times_rollup.csv")
