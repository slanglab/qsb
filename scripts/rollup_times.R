library(tidyverse)
a <- read_csv("bottom_up_clean/all_times.csv")
b <- a %>% group_by(method) %>% summarise(mean(time))
c <- a %>% group_by(method) %>% summarise(sd(time))


# A tibble: 5 x 2
#  method      `mean(time)`
#  <chr>              <dbl>
#1 ablated         0.00437
#2 additive        0.00480
#3 additive_nn     4.84
#4 ilp             0.0567
#5 random          0.000736

d <- inner_join(b,c)

write_csv(d, "bottom_up_clean/all_times_rollup.csv")
