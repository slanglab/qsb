library(tidyverse)
a <- read_csv("output/probs_oracles.csv")
a
ggplot(a, aes(x=oracle, y=p)) + geom_boxplot()
ggsave("latex/oracle_acceptabilty.pdf")
quit()
