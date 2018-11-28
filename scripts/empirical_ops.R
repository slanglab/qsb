library(tidyverse)
a <- read_csv("fast-813281894-nn-prune-greedytrapezoid.csv")
sentence2ops <- a %>% group_by(sentence) %>% summarize(sum(ops))
sentence2len <- a %>% filter(epoch == 0) %>% select(sentence,ops)
sentence2len$len = sentence2len$ops
sentence2ops$observed_ops <- sentence2ops$`sum(ops)` 
sentence2ops <- sentence2ops %>% select(sentence, observed_ops)
sentence2len <- sentence2len %>% select(sentence, len)


len2observedops <- inner_join(sentence2ops, sentence2len) %>% group_by(len) %>% summarize(mean(observed_ops))
len2observedops$theory <- len2observedops$len * len2observedops$len

observed <- len2observedops %>% select(len, `mean(observed_ops)`)
observed$kind <- "observed"
observed$ops <- observed$`mean(observed_ops)`
observed <- observed %>% select(len, kind, ops)


theory <- len2observedops %>% select(len, theory)
theory$kind <- "Worst-Case"
theory$ops <- theory$theory
theory <- theory %>% select(len, kind, ops)

all_ <- rbind(observed,theory)

ggplot(all_, aes(x=len, y=ops, color=kind, shape=kind)) + geom_point()

ggsave("latex/observed.pdf")