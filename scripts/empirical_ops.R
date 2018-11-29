library(tidyverse)
a <- read_csv("output/full-556251071-nn-prune-greedytrapezoid.csv")


## 
sentence2ops <- a %>% group_by(sentence) %>% summarize(sum(ops))
sentence2len <- a %>% filter(epoch == 0) %>% select(sentence,ops)
sentence2len$len = sentence2len$ops
sentence2ops$observed_ops <- sentence2ops$`sum(ops)` 
sentence2ops <- sentence2ops %>% select(sentence, observed_ops)
sentence2len <- sentence2len %>% select(sentence, len)

len2observedops <- inner_join(sentence2ops, sentence2len) %>% group_by(len) %>% summarize(mean(observed_ops))
len2observedops$theory <- len2observedops$len * len2observedops$len

observed <- len2observedops %>% select(len, `mean(observed_ops)`)
observed$kind <- "Empirical performance"
observed$ops <- observed$`mean(observed_ops)`
observed <- observed %>% select(len, kind, ops)

theory <- len2observedops %>% select(len, theory)
theory$kind <- "Theoretical: worst-case"
theory$ops <- theory$theory
theory <- theory %>% select(len, kind, ops)

## 
a <- read_csv("output/full-worst-case-worst-case-compressortrapezoid.csv")
sentence2ops <- a %>% group_by(sentence) %>% summarize(sum(ops))
sentence2len <- a %>% filter(epoch == 0) %>% select(sentence,ops)
sentence2len$len = sentence2len$ops
sentence2ops$observed_ops <- sentence2ops$`sum(ops)` 
sentence2ops <- sentence2ops %>% select(sentence, observed_ops)
sentence2len <- sentence2len %>% select(sentence, len)
len2observedops <- inner_join(sentence2ops, sentence2len) %>% group_by(len) %>% summarize(mean(observed_ops))
len2observedops$theory <- len2observedops$len * len2observedops$len
observed_worst <- len2observedops %>% select(len, `mean(observed_ops)`)
observed_worst$kind <- "Worst-case performance"
observed_worst$ops <- observed_worst$`mean(observed_ops)`
observed_worst <- observed_worst %>% select(len, kind, ops)



all_ <- rbind(observed, observed_worst)  # ,theory

# The palette with black:
cbPalette <- c("black", "red") #red", "#56B4E9", "#009E73", "#F0E442", "#0072B2", "#D55E00", "#CC79A7")

ggplot(all_ %>% filter(len < 30), aes(x=len, y=ops, color=kind, shape=kind)) + geom_line(size = 3, aes(linetype=kind, color=kind)) + ylab("Operations") + xlab("Sentence length") + 
       theme_bw() + theme(legend.title=element_blank(),
                            legend.position = "bottom",
                            legend.spacing.x = unit(.5, 'cm'),
                            axis.title=element_text(size=13,face="bold"),
                            legend.text=element_text(size=13,face="bold"), 
                           axis.text=element_text(size=13, face="bold")) +
                            scale_colour_manual(values=cbPalette)

ggsave("latex/observed.pdf")

#z <- a %>% filter(sentence == 14)
#ggplot(z, aes(x=epoch, y=ops)) + geom_point()
#ggsave("latex/single.pdf")