library(tidyverse)
a <- read_csv("bottom_up_clean/all_times.csv")
a$time <- log10(a$time)
b <- a%>% filter(method=="additive" |  method == "ilp")
b$`seconds (log scale)` <- b$time 

b$method <- b$method %>% str_replace_all("additive", "vertex addition (lr)")

formatBack <- function(x) 10^x 

ggplot(b, aes(x=`seconds (log scale)`, fill=method)) + stat_density(adjust=5,alpha = 0.5,trim=TRUE) + scale_fill_manual(values=c("#999999", "maroon")) + theme_minimal() + 
theme(axis.text.y = element_text(color="black", size=15), 
      axis.title=element_text(size=14,face="bold"),
      legend.title = element_text(size=14, face="bold"),
      legend.text = element_text(size=14, face="bold"),
      axis.text.x = element_text(color="black", size=15)) + scale_x_continuous(labels=function(x) 10^x ) +
ggsave("emnlp/times.pdf", width=7, height=1.5)
