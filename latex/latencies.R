library(tidyverse)
a <- read_csv("all_times.csv")
a$time <- log(a$time)
b <- a%>% filter(method=="additive" |  method == "ilp")
b$`log seconds` <- b$time 

b$method <- b$method %>% str_replace_all("additive", "vertex addition (lr)")

ggplot(b, aes(x=`log seconds`, fill=method)) + geom_density(alpha = 0.2) + scale_fill_manual(values=c("#999999", "maroon")) + theme_minimal() + 
theme(axis.text.y = element_text(color="black", size=15), 
      axis.title=element_text(size=14,face="bold"),
      legend.title = element_text(size=14, face="bold"),
      legend.text = element_text(size=14, face="bold"),
      axis.text.x = element_text(color="black", size=15))
ggsave("times.pdf", width=7, height=1.5)
