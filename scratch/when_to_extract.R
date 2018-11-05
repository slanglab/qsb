R version 3.3.3 (2017-03-06) -- "Another Canoe"
Copyright (C) 2017 The R Foundation for Statistical Computing
Platform: x86_64-apple-darwin13.4.0 (64-bit)

R is free software and comes with ABSOLUTELY NO WARRANTY.
You are welcome to redistribute it under certain conditions.
Type 'license()' or 'licence()' for distribution details.

  Natural language support but running in an English locale

R is a collaborative project with many contributors.
Type 'contributors()' for more information and
'citation()' on how to cite R or R packages in publications.

Type 'demo()' for some demos, 'help()' for on-line help, or
'help.start()' for an HTML browser interface to help.
Type 'q()' to quit R.

> library(tidyvser)
Error in library(tidyvser) : there is no package called ‘tidyvser’
> library(tidyverse)
── Attaching packages ─────────────────────────────────────── tidyverse 1.2.1 ──
✔ ggplot2 3.0.0     ✔ purrr   0.2.5
✔ tibble  1.4.2     ✔ dplyr   0.7.4
✔ tidyr   0.8.1     ✔ stringr 1.3.1
✔ readr   1.1.1     ✔ forcats 0.3.0
── Conflicts ────────────────────────────────────────── tidyverse_conflicts() ──
✖ dplyr::filter() masks stats::filter()
✖ dplyr::lag()    masks stats::lag()
> a <- read_csv("/tmp/scratch.csv")
Parsed with column specification:
cols(
  r = col_integer(),
  po = col_character(),
  mcl = col_integer()
)
> a
# A tibble: 105,363 x 3
       r po      mcl
   <int> <chr> <int>
 1    85 True     42
 2    65 False    42
 3    43 True     23
 4    68 True     43
 5    80 True     21
 6    56 True     23
 7   106 True     43
 8    61 True     28
 9    29 True     10
10    27 True     12
# ... with 105,353 more rows
> a["ec"] <- a$mcl > a$r
> a
# A tibble: 105,363 x 4
       r po      mcl ec
   <int> <chr> <int> <lgl>
 1    85 True     42 FALSE
 2    65 False    42 FALSE
 3    43 True     23 FALSE
 4    68 True     43 FALSE
 5    80 True     21 FALSE
 6    56 True     23 FALSE
 7   106 True     43 FALSE
 8    61 True     28 FALSE
 9    29 True     10 FALSE
10    27 True     12 FALSE
# ... with 105,353 more rows
> a %>% filter(ec == True)
Error in filter_impl(.data, quo) :
  Evaluation error: object 'True' not found.
> a %>% filter(ec == true)
Error in filter_impl(.data, quo) :
  Evaluation error: object 'true' not found.
> a %>% filter(ec == TRUE)
# A tibble: 2,741 x 4
       r po      mcl ec
   <int> <chr> <int> <lgl>
 1    65 False    66 TRUE
 2    22 False    43 TRUE
 3    30 False    37 TRUE
 4    26 False    36 TRUE
 5    28 False    34 TRUE
 6    27 False    30 TRUE
 7    19 False    24 TRUE
 8    42 False    54 TRUE
 9    25 False    26 TRUE
10    30 False    31 TRUE
# ... with 2,731 more rows
> a %>% filter(ec == TRUE) %>% group(po)
Error in function_list[[k]](value) : could not find function "group"
> a %>% filter(ec == TRUE)
> a %>% filter(ec == TRUE) %>% group_by(po)
# A tibble: 2,741 x 4
# Groups:   po [1]
       r po      mcl ec
   <int> <chr> <int> <lgl>
 1    65 False    66 TRUE
 2    22 False    43 TRUE
 3    30 False    37 TRUE
 4    26 False    36 TRUE
 5    28 False    34 TRUE
 6    27 False    30 TRUE
 7    19 False    24 TRUE
 8    42 False    54 TRUE
 9    25 False    26 TRUE
10    30 False    31 TRUE
# ... with 2,731 more rows
> a %>% filter(ec == TRUE) %>% group_by(po) %>% count()
# A tibble: 1 x 2
# Groups:   po [1]
  po        n
  <chr> <int>
1 False  2741
> quit()