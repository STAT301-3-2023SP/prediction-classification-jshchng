library(tidyverse)

load("attempt_1/results/final_pred.rda")
train <- read_csv("data/train.csv")

write.csv(final_pred_bt, file = "attempt_1/attempt1.csv"
          , col.names = FALSE, row.names = FALSE)

write.csv(final_pred_rf, file = "attempt_1/attempt2.csv"
          , col.names = FALSE, row.names = FALSE)

write.csv(final_pred_mars, file = "attempt_1/attempt3.csv"
          , col.names = FALSE, row.names = FALSE)

write.csv(final_pred_svmr, file = "attempt_1/attempt4.csv"
          , col.names = FALSE, row.names = FALSE)

write.csv(final_pred_svmp, file = "attempt_1/attempt5.csv"
          , col.names = FALSE, row.names = FALSE)


write.csv(final_pred_mlp, file = "attempt_1/attempt6.csv"
          , col.names = FALSE, row.names = FALSE)
### RF gave best score, BT next

## NOTE:svmr and svmp did better on training set, did worst on test, reverse for rf and bt