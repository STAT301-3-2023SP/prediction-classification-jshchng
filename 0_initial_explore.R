library(tidyverse)
library(tidymodels)

tidymodels_prefer()

train <- read_csv("data/train.csv")
test <- read_csv("data/test.csv")

# best practice, split train to prevent overfitting
set.seed(301)
split <- initial_split(train, prop = 0.75, strata = y)

train_data <- training(split)
test_data <- testing(split)

#########################################################
# functions for exploration
boxplot_fun <- function(var = NULL){
  ggplot(train_data, aes(x = factor(!!sym(var)), y = y)) +
    geom_boxplot()
}



boxplot_log_fun <- function(var = NULL){
  ggplot(train_data, aes(x = factor(!!sym(var)), y = log(y))) +
    geom_boxplot()
}


#########################################################
# Distribution of y

ggplot(train_data, aes(x = y)) +
  geom_histogram()
# skewed right

# figure best transformation, boxcox recommendation. MLE
MASS::boxcox(lm(y ~1, train_data))

ggplot(train_data, aes(x = log(y))) +
  geom_histogram()

###################################################
# missingness

missing_lst <- list()

for(var in colnames(train_data)) {
  missing_lst[var] <- train_data %>%
    select(any_of(var)) %>%
    filter(is.na(!!sym(var))) %>%
    summarize(num_missing = n())
}

# turn list into table
missing_tbl <- enframe(unlist(missing_lst))

missing_tbl %>%
  mutate(pct = value/4034) %>%
  arrange(desc(pct))

# doesnt meet 20% criteria, likely best to leave in dataset


###################################################
# remove zero_var
# step_zv

var_lst <- list()

for(var in colnames(train_data)) {
  var_lst[var] <- train_data %>%
    select(any_of(var)) %>%
    summarize(sd = sd(!!sym(var), na.rm = TRUE))
}

var_tbl <- enframe(unlist(var_lst))
# remove zero var variables, high var can also benefit from transformation

zero_var <- var_tbl %>%
  filter(value == 0) %>%
  pull(name)

# update training data to remove unwanted variables
train_data <- train_data %>%
  select(!all_of(zero_var))

################################################################################
# high correlation
# step_corr could do the same thing

###############################################################
# miscoded categorical variables
cat_lst <- list()

for(var in colnames(train_data)) {
  cat_lst[var] <- train_data %>%
    select(any_of(var)) %>%
    summarize(unique = length(unique(!!sym(var))))
}

cat_tbl <- enframe(unlist(cat_lst))
# considering under 10 to be categorical

cat_var <- cat_tbl %>%
  filter(value <= 10) %>%
  pull(name)

# test boxplot function
boxplot_fun(var = "x084")

boxplot_log_fun(var = "x025")

map(cat_var, boxplot_fun)

map(cat_var, boxplot_log_fun)

# choose if any look like they have relationship
# turn to factor w mutate
# do so in both train and test

# could write functions for histograms to explore variable
# could write function for scatterplots to explore relation with Y

# save out clean datasets
# consider variable reduction technique with lasso or random forest

train_data <- train_data %>%
  mutate(y = as.factor(y))

write_rds(train_data, file = "data/train_data.rds")
