library(tidyverse)
library(textrecipes)
library(tidymodels)
library(tidytext)
library(stylo)


# читаем данные — загружаются таблицы признаков (MFW, char, POS, lemma)
# фиксируем номера экспериментов возле ссылки на матрицы признаков
# CF_LR_MFW100_31, CF_LR_MFW100_32, CF_SVM_MFW100_33, CF_RF_MFW100_34
url <- "https://raw.githubusercontent.com/AButon-8/iskra-project/refs/heads/main/features/mfw_100_norm.csv"

# CF_LR_MFW100_35, CF_LR_MFW100_36, CF_SVM_MFW100_37, CF_RF_MFW100_38
url_ch <- "https://raw.githubusercontent.com/AButon-8/iskra-project/refs/heads/main/features/mfw_B1_100.csv"

# CF_LR_MFW300_39, CF_LR_MFW300_40, CF_SVM_MFW300_41, CF_RF_MFW300_42
url_ch <- "https://raw.githubusercontent.com/AButon-8/iskra-project/refs/heads/main/features/mfw_B1_300.csv"

# CF_LR_MFW500_43, CF_LR_MFW500_44, CF_SVM_MFW500_45, CF_RF_MFW500_46
url_ch <- "https://raw.githubusercontent.com/AButon-8/iskra-project/refs/heads/main/features/mfw_B1_500.csv"

# CF_LR_MFW300_47, CF_LR_MFW300_48, CF_SVM_MFW300_49, CF_RF_MFW300_50, CF_SVM_MFW300_49_77
url_ch <- "https://raw.githubusercontent.com/AButon-8/iskra-project/refs/heads/main/features/mfw_C3_300.csv"

# CF_LR_MFW500_51, CF_LR_MFW500_52, CF_SVM_MFW500_53, CF_RF_MFW500_54, CF_SVM_MFW500_53_78
url_ch <- "https://raw.githubusercontent.com/AButon-8/iskra-project/refs/heads/main/features/mfw_C2_500.csv"

# CF_LR_MFW300_55,	CF_LR_MFW300_56,	CF_SVM_MFW300_57,	CF_RF_MFW300_58, CF_SVM_MFW300_57_79
url_ch <- "https://raw.githubusercontent.com/AButon-8/iskra-project/refs/heads/main/features/mfw_C2_300.csv"

# CF_LR_MFW500_59,	CF_LR_MFW500_60,	CF_SVM_MFW500_61,	CF_RF_MFW500_62, CF_SVM_MFW500_61_80
url_ch <- "https://raw.githubusercontent.com/AButon-8/iskra-project/refs/heads/main/features/mfw_C3_500.csv"

# CF_LR_MFW300_63,	CF_LR_MFW300_64,	CF_SVM_MFW300_65,	CF_RF_MFW300_66, CF_SVM_MFW300_65_81
url_ch <- "https://raw.githubusercontent.com/AButon-8/iskra-project/refs/heads/main/features/mfw_C1_300.csv"

# CF_LR_MFW500_67,	CF_LR_MFW500_68,	CF_SVM_MFW500_69,	CF_RF_MFW500_70, CF_SVM_MFW500_69_82
url_ch <- "https://raw.githubusercontent.com/AButon-8/iskra-project/refs/heads/main/features/mfw_C1_500.csv"




# запустить для чтения .csv
mfw <- read_csv(url) |> 
  mutate(author = str_extract(file_name, "^[^_]+"), .after = file_name) |> # создаётся колонка author, из имени файла извлекается автор (до первого _)
  select(-author_folder)


# ! ДЛЯ CHUNKS запускаем матрицу для чтения
mfw_ch <- read_csv(url_ch) |> 
  relocate(author, .after = chunk_id)



# =====
# БЛОК 1. Logistic Regression / Логистическая регрессия
# =====

# 1. Загрузка данных
# 2. Разделение train/test
# 3. Обучение модели
# 4. Предсказание
# 5. Оценка (accuracy, confusion matrix)

# НЕ НУЖНО ДЛЯ CHUNKS! Убираем тексты dubia
mfw <- mfw |> 
  filter(author != "dubia")

# НЕ НУЖНО ДЛЯ CHUNKS! Убираем авторов с малым количестом текстов для варианта с регуляризацией
mfw_clean <- mfw |> 
  group_by(author) |> 
  filter(n() >= 5) |> 
  ungroup()

# ДЛЯ CHUNKS делаем только это:
mfw <- mfw_ch

# Проверить распределение количества текстов по авторам
table(mfw$author)
# table(mfw_clean$author) # НЕ НУЖНО ДЛЯ CHUNKS!
nrow(mfw)        # всего текстов
# nrow(mfw_clean) # НЕ НУЖНО ДЛЯ CHUNKS!


# == ВАРИАНТ 1. Baseline - логистическая регрессия (nnet::multinom)

# Разделяем X & y
X <- mfw |>
  # select(-chunk_id, -author) # только ДЛЯ CHUNKS
  select(-file_name, -author) # НЕ НУЖНО ДЛЯ CHUNKS!


y <- mfw$author

# Проверяем тип target
y <- as.factor(y)


# Собираем обратно
df_model <- data.frame(X, author = y)

# Проверяем количество авторов и текстов(чанков)
length(unique(y))

table(mfw$author)
table(df_model$author)


# Train / Test split
library(rsample)

set.seed(818)

split <- initial_split(df_model, prop = 0.8, strata = author)

train_data <- training(split)
test_data  <- testing(split)


# Проверяем количество авторов / текстов авторов
length(unique(y))
table(train_data$author)
table(test_data$author)


# Масштабируем, используя параметры train
train_X <- scale(train_data[, -ncol(train_data)])
test_X <- scale(test_data[, -ncol(test_data)],
                center = attr(train_X, "scaled:center"),
                scale = attr(train_X, "scaled:scale"))

# Собираем обратно
train_final <- data.frame(train_X, author = train_data$author)
test_final <- data.frame(test_X, author = test_data$author)



# multinomial logistic regression
library(nnet)

model_logit <- multinom(author ~ ., data = train_data)


# Предсказание
pred <- predict(model_logit, newdata = test_data)


# Оценка: confusion matrix (матрица ошибок)
table(Predicted = pred, Actual = test_data$author)

# Accuracy
mean(pred == test_data$author)


# выгрузить confusion matrix
cm <- table(Predicted = pred, Actual = test_data$author) # сохраняем матрицу
cat(capture.output(write.table(cm, sep = "\t", col.names = NA)), sep = "\n")


# Считаем F1
library(yardstick)
df_lr <- data.frame(
  truth = test_data$author,
  pred = pred
)

f_meas(df_lr, truth, pred, estimator = "macro")



# === ВАРИАНТ. 2 Regularized - логистическая регрессия с регуляризацией (glmnet)

library(tidyverse)
library(rsample)
library(glmnet)


# НЕ НУЖНО ДЛЯ CHUNKS! Убираем тексты dubia
mfw <- mfw |> 
  filter(author != "dubia")

# НЕ НУЖНО ДЛЯ CHUNKS! Убираем авторов с малым количестом текстов для варианта с регуляризацией
mfw_clean <- mfw |> 
  group_by(author) |> 
  filter(n() >= 5) |> 
  ungroup()

# ДЛЯ CHUNKS! запускаем это
mfw_clean <- mfw_ch


# Проверяем количество текстов на автора
table(mfw_clean$author)


# Подготовка X и y
X <- mfw_clean |> 
  select(-file_name, -author)

# для CHUNKS
X <- mfw_clean |> 
  select(-chunk_id, -author)

y <- as.factor(mfw_clean$author)



df_model <- data.frame(X, author = y)



# Train / Test split

set.seed(818)

split <- initial_split(df_model, prop = 0.8, strata = author)

train_data <- training(split)
test_data  <- testing(split)


# Проверяем количество авторов / текстов авторов
length(unique(y))
table(train_data$author)
table(test_data$author)


# матрицы для glmnet
x_train <- as.matrix(train_data |> select(-author))
y_train <- train_data$author

x_test  <- as.matrix(test_data |> select(-author))
y_test  <- test_data$author



# Обучение модели (LASSO)
set.seed(818)

# матрицы - glmnet по умеолчанию делает standardize = TRUE, alpha = 1 → LASSO
cv_model <- cv.glmnet(
  x_train,
  y_train,
  family = "multinomial",
  alpha = 1
)


# Предсказание
pred <- predict(
  cv_model,
  newx = x_test,
  s = "lambda.min",
  type = "class"
)


# Оценка: confusion matrix (матрица ошибок)
table(Predicted = pred, Actual = y_test)

# Accuracy
mean(pred == y_test)


# выгрузить confusion matrix
levels_all <- levels(y)  # все авторы из исходных данных

# сохраняем матрицу
cm <- table(
  Predicted = factor(pred, levels = levels_all),
  Actual    = factor(y_test, levels = levels_all)
)

cat(capture.output(write.table(cm, sep = "\t", col.names = NA)), sep = "\n")


# проверить какие авторы есть 
# в столбцах(y_test=что реально было) и в строках(pred=что модель предсказала)
table(y_test)
table(pred)


# Считаем F1
library(yardstick)
df_lasso <- data.frame(
  truth = y_test,
  pred = as.vector(pred)
)

f_meas(df_lasso, truth, pred, estimator = "macro")




# =====
# БЛОК 2. SVM (Support Vector Machine) / Опорные векторы
# =====

# 1. Загрузка данных
# 2. Разделение train/test
# 3. Обучение модели
# 4. Предсказание
# 5. Оценка (accuracy, confusion matrix)


library(e1071)

# Убираем тексты dubia
mfw_c <- mfw |> 
  filter(author != "dubia")

# Убираем авторов с малым количестом текстов для варианта с регуляризацией
mfw_clean <- mfw_c |> 
  group_by(author) |> 
  filter(n() >= 5) |> 
  ungroup()


# ДЛЯ CHUNKS! запускаем это
mfw_clean <- mfw_ch


# Проверяем количество текстов на автора
table(mfw_clean$author)


# Подготовка X и y
X <- mfw_clean |> 
  select(-file_name, -author)

# CHUNKS / Подготовка X и y 
X <- mfw_clean |> 
  select(-chunk_id, -author)

y <- as.factor(mfw_clean$author)



df_model <- data.frame(X, author = y)


# Train / Test split

set.seed(818)

split <- initial_split(df_model, prop = 0.8, strata = author)

train_data <- training(split)
test_data  <- testing(split)


# отделяем признаки
X_train <- train_data |> select(-author)
X_test  <- test_data  |> select(-author)

# scaling ТОЛЬКО по train
X_train_scaled <- scale(X_train)

# применяем те же параметры к test
X_test_scaled <- scale(
  X_test,
  center = attr(X_train_scaled, "scaled:center"),
  scale  = attr(X_train_scaled, "scaled:scale")
)

# собираем обратно
train_data <- data.frame(X_train_scaled, author = train_data$author)
test_data  <- data.frame(X_test_scaled,  author = test_data$author)




# author — factor
train_data$author <- as.factor(train_data$author)
test_data$author  <- as.factor(test_data$author)



# Обучение SVM. RBF kernel
set.seed(818)

svm_model <- svm(
  author ~ .,
  data = train_data,
  kernel = "radial", # kernel = "radial" 
  cost = 1,  # базовая регуляризация
  gamma = 1 / ncol(train_data) # зависит от числа признаков (стабилизирует модель)
)


#  ============================

# !!!! NEW - SVM. Linear kernel
# ========================
set.seed(818)

svm_model <- svm(
  author ~ .,
  data = train_data,
  kernel = "linear",  # линейное ядро
  cost = 1,           # базовая регуляризация
  probability = TRUE  # для получения вероятностей
)



# Предсказание
pred <- predict(svm_model, newdata = test_data)


# Оценка. Confusion matrix
table(Predicted = pred, Actual = test_data$author)

# Accuracy
mean(pred == test_data$author)


# Выгрузка
cm <- table(Predicted = pred, Actual = test_data$author)
cat(capture.output(write.table(cm, sep = "\t", col.names = NA)), sep = "\n")


# NEW посчитать F1
library(yardstick)

df_svm <- data.frame(
  truth = test_data$author,
  pred = pred
)

f_meas(df_svm, truth, pred, estimator = "macro")





# =====
# БЛОК 3. Random Forest / Случайный лес
# =====

# 1. Загрузка данных
# 2. Разделение train/test
# 3. Обучение модели
# 4. Предсказание
# 5. Оценка (accuracy, confusion matrix)



library(randomForest)

# Убираем тексты dubia
mfw <- mfw |> 
  filter(author != "dubia")

# Убираем авторов с малым количестом текстов для варианта с регуляризацией
mfw_clean <- mfw |> 
  group_by(author) |> 
  filter(n() >= 5) |> 
  ungroup()

# CHUNKS
mfw_clean <- mfw_ch


# Проверяем количество текстов на автора
table(mfw_clean$author)


# Подготовка X и Y
X <- mfw_clean |> 
  select(-file_name, -author)


# CHUNKS Подготовка X и Y
X <- mfw_clean |> 
  select(-chunk_id, -author)


y <- as.factor(mfw_clean$author)



# Scaling - НЕ ЗДЕСЬ!
# X_scaled <- scale(X)

df_model <- data.frame(X, author = y)


# Train / Test split

set.seed(818)

split <- initial_split(df_model, prop = 0.8, strata = author)

train_data <- training(split)
test_data  <- testing(split)


# author — factor
train_data$author <- as.factor(train_data$author)
test_data$author  <- as.factor(test_data$author)


# Обучение модели
set.seed(818)


# author - целевая переменная, ntree = 500 - кол-во деревьев, 
# mtry = sqrt(ncol(train_data) - 1) - сколько признаков рассматривается при каждом разбиении дерева
# importance = TRUE - включена оценка важности признаков
rf_model <- randomForest(
  author ~ .,
  data = train_data,
  ntree = 500,      # можно 300–1000, 500 — стандарт
  #mtry = sqrt(ncol(train_data) - 1),
  importance = TRUE
)




# Предсказание
pred <- predict(rf_model, newdata = test_data)


# Confusion matrix
table(Predicted = pred, Actual = test_data$author)


# Accuracy
mean(pred == test_data$author)


# Выгрузка
cm <- table(Predicted = pred, Actual = test_data$author)
cat(capture.output(write.table(cm, sep = "\t", col.names = NA)), sep = "\n")


# Считаем F1
library(yardstick)

df_eval <- data.frame(
  truth = test_data$author,
  pred = pred
)

accuracy(df_eval, truth, pred)
f_meas(df_eval, truth, pred, estimator = "macro")




# ДОПОЛНИТЕЛЬНО. Feature importance
importance(rf_model)
varImpPlot(rf_model)




# =====
# БЛОК 4. DUBIA-test
# =====

library(tidyverse)
library(rsample)
library(e1071)
library(randomForest)
library(glmnet)
library(nnet)
library(yardstick)


# Убираем дубиа
mfw_c <- mfw |> 
  filter(author != "dubia")


# Убираем "коротких" авторов
mfw_clean <- mfw_c |> 
  group_by(author) |> 
  filter(n() >= 5) |> 
  ungroup()

# CHUNKS
mfw_clean <- mfw_ch


# X y
X <- mfw_clean |> select(-file_name, -author)

# CHUNKS
X <- mfw_clean |> select(-chunk_id, -author)
y <- as.factor(mfw_clean$author)


# TRAIN / TEST SPLIT
set.seed(818)

df <- data.frame(X, author = y)

split <- initial_split(df, prop = 0.8, strata = author)

train_data <- training(split)
test_data  <- testing(split)


# SCALING
train_X <- train_data |> select(-author)
test_X  <- test_data  |> select(-author)

train_X_scaled <- scale(train_X)

scale_center <- attr(train_X_scaled, "scaled:center")
scale_scale  <- attr(train_X_scaled, "scaled:scale")

test_X_scaled <- scale(test_X,
                       center = scale_center,
                       scale = scale_scale)

train_data2 <- data.frame(train_X_scaled, author = train_data$author)
test_data2  <- data.frame(test_X_scaled, author = test_data$author)


# MODEL 1 — Logistic Regression
# слишком много признаков (MWF300 = 2416)
set.seed(818)

logit_model <- nnet::multinom(
  author ~ .,
  data = train_data2,
  MaxNWts = 10000
)

# НЕ ЗАПУСКАЮ - для модели с небольшим количеством признаков!
logit_model <- nnet::multinom(author ~ ., data = train_data2)


pred_lr <- predict(logit_model, test_data2)

mean(pred_lr == test_data2$author)

df_lr <- data.frame(
  truth = test_data2$author,
  pred = pred_lr
)


# MODEL 2 — LASSO
x_train <- as.matrix(train_data2 |> select(-author))
y_train <- train_data2$author

x_test <- as.matrix(test_data2 |> select(-author))
y_test <- test_data2$author



# Проверяем количество авторов / текстов авторов
length(unique(y))
table(train_data2$author)
table(test_data2$author)



set.seed(818)

lasso_model <- cv.glmnet(
  x_train, y_train,
  family = "multinomial",
  alpha = 1
)


# ПРИ НЕОБХОДИМОСТИ уменьшаем число фолдов для малых авторов, по умолчанию nfolds = 10
lasso_model <- cv.glmnet(
  x_train, y_train,
  family = "multinomial",
  alpha = 1,
  nfolds = 5
)


pred_lasso <- predict(lasso_model, x_test, s = "lambda.min", type = "class")


mean(pred_lasso == y_test)


# извлекаем из matrix - factor
pred_lasso <- factor(pred_lasso[, 1], levels = levels(y_test))



df_lasso <- data.frame(
  truth = y_test,
  pred = pred_lasso
)




# MODEL 3 — SVM
set.seed(818)

svm_model <- svm(
  author ~ .,
  data = train_data2,
  kernel = "radial"
)

pred_svm <- predict(svm_model, test_data2)

mean(pred_svm == test_data2$author)

df_svm <- data.frame(
  truth = test_data2$author,
  pred = pred_svm
)


f_meas(df_svm, truth, pred, estimator = "macro")
 
# СОХРАНЯЕМ МОДЕЛЬ
saveRDS(svm_model, "/Users/anastasiabogdanova/R_directory/iskra-project/models/svm_model_C2_300_F1_0.767.rds")
saveRDS(svm_model, "/Users/anastasiabogdanova/R_directory/iskra-project/models/svm_model_C1_300_F1_0.793.rds")
saveRDS(svm_model, "/Users/anastasiabogdanova/R_directory/iskra-project/models/svm_model_C3_300_F1_0.812.rds")





# MODEL 4 — RANDOM FOREST
set.seed(818)

rf_model <- randomForest(
  author ~ .,
  data = train_data,
  ntree = 500
)

pred_rf <- predict(rf_model, test_data)

mean(pred_rf == test_data$author)


df_rf <- data.frame(
  truth = test_data$author,
  pred = pred_rf
)


# список признаков из train
feature_names <- colnames(train_data2 |> select(-author))


# CONFUSION MATRICES
cm_lr <- table(pred_lr, test_data2$author)
cm_lasso <- table(pred_lasso, y_test)
cm_svm <- table(pred_svm, test_data2$author)
cm_rf <- table(pred_rf, test_data2$author)



# Выгружаем

cat(capture.output(write.table(cm_lr, sep = "\t", col.names = NA)), sep = "\n")
cat(capture.output(write.table(cm_lasso, sep = "\t", col.names = NA)), sep = "\n")
cat(capture.output(write.table(cm_svm, sep = "\t", col.names = NA)), sep = "\n")
cat(capture.output(write.table(cm_rf, sep = "\t", col.names = NA)), sep = "\n")


# accuracy, F1
results <- data.frame(
  model = c("LR", "LASSO", "SVM", "RF"),
  accuracy = c(
    mean(pred_lr == test_data2$author),
    mean(pred_lasso == y_test),
    mean(pred_svm == test_data2$author),
    mean(pred_rf == test_data2$author)
  ),
  f1_macro = c(
    f_meas(df_lr, truth, pred, estimator = "macro")$.estimate,
    f_meas(df_lasso, truth, pred, estimator = "macro")$.estimate,
    f_meas(df_svm, truth, pred, estimator = "macro")$.estimate,
    f_meas(df_rf, truth, pred, estimator = "macro")$.estimate
  )
)

results

