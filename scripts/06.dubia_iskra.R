# ==============

# DUBIA на C1,C2,C3 MFW 300/500

# =============

library(tidyverse)
library(tidytext)
library(e1071)
library(caret)
library(yardstick)


# ==============

# МАСТЕР-КОРПУС
# FEATURES C1,C2,C3 MFW500

# =============


# Загружаем
dubia <- read_csv("/iskra-project/data/processed/corpus_dubia.csv")

corpus_C1 <- read_csv("/iskra-project/data/processed/corpus_C1.csv")
corpus_C2 <- read_csv("/iskra-project/data/processed/corpus_C2.csv")
corpus_C3 <- read_csv("/iskra-project/data/processed/corpus_C3.csv")



# Добавляем идентификатор реплики (полезно для отслеживания)
corpus_C1 <- corpus_C1 |>  mutate(replica = "C1")
corpus_C2 <- corpus_C2 |>  mutate(replica = "C2")
corpus_C3 <- corpus_C3 |>  mutate(replica = "C3")

# Объединяем в мастер-корпус C
master_C <- bind_rows(corpus_C1, corpus_C2, corpus_C3)

# Проверка структуры
glimpse(master_C)


# Детальная проверка баланса в мастер-корпусе C
master_C |> 
  group_by(author) |> 
  summarise(
    n_chunks = n(),
    total_words = sum(n_words, na.rm = TRUE),
    avg_words = mean(n_words, na.rm = TRUE),
    n_replicas = n_distinct(replica)  # у скольких авторов есть все 3 реплики?
  ) |> 
  arrange(desc(n_chunks)) |> 
  print(n = 10)  # покажем всех авторов


# Функция для удаления чисел из текста (базовый R, без пайпа)
remove_numbers_from_text <- function(text) {
  # Шаг 1: удаляем отдельно стоящие числа
  text <- gsub("\\b[0-9]+\\b", "", text)
  # Шаг 2: удаляем все цифры (на всякий случай)
  text <- gsub("[0-9]+", "", text)
  # Шаг 3: заменяем несколько пробелов на один
  text <- gsub("\\s+", " ", text)
  # Шаг 4: удаляем пробелы в начале и конце
  text <- trimws(text)
  return(text)
}


# Применяем к текстам в мастер-корпусе
master_C_clean <- master_C |> 
  mutate(text_raw_clean = remove_numbers_from_text(text_raw))


# Проверяем результат
cat("Исходный текст (первые 200 символов):\n")
cat(substr(master_C$text_raw[1], 1, 200), "\n\n")

cat("Очищенный текст (первые 200 символов):\n")
cat(substr(master_C_clean$text_raw_clean[1], 1, 200), "\n")


# Токенизация (уже без чисел)
tokens_C <- master_C_clean |> 
  unnest_tokens(word, text_raw_clean)


# Проверяем, что чисел нет
cat("Числовые токены (если есть):\n")
tokens_C |> 
  filter(grepl("^[0-9]+$", word)) |> 
  count(word, sort = TRUE) |> 
  head(10)

# Если есть — удаляем
tokens_C <- tokens_C |> 
  filter(!grepl("^[0-9]+$", word))



# Общая статистика по токенам
# 2026/04/30 = total_tokens = 811876, unique_words = 57476
# позже в этот же день = total_tokens = 806642, unique_words = 57087
tokens_C |> 
  summarise(
    total_tokens = n(),
    unique_words = n_distinct(word)
  )




# Топ-500 самых частотных слов (глобально)
mfw_500_global <- tokens_C |> 
  count(word, sort = TRUE) |> 
  slice_head(n = 500) |> 
  pull(word)

# Топ-300 самых частотных слов
mfw_300_global <- tokens_C |> 
  count(word, sort = TRUE) |> 
  slice_head(n = 300) |> 
  pull(word)

# Посмотрим на первые 30 слов
head(mfw_300_global, 30)


# Топ-30 слов для каждого автора отдельно
author_top_words <- tokens_C  |> 
  group_by(author, word) |> 
  summarise(count = n(), .groups = "drop")  |> 
  group_by(author) |> 
  mutate(
    author_total = sum(count),
    rel_freq = count / author_total,
    rank = row_number(desc(rel_freq))
  ) |> 
  filter(rank <= 50)  |>   # топ-50 для быстрой проверки
  select(author, word, rank)


# Посмотрим, насколько различаются топ-слова у разных авторов
author_top_words |> 
  filter(rank <= 10)  |> 
  arrange(author, rank)  |> 
  print(n = 100)



top10 <- author_top_words |> 
  filter(rank <= 10)  |> 
  arrange(author, rank)  |> 
  print(n = 100)

cat(capture.output(write.table(top10, sep = "\t", col.names = NA)), sep = "\n") 

# Проверка: есть ли числа в MFW?
cat("Чисел в MFW300:", sum(grepl("^[0-9]+$|^X[0-9]+$", mfw_300_global)), "\n")
cat("Чисел в MFW500:", sum(grepl("^[0-9]+$|^X[0-9]+$", mfw_500_global)), "\n")


# Пути
features_dir <- "/Users/anastasiabogdanova/R_directory/iskra-project/features/"
data_dir <- "/Users/anastasiabogdanova/R_directory/iskra-project/data/processed/"



# Сохраняем словари
write_lines(mfw_300_global, paste0(features_dir, "mfw_300_C_master.txt"))
write_lines(mfw_500_global, paste0(features_dir, "mfw_500_C_master.txt"))

# Дополнительно: сохраняем как R-объект (для быстрой загрузки)
saveRDS(mfw_300_global, paste0(features_dir, "mfw_300_C_master.rds"))
saveRDS(mfw_500_global, paste0(features_dir, "mfw_500_C_master.rds"))

# И сохраняем мастер-корпус C (на всякий случай)
saveRDS(master_C, paste0(data_dir, "master_C_corpus.rds"))



# MFW фиксируем по мастер-корпусу C, затем используем для C1, C2, C3, dubia
# Загружаем фиксированный словарь (подсчитан на мастер-корпусе C)

# Путь
features_dir <- "/Users/anastasiabogdanova/R_directory/iskra-project/features/"
data_dir <- "/Users/anastasiabogdanova/R_directory/iskra-project/data/processed/"

# Загружаем словари (уже подсчитаны на мастер-корпусе C)
mfw_300 <- read_lines(paste0(features_dir, "mfw_300_C_master.txt"))
mfw_500 <- read_lines(paste0(features_dir, "mfw_500_C_master.txt"))

# Проверка
length(mfw_300)  # должно быть 300
length(mfw_500)  # должно быть 500

# Первые 20 слов
head(mfw_300, 20)


# =========
# Матрица для C1 (MFW300)


# =======

# Загружаем корпус C1
corpus_C1 <- read_csv(paste0(data_dir, "corpus_C1.csv"))

# Проверка 
glimpse(corpus_C1)


# Токенизация
tokens_C1 <- corpus_C1 |>
  unnest_tokens(word, text_raw)

# Фильтруем ТОЛЬКО слова из фиксированного словаря mfw_300
tokens_filtered <- tokens_C1 |>
  filter(word %in% mfw_300)

# Считаем частоту каждого слова в каждом чанке
mfw_matrix <- tokens_filtered |>
  count(chunk_id, word) |>
  pivot_wider(id_cols = chunk_id, names_from = word, values_from = n, values_fill = 0)

# Добавляем информацию об авторе и длине текста
author_info <- corpus_C1 |>
  select(chunk_id, author, n_words) |>
  distinct()

result <- mfw_matrix |>
  left_join(author_info, by = "chunk_id") |>
  select(chunk_id, author, n_words, everything())

# Нормализация на 1000 слов
word_cols <- setdiff(names(result), c("chunk_id", "author", "n_words"))

for (col in word_cols) {
  result[[col]] <- result[[col]] / result[["n_words"]] * 1000
}

# Удаляем столбец n_words 
result_final <- result |> select(-n_words)

# Проверка (долджно быть число чанков (279) и колонок (300 + 2)
dim(result_final)  # сколько строк и колонок?
# Должно быть: число чанков в C1 строк и (2 + 300) колонок

# Проверка: нет ли артефактной колонки X1?
any(grepl("^X[0-9]+$", colnames(result_final)))  # должно быть FALSE

# Сохраняем
write_csv(result_final, paste0(features_dir, "mfw_C.C1_300.csv"))

# Проверка
head(result_final[, 1:8])




# =========
# Матрица для C1 (MFW500)


# =======

# Загружаем корпус C1
corpus_C1 <- read_csv(paste0(data_dir, "corpus_C1.csv"))

# Токенизация
tokens_C1 <- corpus_C1 |>
  unnest_tokens(word, text_raw)

# ТОЛЬКО слова из фиксированного словаря mfw_500
tokens_filtered <- tokens_C1 |>
  filter(word %in% mfw_500)

# Считаем частоту каждого слова в каждом чанке
mfw_matrix <- tokens_filtered |>
  count(chunk_id, word) |>
  pivot_wider(id_cols = chunk_id, names_from = word, values_from = n, values_fill = 0)

# Добавляем автора и длину
author_info <- corpus_C1 |>
  select(chunk_id, author, n_words) |>
  distinct()

result <- mfw_matrix |>
  left_join(author_info, by = "chunk_id") |>
  select(chunk_id, author, n_words, everything())

# Нормализация на 1000 слов
word_cols <- setdiff(names(result), c("chunk_id", "author", "n_words"))

for (col in word_cols) {
  result[[col]] <- result[[col]] / result[["n_words"]] * 1000
}

# Удаляем n_words
result_final <- result |> select(-n_words)

# Проверка
dim(result_final)  # ожидаем: 279 строк x 502 колонки
any(grepl("^X[0-9]+$", colnames(result_final)))  # должно быть FALSE

# Сохраняем
write_csv(result_final, paste0(features_dir, "mfw_C.C1_500.csv"))


head(result_final[, 1:8])



# =========
# Матрица для C2 (MFW300 и MFW500)


# =======


# Загружаем C2
corpus_C2 <- read_csv(paste0(data_dir, "corpus_C2.csv"))


# --- MFW300 ---
tokens_C2 <- corpus_C2 |> 
  unnest_tokens(word, text_raw)

tokens_filtered <- tokens_C2 |> 
  filter(word %in% mfw_300)

mfw_matrix <- tokens_filtered |>
  count(chunk_id, word) |>
  pivot_wider(id_cols = chunk_id, names_from = word, values_from = n, values_fill = 0)

author_info <- corpus_C2 |> select(chunk_id, author, n_words) |> distinct()

result <- mfw_matrix |> left_join(author_info, by = "chunk_id") |> 
  select(chunk_id, author, n_words, everything())

word_cols <- setdiff(names(result), c("chunk_id", "author", "n_words"))

for (col in word_cols) {
  result[[col]] <- result[[col]] / result[["n_words"]] * 1000
}

result_final_300 <- result |> select(-n_words)

dim(result_final_300)
write_csv(result_final_300, paste0(features_dir, "mfw_C.C2_300.csv"))



# --- MFW500 ---
tokens_filtered <- tokens_C2 |> filter(word %in% mfw_500)

mfw_matrix <- tokens_filtered |>
  count(chunk_id, word) |>
  pivot_wider(id_cols = chunk_id, names_from = word, values_from = n, values_fill = 0)

result <- mfw_matrix |> left_join(author_info, by = "chunk_id") |> 
  select(chunk_id, author, n_words, everything())

word_cols <- setdiff(names(result), c("chunk_id", "author", "n_words"))

for (col in word_cols) {
  result[[col]] <- result[[col]] / result[["n_words"]] * 1000
}

result_final_500 <- result |> select(-n_words)

dim(result_final_500)
write_csv(result_final_500, paste0(features_dir, "mfw_C.C2_500.csv"))

# Проверка на X1
any(grepl("^X[0-9]+$", colnames(result_final_300)))
any(grepl("^X[0-9]+$", colnames(result_final_500)))




# =========
# Матрица для C3 (MFW300 и MFW500)


# =======


corpus_C3 <- read_csv(paste0(data_dir, "corpus_C3.csv"))

# --- MFW300 ---

tokens_C3 <- corpus_C3 |> unnest_tokens(word, text_raw)

tokens_filtered <- tokens_C3 |> filter(word %in% mfw_300)

mfw_matrix <- tokens_filtered |>
  count(chunk_id, word) |>
  pivot_wider(id_cols = chunk_id, names_from = word, values_from = n, values_fill = 0)

author_info <- corpus_C3 |> select(chunk_id, author, n_words) |> distinct()

result <- mfw_matrix |> left_join(author_info, by = "chunk_id") |> 
  select(chunk_id, author, n_words, everything())

word_cols <- setdiff(names(result), c("chunk_id", "author", "n_words"))

for (col in word_cols) {
  result[[col]] <- result[[col]] / result[["n_words"]] * 1000
}

result_final_300 <- result |> select(-n_words)

dim(result_final_300)
write_csv(result_final_300, paste0(features_dir, "mfw_C.C3_300.csv"))



# --- MFW500 ---

tokens_filtered <- tokens_C3 |> filter(word %in% mfw_500)

mfw_matrix <- tokens_filtered |>
  count(chunk_id, word) |>
  pivot_wider(id_cols = chunk_id, names_from = word, values_from = n, values_fill = 0)

result <- mfw_matrix |> left_join(author_info, by = "chunk_id") |> 
  select(chunk_id, author, n_words, everything())

word_cols <- setdiff(names(result), c("chunk_id", "author", "n_words"))

for (col in word_cols) {
  result[[col]] <- result[[col]] / result[["n_words"]] * 1000
}

result_final_500 <- result |> select(-n_words)

dim(result_final_500)
write_csv(result_final_500, paste0(features_dir, "mfw_C.C3_500.csv"))

# Проверка
any(grepl("^X[0-9]+$", colnames(result_final_300)))
any(grepl("^X[0-9]+$", colnames(result_final_500)))




# =========
# Матрица для DUBIA (MFW300 и MFW500)


# =======

library(tidyverse)
library(tidytext)

# Пути
data_dir <- "/Users/anastasiabogdanova/R_directory/iskra-project/data/processed/"
features_dir <- "/Users/anastasiabogdanova/R_directory/iskra-project/features/"

# Загружаем фиксированные словари (из мастер-корпуса C)
mfw_300 <- read_lines(paste0(features_dir, "mfw_300_C_master.txt"))
mfw_500 <- read_lines(paste0(features_dir, "mfw_500_C_master.txt"))

# Проверка
cat("MFW300:", length(mfw_300), "слов\n")
cat("MFW500:", length(mfw_500), "слов\n")
head(mfw_300, 10)

# Загружаем dubia (уточните название файла!)
dubia <- read_csv("/Users/anastasiabogdanova/R_directory/iskra-project/data/processed/corpus_dubia.csv")

# Проверяем структуру dubia
glimpse(dubia)

# Проверяем, что все колонки есть
cat("Колонки в dubia:", paste(colnames(dubia), collapse=", "), "\n")


# =====
# Строим матрицу dubia_MFW300


# Токенизация
tokens_dubia <- dubia |>
  unnest_tokens(word, text_raw)

# Фильтруем по словарю MFW300
tokens_filtered <- tokens_dubia |>
  filter(word %in% mfw_300)

# Создаем матрицу частот
mfw_matrix <- tokens_filtered |>
  count(chunk_id, word) |>
  pivot_wider(id_cols = chunk_id, names_from = word, values_from = n, values_fill = 0)

# Добавляем метаданные
author_info <- dubia |>
  select(chunk_id, author, n_words) |>
  distinct()

result <- mfw_matrix |>
  left_join(author_info, by = "chunk_id") |>
  select(chunk_id, author, n_words, everything())

# Нормализация на 1000 слов
word_cols <- setdiff(names(result), c("chunk_id", "author", "n_words"))

for (col in word_cols) {
  result[[col]] <- result[[col]] / result[["n_words"]] * 1000
}


# Удаляем n_words
dubia_300 <- result |> select(-n_words)


# Загружаем C1_300 для получения списка всех нужных колонок
C1_300 <- read_csv(paste0(features_dir, "mfw_C.C1_300.csv"))
needed_cols <- colnames(C1_300)  # все колонки, включая chunk_id и author

# Добавляем НЕДОСТАЮЩИЕ колонки (которых нет в dubia_300)
missing_cols <- setdiff(needed_cols, colnames(dubia_300))

cat("Добавляем", length(missing_cols), "недостающих колонок\n")
cat("Примеры:", head(missing_cols, 10), "\n")

for (col in missing_cols) {
  dubia_300[[col]] <- 0
}


# Теперь приводим порядок колонок к тому же, что в C1_300
dubia_300 <- dubia_300 |> select(all_of(needed_cols))

# Проверка
cat("\n=== dubia_MFW300 ===\n")
cat("Размер матрицы:", dim(dubia_300), "\n")
cat("Колонки совпадают с C1_300:", identical(colnames(C1_300), colnames(dubia_300)), "\n")
cat("Наличие X1:", any(grepl("^X[0-9]+$", colnames(dubia_300))), "\n")

# Сохраняем
write_csv(dubia_300, paste0(features_dir, "dubia_C_mfw_300.csv"))
cat("Сохранено:", paste0(features_dir, "dubia_C_mfw_300.csv"), "\n")



# =====
# Строим матрицу dubia_MFW500

# Фильтруем по словарю MFW500
tokens_filtered <- tokens_dubia |>
  filter(word %in% mfw_500)

# Создаем матрицу частот
mfw_matrix <- tokens_filtered |>
  count(chunk_id, word) |>
  pivot_wider(id_cols = chunk_id, names_from = word, values_from = n, values_fill = 0)

# Добавляем метаданные
result <- mfw_matrix |>
  left_join(author_info, by = "chunk_id") |>
  select(chunk_id, author, n_words, everything())

# Нормализация
word_cols <- setdiff(names(result), c("chunk_id", "author", "n_words"))

for (col in word_cols) {
  result[[col]] <- result[[col]] / result[["n_words"]] * 1000
}

# Удаляем n_words
dubia_500 <- result |> select(-n_words)

# Загружаем C1_500 для получения списка всех нужных колонок
C1_500 <- read_csv(paste0(features_dir, "mfw_C.C1_500.csv"))

needed_cols_500 <- colnames(C1_500)

# Добавляем НЕДОСТАЮЩИЕ колонки
missing_cols <- setdiff(needed_cols_500, colnames(dubia_500))

cat("\nДобавляем", length(missing_cols), "недостающих колонок для MFW500\n")

for (col in missing_cols) {
  dubia_500[[col]] <- 0
}

# Приводим порядок колонок
dubia_500 <- dubia_500 |> select(all_of(needed_cols_500))

# Проверка
cat("\n=== dubia_MFW500 ===\n")
cat("Размер матрицы:", dim(dubia_500), "\n")
cat("Колонки совпадают с C1_500:", identical(colnames(C1_500), colnames(dubia_500)), "\n")
cat("Наличие X1:", any(grepl("^X[0-9]+$", colnames(dubia_500))), "\n")

# Сохраняем
write_csv(dubia_500, paste0(features_dir, "dubia_C_mfw_500.csv"))
cat("Сохранено:", paste0(features_dir, "dubia_C_mfw_500.csv"), "\n")



# ФИНАЛЬНАЯ Проверка
# Проверяем, что все матрицы имеют одинаковые колонки
cat("\n=== ФИНАЛЬНАЯ ПРОВЕРКА ===\n")
cat("C1_300 vs dubia_300:", identical(colnames(C1_300), colnames(dubia_300)), "\n")
cat("C1_500 vs dubia_500:", identical(colnames(C1_500), colnames(dubia_500)), "\n")

# Смотрим на первые строки dubia
cat("\nПервые 5 строк dubia_300 (первые 10 колонок):\n")
print(head(dubia_300[, 1:10]))



# ====================

# Строим МОДЕЛИ - SVM
# ОБУЧЕНИЕ И ПРЕДСКАЗАНИЕ DUBIA

# ===================

# =======

# Библиотеки
library(tidyverse)
library(caret)
library(e1071)
library(yardstick)

# =================================================

# # C.C1 с MFW300 == CF_SVM_MFW300_71, CF_SVM_MFW300_71_83

# 1. ЗАГРУЗКА ДАННЫХ 

# ===============================================


mfw_clean <- read_csv("/Users/anastasiabogdanova/R_directory/iskra-project/features/mfw_C.C1_300.csv")

# Подготовка
X <- mfw_clean |> select(-chunk_id, -author)
y <- as.factor(mfw_clean$author)

cat("Размер матрицы:", dim(X), "\n")
cat("Распределение авторов:\n")
table(y)

# ============================================
# 2. STRATIFIED TRAIN/TEST SPLIT (80/20)
# ============================================

set.seed(818)  # фиксируем для воспроизводимости

df <- data.frame(X, author = y)
split <- initial_split(df, prop = 0.8, strata = author)

train_data <- training(split)
test_data  <- testing(split)

cat("\nTrain размер:", nrow(train_data), "\n")
cat("Test размер:", nrow(test_data), "\n")
cat("Train распределение:\n")
table(train_data$author)
cat("\nTest распределение:\n")
table(test_data$author)

# ============================================
# 3. SCALING (для SVM с радиальным ядром?)
# ============================================

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

# ============================================
# 4. SVM МОДЕЛЬ (линейное ядро + probability)
# ============================================

set.seed(818)

svm_model <- svm(
  author ~ .,
  data = train_data2,
  kernel = "linear",      # линейное ядро (хорошо для текстов)
  probability = TRUE,     # для predict_proba
  cost = 1                # можно будет настроить позже
)


# ============================================
# 4. SVM МОДЕЛЬ (радиальное ядро + probability)
# ============================================

set.seed(818)

n_features <- 300

svm_model <- svm(
  author ~ .,
  data = train_data2,
  kernel = "radial",        # RBF/радиальное ядро
  probability = TRUE,       # для predict_proba
  cost = 1,                 # регуляризация
  gamma = 1 / n_features    # гамма = 1 / число признаков
)

# ============================================
# 5. ПРЕДСКАЗАНИЕ НА ТЕСТОВОЙ ВЫБОРКЕ
# ============================================

# Жесткие предсказания
pred_class <- predict(svm_model, test_data2)

# Вероятности (для диагностической функции)
pred_prob <- attr(predict(svm_model, test_data2, probability = TRUE), "probabilities")

# ============================================
# 6. МЕТРИКИ
# ============================================

# Accuracy
accuracy <- mean(pred_class == test_data2$author)
cat("\n=== РЕЗУЛЬТАТЫ ===\n")
cat("Accuracy:", round(accuracy, 4), "\n")

# F1 macro (через yardstick)
df_results <- data.frame(
  truth = test_data2$author,
  pred = pred_class
)

f1_macro <- f_meas(df_results, truth, pred, estimator = "macro")
cat("F1 (macro):", round(f1_macro$.estimate, 4), "\n")

# Confusion matrix
cat("\nConfusion matrix:\n")
conf_matrix <- table(Predicted = pred_class, Actual = test_data2$author)
print(conf_matrix)

# Выгрузка
cm <- table(Predicted = pred_class, Actual = test_data2$author)
cat(capture.output(write.table(cm, sep = "\t", col.names = NA)), sep = "\n")

# ============================================
# 7. СОХРАНЕНИЕ МОДЕЛИ
# ============================================

saveRDS(svm_model, "/Users/anastasiabogdanova/R_directory/iskra-project/models/svm_model_C.C1_300_F1_0.888.rds")
#saveRDS(list(scale_center = scale_center, scale_scale = scale_scale), "/Users/anastasiabogdanova/R_directory/iskra-project/models/scaling_params_C1_300.rds")

cat("\nМодель и параметры scaling сохранены\n")



# ============================================
#Применение к dubia


# Загружаем dubia матрицу
dubia_matrix <- read_csv("/Users/anastasiabogdanova/R_directory/iskra-project/features/dubia_C_mfw_300.csv")


# Подготовка dubia (только признаки, без chunk_id и author)
dubia_X <- dubia_matrix |> select(-chunk_id, -author)

# Масштабирование (используя параметры из train_data)
dubia_X_scaled <- scale(dubia_X,
                        center = scale_center,
                        scale = scale_scale)



#-----ЧИНИМ МАТРИЦЫ!!! --- start

# Проверяем наличие X1 в train_data2
cat("X1 в train_data2:", "X1" %in% colnames(train_data2), "\n")

# Проверяем наличие X1 в dubia_X_scaled
cat("X1 в dubia_X_scaled:", "X1" %in% colnames(dubia_X_scaled), "\n")

# Проверяем, совпадают ли колонки
cat("Колонки совпадают:", identical(colnames(train_data2[, -ncol(train_data2)]), 
                                    colnames(dubia_X_scaled)), "\n")

# Если не совпадают — показываем
if (!identical(colnames(train_data2[, -ncol(train_data2)]), colnames(dubia_X_scaled))) {
  cat("\nКолонки в train, которых нет в dubia:\n")
  print(setdiff(colnames(train_data2[, -ncol(train_data2)]), colnames(dubia_X_scaled)))
  
  cat("\nКолонки в dubia, которых нет в train:\n")
  print(setdiff(colnames(dubia_X_scaled), colnames(train_data2[, -ncol(train_data2)])))
}


# Загружаем текущие словари
mfw_300 <- read_lines("/Users/anastasiabogdanova/R_directory/iskra-project/features/mfw_300_C_master.txt")
mfw_500 <- read_lines("/Users/anastasiabogdanova/R_directory/iskra-project/features/mfw_500_C_master.txt")


# Функция для удаления цифровых слов (чистые числа и числа с X)
remove_numeric_words <- function(words) {
  # Удаляем: чистые цифры, слова начинающиеся с X и затем цифры, слова с цифрами
  cleaned <- words[!grepl("^[0-9]+$", words)]  # "1", "1905"
  cleaned <- cleaned[!grepl("^X[0-9]+$", cleaned)]  # "X1", "X1905"
  cleaned <- cleaned[!grepl("[0-9]", cleaned)]  # любые слова с цифрами
  return(cleaned)
}


# Очищаем
mfw_300_clean <- remove_numeric_words(mfw_300)
mfw_500_clean <- remove_numeric_words(mfw_500)


# Сколько удалили?
cat("MFW300: было", length(mfw_300), "стало", length(mfw_300_clean), "\n")
cat("MFW500: было", length(mfw_500), "стало", length(mfw_500_clean), "\n")



#---- ЧИНИМ МАТРИЦЫ finish




# Предсказание вероятностей для dubia
dubia_prob <- attr(predict(svm_model, dubia_X_scaled, probability = TRUE), "probabilities")



# ==============================

# Определяем функцию DIAGNOSE_TEXT

# =============================

# Полная версия функции
diagnose_text <- function(prob_row, 
                          chunk_id_value,      # id текста
                          true_author_value = NULL,  # можно передать, если знаем
                          threshold = 0.5) {
  
  sorted <- sort(prob_row, decreasing = TRUE)
  top1 <- sorted[1]
  top2 <- sorted[2]
  ratio <- top1 / top2
  
  # Определяем статус
  if (top1 < threshold) {
    status_tag <- "COLLABORATIVE / UNKNOWN"
    interpretation <- "Низкая уверенность — возможна редактура или чужой автор"
  } else if (ratio < 3) {
    status_tag <- "MIXED"
    interpretation <- "Топ-2 автора слишком близки — следы чужого влияния"
  } else {
    status_tag <- "CLEAN"
    interpretation <- "Стилистически чистый текст"
  }
  
  # Собираем результат
  result <- list(
    chunk_id = chunk_id_value,
    status = status_tag,
    interpretation = interpretation,
    top1_author = names(top1),
    top1_prob = round(top1 * 100, 1),
    top2_author = names(top2),
    top2_prob = round(top2 * 100, 1),
    ratio = round(ratio, 2),
    all_probs = round(prob_row * 100, 1)
  )
  
  # Если передан настоящий автор — добавим сравнение
  if (!is.null(true_author_value)) {
    result$true_author <- true_author_value
    result$correct <- (names(top1) == true_author_value)
    if (!result$correct && top1 >= threshold) {
      result$warning <- "⚠️ Модель уверена, но ошиблась — возможно, спорный текст"
    }
  }
  
  return(result)
}


# Применяем диагностику
dubia_results <- list()
for (i in 1:nrow(dubia_matrix)) {
  dubia_results[[i]] <- diagnose_text(
    prob_row = dubia_prob[i, ],
    chunk_id_value = dubia_matrix$chunk_id[i],
    true_author_value = "dubia",  # указываем, что это неизвестный текст
    threshold = 0.5
  )
}

# Создаем удобную таблицу результатов
dubia_table <- data.frame(
  chunk_id = character(),
  true_author = character(),
  predicted = character(),
  plehanov = numeric(),
  parvus = numeric(),
  ortodox = numeric(),
  zasulich = numeric(),
  martov = numeric(),
  krupskaya = numeric(),
  lenin = numeric(),
  trotsky = numeric(),
  status = character(),
  top1_prob = numeric(),
  top2_author = character(),
  top2_prob = numeric(),
  ratio = numeric()
)

for (i in 1:length(dubia_results)) {
  res <- dubia_results[[i]]
  
  # Извлекаем вероятности для всех авторов
  probs <- res$all_probs
  names(probs) <- gsub("\\.", " ", names(probs))  # восстанавливаем имена
  
  # Создаем строку таблицы
  row_data <- data.frame(
    chunk_id = res$chunk_id,
    true_author = "dubia",
    predicted = res$top1_author,
    plehanov = ifelse("plehanov" %in% names(probs), probs["plehanov"], NA),
    parvus = ifelse("parvus" %in% names(probs), probs["parvus"], NA),
    ortodox = ifelse("ortodox" %in% names(probs), probs["ortodox"], NA),
    zasulich = ifelse("zasulich" %in% names(probs), probs["zasulich"], NA),
    martov = ifelse("martov" %in% names(probs), probs["martov"], NA),
    krupskaya = ifelse("krupskaya" %in% names(probs), probs["krupskaya"], NA),
    lenin = ifelse("lenin" %in% names(probs), probs["lenin"], NA),
    trotsky = ifelse("trotsky" %in% names(probs), probs["trotsky"], NA),
    status = res$status,
    top1_prob = res$top1_prob,
    top2_author = res$top2_author,
    top2_prob = res$top2_prob,
    ratio = res$ratio
  )
  
  dubia_table <- rbind(dubia_table, row_data)
}

# Просматриваем результат
print(dubia_table)


# Полная таблица результатов (для копирования в Google Sheets)
cat("\n=== DUBIA RESULTS (скопируйте таблицу ниже) ===\n\n")
cat(capture.output(write.table(dubia_table, sep = "\t", row.names = FALSE)), sep = "\n")


# Сохраняем в CSV
write_csv(dubia_table, "/Users/anastasiabogdanova/R_directory/iskra-project/results/dubia_predictions_SVM_71.csv")
cat("\nРезультаты сохранены в: /Users/anastasiabogdanova/R_directory/iskra-project/results/dubia_predictions_SVM_71.csv\n")



# Краткий вывод
# Сводка по статусам чанков
cat("\n=== СВОДКА ПО DUBIA ===\n")
table(dubia_table$status)

# Какие чанки получили какой статус
cat("\n=== ДЕТАЛИ ПО ЧАНКАМ ===\n")
cm <- dubia_table |> 
  select(chunk_id, predicted, top1_prob, status) |> 
  print()

#выгрузка
cat(capture.output(write.table(cm, sep = "\t", row.names = FALSE)), sep = "\n")




# ============================================

# C.C2_MFW300: CF_SVM_MFW300_72, CF_SVM_MFW300_72_84

# ============================================


# 1. ЗАГРУЗКА ДАННЫХ
mfw_clean <- read_csv("/Users/anastasiabogdanova/R_directory/iskra-project/features/mfw_C.C2_300.csv")
dubia_matrix <- read_csv("/Users/anastasiabogdanova/R_directory/iskra-project/features/dubia_C_mfw_300.csv")

# 2. ПОДГОТОВКА
X <- mfw_clean |> select(-chunk_id, -author)
y <- as.factor(mfw_clean$author)

# 3. TRAIN/TEST SPLIT
set.seed(818)  # можно поменять seed для разных реплик

df <- data.frame(X, author = y)

split <- initial_split(df, prop = 0.8, strata = author)

train_data <- training(split)
test_data <- testing(split)

# 4. SCALING
train_X <- train_data |> select(-author)
test_X <- test_data |> select(-author)

train_X_scaled <- scale(train_X)
scale_center <- attr(train_X_scaled, "scaled:center")
scale_scale <- attr(train_X_scaled, "scaled:scale")

test_X_scaled <- scale(test_X, center = scale_center, scale = scale_scale)
train_data2 <- data.frame(train_X_scaled, author = train_data$author)
test_data2 <- data.frame(test_X_scaled, author = test_data$author)

# 5. SVM МОДЕЛЬ == ЛИНЕЙНОЕ ЯДРО
set.seed(818)

svm_model <- svm(author ~ ., data = train_data2, kernel = "linear", probability = TRUE, cost = 1)

#!!! SVM RADIAL!! new
set.seed(818)

n_features <- 300

svm_model <- svm(
  author ~ .,
  data = train_data2,
  kernel = "radial",        # RBF/радиальное ядро
  probability = TRUE,       # для predict_proba
  cost = 1,                 # регуляризация
  gamma = 1 / n_features    # гамма = 1 / число признаков
)


# 6. ОЦЕНКА НА ТЕСТЕ
pred_class <- predict(svm_model, test_data2)
pred_prob <- attr(predict(svm_model, test_data2, probability = TRUE), "probabilities")

accuracy <- mean(pred_class == test_data2$author)
df_results <- data.frame(truth = test_data2$author, pred = pred_class)
f1_macro <- f_meas(df_results, truth, pred, estimator = "macro")

cat("\n=== C2_300 РЕЗУЛЬТАТЫ ===\n")
cat("Accuracy:", round(accuracy, 4), "\n")
cat("F1 (macro):", round(f1_macro$.estimate, 4), "\n")


# Confusion matrix
cat("\nConfusion matrix:\n")
conf_matrix <- table(Predicted = pred_class, Actual = test_data2$author)
print(conf_matrix)

# Выгрузка
cm <- table(Predicted = pred_class, Actual = test_data2$author)
cat(capture.output(write.table(cm, sep = "\t", col.names = NA)), sep = "\n")


# 7. ПОДГОТОВКА DUBIA
dubia_X <- dubia_matrix |> select(-chunk_id, -author)

# Добавляем недостающие колонки (если есть)
missing_cols <- setdiff(colnames(train_X), colnames(dubia_X))

if (length(missing_cols) > 0) {
  for (col in missing_cols) {
    dubia_X[[col]] <- 0
  }
}

dubia_X <- dubia_X |> select(all_of(colnames(train_X)))
dubia_X_scaled <- scale(dubia_X, center = scale_center, scale = scale_scale)

# 8. ПРЕДСКАЗАНИЕ DUBIA
dubia_prob <- attr(predict(svm_model, dubia_X_scaled, probability = TRUE), "probabilities")

# ==============================

#  9. Определяем функцию DIAGNOSE_TEXT

# =============================

# Полная версия функции
diagnose_text <- function(prob_row, 
                          chunk_id_value,      # id текста
                          true_author_value = NULL,  # можно передать, если знаем
                          threshold = 0.5) {
  
  sorted <- sort(prob_row, decreasing = TRUE)
  top1 <- sorted[1]
  top2 <- sorted[2]
  ratio <- top1 / top2
  
  # Определяем статус
  if (top1 < threshold) {
    status_tag <- "COLLABORATIVE / UNKNOWN"
    interpretation <- "Низкая уверенность — возможна редактура или чужой автор"
  } else if (ratio < 3) {
    status_tag <- "MIXED"
    interpretation <- "Топ-2 автора слишком близки — следы чужого влияния"
  } else {
    status_tag <- "CLEAN"
    interpretation <- "Стилистически чистый текст"
  }
  
  # Собираем результат
  result <- list(
    chunk_id = chunk_id_value,
    status = status_tag,
    interpretation = interpretation,
    top1_author = names(top1),
    top1_prob = round(top1 * 100, 1),
    top2_author = names(top2),
    top2_prob = round(top2 * 100, 1),
    ratio = round(ratio, 2),
    all_probs = round(prob_row * 100, 1)
  )
  
  # Если передан настоящий автор — добавим сравнение
  if (!is.null(true_author_value)) {
    result$true_author <- true_author_value
    result$correct <- (names(top1) == true_author_value)
    if (!result$correct && top1 >= threshold) {
      result$warning <- "⚠️ Модель уверена, но ошиблась — возможно, спорный текст"
    }
  }
  
  return(result)
}


# Применяем диагностику
dubia_results <- list()
for (i in 1:nrow(dubia_matrix)) {
  dubia_results[[i]] <- diagnose_text(
    prob_row = dubia_prob[i, ],
    chunk_id_value = dubia_matrix$chunk_id[i],
    true_author_value = "dubia",  # указываем, что это неизвестный текст
    threshold = 0.5
  )
}

# Создаем удобную таблицу результатов
dubia_table <- data.frame(
  chunk_id = character(),
  true_author = character(),
  predicted = character(),
  plehanov = numeric(),
  parvus = numeric(),
  ortodox = numeric(),
  zasulich = numeric(),
  martov = numeric(),
  krupskaya = numeric(),
  lenin = numeric(),
  trotsky = numeric(),
  status = character(),
  top1_prob = numeric(),
  top2_author = character(),
  top2_prob = numeric(),
  ratio = numeric()
)

for (i in 1:length(dubia_results)) {
  res <- dubia_results[[i]]
  
  # Извлекаем вероятности для всех авторов
  probs <- res$all_probs
  names(probs) <- gsub("\\.", " ", names(probs))  # восстанавливаем имена
  
  # Создаем строку таблицы
  row_data <- data.frame(
    chunk_id = res$chunk_id,
    true_author = "dubia",
    predicted = res$top1_author,
    plehanov = ifelse("plehanov" %in% names(probs), probs["plehanov"], NA),
    parvus = ifelse("parvus" %in% names(probs), probs["parvus"], NA),
    ortodox = ifelse("ortodox" %in% names(probs), probs["ortodox"], NA),
    zasulich = ifelse("zasulich" %in% names(probs), probs["zasulich"], NA),
    martov = ifelse("martov" %in% names(probs), probs["martov"], NA),
    krupskaya = ifelse("krupskaya" %in% names(probs), probs["krupskaya"], NA),
    lenin = ifelse("lenin" %in% names(probs), probs["lenin"], NA),
    trotsky = ifelse("trotsky" %in% names(probs), probs["trotsky"], NA),
    status = res$status,
    top1_prob = res$top1_prob,
    top2_author = res$top2_author,
    top2_prob = res$top2_prob,
    ratio = res$ratio
  )
  
  dubia_table <- rbind(dubia_table, row_data)
}

# Просматриваем результат
print(dubia_table)


# Полная таблица результатов (для копирования в Google Sheets)
cat("\n=== DUBIA RESULTS (скопируйте таблицу ниже) ===\n\n")
cat(capture.output(write.table(dubia_table, sep = "\t", row.names = FALSE)), sep = "\n")


# Сохраняем в CSV
write_csv(dubia_table, "/Users/anastasiabogdanova/R_directory/iskra-project/results/dubia_predictions_SVM_72.csv")
cat("\nРезультаты сохранены в: /Users/anastasiabogdanova/R_directory/iskra-project/results/dubia_predictions_SVM_72.csv\n")



# Краткий вывод
# Сводка по статусам чанков
cat("\n=== СВОДКА ПО DUBIA ===\n")
table(dubia_table$status)

# Какие чанки получили какой статус
cat("\n=== ДЕТАЛИ ПО ЧАНКАМ ===\n")
cm <- dubia_table |> 
  select(chunk_id, predicted, top1_prob, status) |> 
  print()

#выгрузка
cat(capture.output(write.table(cm, sep = "\t", row.names = FALSE)), sep = "\n")






# ============================================

# C.C3_MFW300: CF_SVM_MFW300_73, CF_SVM_MFW300_73_85

# ============================================


# 1. ЗАГРУЗКА ДАННЫХ
mfw_clean <- read_csv("/Users/anastasiabogdanova/R_directory/iskra-project/features/mfw_C.C3_300.csv")
dubia_matrix <- read_csv("/Users/anastasiabogdanova/R_directory/iskra-project/features/dubia_C_mfw_300.csv")

# 2. ПОДГОТОВКА
X <- mfw_clean |> select(-chunk_id, -author)
y <- as.factor(mfw_clean$author)

# 3. TRAIN/TEST SPLIT
set.seed(818)  # можно поменять seed для разных реплик

df <- data.frame(X, author = y)

split <- initial_split(df, prop = 0.8, strata = author)

train_data <- training(split)
test_data <- testing(split)

# 4. SCALING
train_X <- train_data |> select(-author)
test_X <- test_data |> select(-author)

train_X_scaled <- scale(train_X)
scale_center <- attr(train_X_scaled, "scaled:center")
scale_scale <- attr(train_X_scaled, "scaled:scale")

test_X_scaled <- scale(test_X, center = scale_center, scale = scale_scale)
train_data2 <- data.frame(train_X_scaled, author = train_data$author)
test_data2 <- data.frame(test_X_scaled, author = test_data$author)

# 5. SVM МОДЕЛЬ LINEAR
set.seed(818)

svm_model <- svm(author ~ ., data = train_data2, kernel = "linear", probability = TRUE, cost = 1)


# !! SVM RADIAL!!!
set.seed(818)

n_features <- 300

svm_model <- svm(
  author ~ .,
  data = train_data2,
  kernel = "radial",        # RBF/радиальное ядро
  probability = TRUE,       # для predict_proba
  cost = 1,                 # регуляризация
  gamma = 1 / n_features    # гамма = 1 / число признаков
)


# 6. ОЦЕНКА НА ТЕСТЕ
pred_class <- predict(svm_model, test_data2)
pred_prob <- attr(predict(svm_model, test_data2, probability = TRUE), "probabilities")

accuracy <- mean(pred_class == test_data2$author)
df_results <- data.frame(truth = test_data2$author, pred = pred_class)
f1_macro <- f_meas(df_results, truth, pred, estimator = "macro")

cat("\n=== C2_300 РЕЗУЛЬТАТЫ ===\n")
cat("Accuracy:", round(accuracy, 4), "\n")
cat("F1 (macro):", round(f1_macro$.estimate, 4), "\n")


# Confusion matrix
cat("\nConfusion matrix:\n")
conf_matrix <- table(Predicted = pred_class, Actual = test_data2$author)
print(conf_matrix)

# Выгрузка
cm <- table(Predicted = pred_class, Actual = test_data2$author)
cat(capture.output(write.table(cm, sep = "\t", col.names = NA)), sep = "\n")


# 7. ПОДГОТОВКА DUBIA
dubia_X <- dubia_matrix |> select(-chunk_id, -author)

# Добавляем недостающие колонки (если есть)
missing_cols <- setdiff(colnames(train_X), colnames(dubia_X))

if (length(missing_cols) > 0) {
  for (col in missing_cols) {
    dubia_X[[col]] <- 0
  }
}

dubia_X <- dubia_X |> select(all_of(colnames(train_X)))
dubia_X_scaled <- scale(dubia_X, center = scale_center, scale = scale_scale)

# 8. ПРЕДСКАЗАНИЕ DUBIA
dubia_prob <- attr(predict(svm_model, dubia_X_scaled, probability = TRUE), "probabilities")

# ==============================

#  9. Определяем функцию DIAGNOSE_TEXT

# =============================

# Полная версия функции
diagnose_text <- function(prob_row, 
                          chunk_id_value,      # id текста
                          true_author_value = NULL,  # можно передать, если знаем
                          threshold = 0.5) {
  
  sorted <- sort(prob_row, decreasing = TRUE)
  top1 <- sorted[1]
  top2 <- sorted[2]
  ratio <- top1 / top2
  
  # Определяем статус
  if (top1 < threshold) {
    status_tag <- "COLLABORATIVE / UNKNOWN"
    interpretation <- "Низкая уверенность — возможна редактура или чужой автор"
  } else if (ratio < 3) {
    status_tag <- "MIXED"
    interpretation <- "Топ-2 автора слишком близки — следы чужого влияния"
  } else {
    status_tag <- "CLEAN"
    interpretation <- "Стилистически чистый текст"
  }
  
  # Собираем результат
  result <- list(
    chunk_id = chunk_id_value,
    status = status_tag,
    interpretation = interpretation,
    top1_author = names(top1),
    top1_prob = round(top1 * 100, 1),
    top2_author = names(top2),
    top2_prob = round(top2 * 100, 1),
    ratio = round(ratio, 2),
    all_probs = round(prob_row * 100, 1)
  )
  
  # Если передан настоящий автор — добавим сравнение
  if (!is.null(true_author_value)) {
    result$true_author <- true_author_value
    result$correct <- (names(top1) == true_author_value)
    if (!result$correct && top1 >= threshold) {
      result$warning <- "⚠️ Модель уверена, но ошиблась — возможно, спорный текст"
    }
  }
  
  return(result)
}


# Применяем диагностику
dubia_results <- list()
for (i in 1:nrow(dubia_matrix)) {
  dubia_results[[i]] <- diagnose_text(
    prob_row = dubia_prob[i, ],
    chunk_id_value = dubia_matrix$chunk_id[i],
    true_author_value = "dubia",  # указываем, что это неизвестный текст
    threshold = 0.5
  )
}

# Создаем удобную таблицу результатов
dubia_table <- data.frame(
  chunk_id = character(),
  true_author = character(),
  predicted = character(),
  plehanov = numeric(),
  parvus = numeric(),
  ortodox = numeric(),
  zasulich = numeric(),
  martov = numeric(),
  krupskaya = numeric(),
  lenin = numeric(),
  trotsky = numeric(),
  status = character(),
  top1_prob = numeric(),
  top2_author = character(),
  top2_prob = numeric(),
  ratio = numeric()
)

for (i in 1:length(dubia_results)) {
  res <- dubia_results[[i]]
  
  # Извлекаем вероятности для всех авторов
  probs <- res$all_probs
  names(probs) <- gsub("\\.", " ", names(probs))  # восстанавливаем имена
  
  # Создаем строку таблицы
  row_data <- data.frame(
    chunk_id = res$chunk_id,
    true_author = "dubia",
    predicted = res$top1_author,
    plehanov = ifelse("plehanov" %in% names(probs), probs["plehanov"], NA),
    parvus = ifelse("parvus" %in% names(probs), probs["parvus"], NA),
    ortodox = ifelse("ortodox" %in% names(probs), probs["ortodox"], NA),
    zasulich = ifelse("zasulich" %in% names(probs), probs["zasulich"], NA),
    martov = ifelse("martov" %in% names(probs), probs["martov"], NA),
    krupskaya = ifelse("krupskaya" %in% names(probs), probs["krupskaya"], NA),
    lenin = ifelse("lenin" %in% names(probs), probs["lenin"], NA),
    trotsky = ifelse("trotsky" %in% names(probs), probs["trotsky"], NA),
    status = res$status,
    top1_prob = res$top1_prob,
    top2_author = res$top2_author,
    top2_prob = res$top2_prob,
    ratio = res$ratio
  )
  
  dubia_table <- rbind(dubia_table, row_data)
}

# Просматриваем результат
print(dubia_table)


# Полная таблица результатов (для копирования в Google Sheets)
cat("\n=== DUBIA RESULTS (скопируйте таблицу ниже) ===\n\n")
cat(capture.output(write.table(dubia_table, sep = "\t", row.names = FALSE)), sep = "\n")


# Сохраняем в CSV
write_csv(dubia_table, "/Users/anastasiabogdanova/R_directory/iskra-project/results/dubia_predictions_SVM_73.csv")
cat("\nРезультаты сохранены в: /Users/anastasiabogdanova/R_directory/iskra-project/results/dubia_predictions_SVM_73.csv\n")



# Краткий вывод
# Сводка по статусам чанков
cat("\n=== СВОДКА ПО DUBIA ===\n")
table(dubia_table$status)

# Какие чанки получили какой статус
cat("\n=== ДЕТАЛИ ПО ЧАНКАМ ===\n")
cm <- dubia_table |> 
  select(chunk_id, predicted, top1_prob, status) |> 
  print()

#выгрузка
cat(capture.output(write.table(cm, sep = "\t", row.names = FALSE)), sep = "\n")





# ============================================

# C.C1_MFW500: CF_SVM_MFW500_74, CF_SVM_MFW500_74_86

# ============================================


# 1. ЗАГРУЗКА ДАННЫХ
mfw_clean <- read_csv("/Users/anastasiabogdanova/R_directory/iskra-project/features/mfw_C.C1_500.csv")
dubia_matrix <- read_csv("/Users/anastasiabogdanova/R_directory/iskra-project/features/dubia_C_mfw_500.csv")

# 2. ПОДГОТОВКА
X <- mfw_clean |> select(-chunk_id, -author)
y <- as.factor(mfw_clean$author)

# 3. TRAIN/TEST SPLIT
set.seed(818)  # можно поменять seed для разных реплик

df <- data.frame(X, author = y)

split <- initial_split(df, prop = 0.8, strata = author)

train_data <- training(split)
test_data <- testing(split)

# 4. SCALING
train_X <- train_data |> select(-author)
test_X <- test_data |> select(-author)

train_X_scaled <- scale(train_X)
scale_center <- attr(train_X_scaled, "scaled:center")
scale_scale <- attr(train_X_scaled, "scaled:scale")

test_X_scaled <- scale(test_X, center = scale_center, scale = scale_scale)
train_data2 <- data.frame(train_X_scaled, author = train_data$author)
test_data2 <- data.frame(test_X_scaled, author = test_data$author)

# 5. SVM МОДЕЛЬ - LINEAR
set.seed(818)

svm_model <- svm(author ~ ., data = train_data2, kernel = "linear", probability = TRUE, cost = 1)

# ! SVM RADIAL !!!
set.seed(818)

n_features <- 500

svm_model <- svm(
  author ~ .,
  data = train_data2,
  kernel = "radial",        # RBF/радиальное ядро
  probability = TRUE,       # для predict_proba
  cost = 1,                 # регуляризация
  gamma = 1 / n_features    # гамма = 1 / число признаков
)


# 6. ОЦЕНКА НА ТЕСТЕ
pred_class <- predict(svm_model, test_data2)
pred_prob <- attr(predict(svm_model, test_data2, probability = TRUE), "probabilities")

accuracy <- mean(pred_class == test_data2$author)
df_results <- data.frame(truth = test_data2$author, pred = pred_class)
f1_macro <- f_meas(df_results, truth, pred, estimator = "macro")

cat("\n=== C2_300 РЕЗУЛЬТАТЫ ===\n")
cat("Accuracy:", round(accuracy, 4), "\n")
cat("F1 (macro):", round(f1_macro$.estimate, 4), "\n")


# Confusion matrix
cat("\nConfusion matrix:\n")
conf_matrix <- table(Predicted = pred_class, Actual = test_data2$author)
print(conf_matrix)

# Выгрузка
cm <- table(Predicted = pred_class, Actual = test_data2$author)
cat(capture.output(write.table(cm, sep = "\t", col.names = NA)), sep = "\n")


# 7. ПОДГОТОВКА DUBIA
dubia_X <- dubia_matrix |> select(-chunk_id, -author)

# Добавляем недостающие колонки (если есть)
missing_cols <- setdiff(colnames(train_X), colnames(dubia_X))

if (length(missing_cols) > 0) {
  for (col in missing_cols) {
    dubia_X[[col]] <- 0
  }
}

dubia_X <- dubia_X |> select(all_of(colnames(train_X)))
dubia_X_scaled <- scale(dubia_X, center = scale_center, scale = scale_scale)

# 8. ПРЕДСКАЗАНИЕ DUBIA
dubia_prob <- attr(predict(svm_model, dubia_X_scaled, probability = TRUE), "probabilities")

# ==============================

#  9. Определяем функцию DIAGNOSE_TEXT

# =============================

# Полная версия функции
diagnose_text <- function(prob_row, 
                          chunk_id_value,      # id текста
                          true_author_value = NULL,  # можно передать, если знаем
                          threshold = 0.5) {
  
  sorted <- sort(prob_row, decreasing = TRUE)
  top1 <- sorted[1]
  top2 <- sorted[2]
  ratio <- top1 / top2
  
  # Определяем статус
  if (top1 < threshold) {
    status_tag <- "COLLABORATIVE / UNKNOWN"
    interpretation <- "Низкая уверенность — возможна редактура или чужой автор"
  } else if (ratio < 3) {
    status_tag <- "MIXED"
    interpretation <- "Топ-2 автора слишком близки — следы чужого влияния"
  } else {
    status_tag <- "CLEAN"
    interpretation <- "Стилистически чистый текст"
  }
  
  # Собираем результат
  result <- list(
    chunk_id = chunk_id_value,
    status = status_tag,
    interpretation = interpretation,
    top1_author = names(top1),
    top1_prob = round(top1 * 100, 1),
    top2_author = names(top2),
    top2_prob = round(top2 * 100, 1),
    ratio = round(ratio, 2),
    all_probs = round(prob_row * 100, 1)
  )
  
  # Если передан настоящий автор — добавим сравнение
  if (!is.null(true_author_value)) {
    result$true_author <- true_author_value
    result$correct <- (names(top1) == true_author_value)
    if (!result$correct && top1 >= threshold) {
      result$warning <- "⚠️ Модель уверена, но ошиблась — возможно, спорный текст"
    }
  }
  
  return(result)
}


# Применяем диагностику
dubia_results <- list()
for (i in 1:nrow(dubia_matrix)) {
  dubia_results[[i]] <- diagnose_text(
    prob_row = dubia_prob[i, ],
    chunk_id_value = dubia_matrix$chunk_id[i],
    true_author_value = "dubia",  # указываем, что это неизвестный текст
    threshold = 0.5
  )
}

# Создаем удобную таблицу результатов
dubia_table <- data.frame(
  chunk_id = character(),
  true_author = character(),
  predicted = character(),
  plehanov = numeric(),
  parvus = numeric(),
  ortodox = numeric(),
  zasulich = numeric(),
  martov = numeric(),
  krupskaya = numeric(),
  lenin = numeric(),
  trotsky = numeric(),
  status = character(),
  top1_prob = numeric(),
  top2_author = character(),
  top2_prob = numeric(),
  ratio = numeric()
)

for (i in 1:length(dubia_results)) {
  res <- dubia_results[[i]]
  
  # Извлекаем вероятности для всех авторов
  probs <- res$all_probs
  names(probs) <- gsub("\\.", " ", names(probs))  # восстанавливаем имена
  
  # Создаем строку таблицы
  row_data <- data.frame(
    chunk_id = res$chunk_id,
    true_author = "dubia",
    predicted = res$top1_author,
    plehanov = ifelse("plehanov" %in% names(probs), probs["plehanov"], NA),
    parvus = ifelse("parvus" %in% names(probs), probs["parvus"], NA),
    ortodox = ifelse("ortodox" %in% names(probs), probs["ortodox"], NA),
    zasulich = ifelse("zasulich" %in% names(probs), probs["zasulich"], NA),
    martov = ifelse("martov" %in% names(probs), probs["martov"], NA),
    krupskaya = ifelse("krupskaya" %in% names(probs), probs["krupskaya"], NA),
    lenin = ifelse("lenin" %in% names(probs), probs["lenin"], NA),
    trotsky = ifelse("trotsky" %in% names(probs), probs["trotsky"], NA),
    status = res$status,
    top1_prob = res$top1_prob,
    top2_author = res$top2_author,
    top2_prob = res$top2_prob,
    ratio = res$ratio
  )
  
  dubia_table <- rbind(dubia_table, row_data)
}

# Просматриваем результат
print(dubia_table)


# Полная таблица результатов (для копирования в Google Sheets)
cat("\n=== DUBIA RESULTS (скопируйте таблицу ниже) ===\n\n")
cat(capture.output(write.table(dubia_table, sep = "\t", row.names = FALSE)), sep = "\n")


# Сохраняем в CSV
write_csv(dubia_table, "/Users/anastasiabogdanova/R_directory/iskra-project/results/dubia_predictions_SVM_74.csv")
cat("\nРезультаты сохранены в: /Users/anastasiabogdanova/R_directory/iskra-project/results/dubia_predictions_SVM_74.csv\n")



# Краткий вывод
# Сводка по статусам чанков
cat("\n=== СВОДКА ПО DUBIA ===\n")
table(dubia_table$status)

# Какие чанки получили какой статус
cat("\n=== ДЕТАЛИ ПО ЧАНКАМ ===\n")
cm <- dubia_table |> 
  select(chunk_id, predicted, top1_prob, status) |> 
  print()

#выгрузка
cat(capture.output(write.table(cm, sep = "\t", row.names = FALSE)), sep = "\n")






# ============================================

# C.C2_MFW500: CF_SVM_MFW500_75, CF_SVM_MFW500_75_87

# ============================================


# 1. ЗАГРУЗКА ДАННЫХ
mfw_clean <- read_csv("/Users/anastasiabogdanova/R_directory/iskra-project/features/mfw_C.C2_500.csv")
dubia_matrix <- read_csv("/Users/anastasiabogdanova/R_directory/iskra-project/features/dubia_C_mfw_500.csv")

# 2. ПОДГОТОВКА
X <- mfw_clean |> select(-chunk_id, -author)
y <- as.factor(mfw_clean$author)

# 3. TRAIN/TEST SPLIT
set.seed(818)  # можно поменять seed для разных реплик

df <- data.frame(X, author = y)

split <- initial_split(df, prop = 0.8, strata = author)

train_data <- training(split)
test_data <- testing(split)

# 4. SCALING
train_X <- train_data |> select(-author)
test_X <- test_data |> select(-author)

train_X_scaled <- scale(train_X)
scale_center <- attr(train_X_scaled, "scaled:center")
scale_scale <- attr(train_X_scaled, "scaled:scale")

test_X_scaled <- scale(test_X, center = scale_center, scale = scale_scale)
train_data2 <- data.frame(train_X_scaled, author = train_data$author)
test_data2 <- data.frame(test_X_scaled, author = test_data$author)

# 5. SVM МОДЕЛЬ - LINEAR
set.seed(818)

svm_model <- svm(author ~ ., data = train_data2, kernel = "linear", probability = TRUE, cost = 1)

# !! SVM RADIAL !!!
set.seed(818)

n_features <- 500

svm_model <- svm(
  author ~ .,
  data = train_data2,
  kernel = "radial",        # RBF/радиальное ядро
  probability = TRUE,       # для predict_proba
  cost = 1,                 # регуляризация
  gamma = 1 / n_features    # гамма = 1 / число признаков
)


# 6. ОЦЕНКА НА ТЕСТЕ
pred_class <- predict(svm_model, test_data2)
pred_prob <- attr(predict(svm_model, test_data2, probability = TRUE), "probabilities")

accuracy <- mean(pred_class == test_data2$author)
df_results <- data.frame(truth = test_data2$author, pred = pred_class)
f1_macro <- f_meas(df_results, truth, pred, estimator = "macro")

cat("\n=== C2_300 РЕЗУЛЬТАТЫ ===\n")
cat("Accuracy:", round(accuracy, 4), "\n")
cat("F1 (macro):", round(f1_macro$.estimate, 4), "\n")


# Confusion matrix
cat("\nConfusion matrix:\n")
conf_matrix <- table(Predicted = pred_class, Actual = test_data2$author)
print(conf_matrix)

# Выгрузка
cm <- table(Predicted = pred_class, Actual = test_data2$author)
cat(capture.output(write.table(cm, sep = "\t", col.names = NA)), sep = "\n")


# 7. ПОДГОТОВКА DUBIA
dubia_X <- dubia_matrix |> select(-chunk_id, -author)

# Добавляем недостающие колонки (если есть)
missing_cols <- setdiff(colnames(train_X), colnames(dubia_X))

if (length(missing_cols) > 0) {
  for (col in missing_cols) {
    dubia_X[[col]] <- 0
  }
}

dubia_X <- dubia_X |> select(all_of(colnames(train_X)))
dubia_X_scaled <- scale(dubia_X, center = scale_center, scale = scale_scale)

# 8. ПРЕДСКАЗАНИЕ DUBIA
dubia_prob <- attr(predict(svm_model, dubia_X_scaled, probability = TRUE), "probabilities")

# ==============================

#  9. Определяем функцию DIAGNOSE_TEXT

# =============================

# Полная версия функции
diagnose_text <- function(prob_row, 
                          chunk_id_value,      # id текста
                          true_author_value = NULL,  # можно передать, если знаем
                          threshold = 0.5) {
  
  sorted <- sort(prob_row, decreasing = TRUE)
  top1 <- sorted[1]
  top2 <- sorted[2]
  ratio <- top1 / top2
  
  # Определяем статус
  if (top1 < threshold) {
    status_tag <- "COLLABORATIVE / UNKNOWN"
    interpretation <- "Низкая уверенность — возможна редактура или чужой автор"
  } else if (ratio < 3) {
    status_tag <- "MIXED"
    interpretation <- "Топ-2 автора слишком близки — следы чужого влияния"
  } else {
    status_tag <- "CLEAN"
    interpretation <- "Стилистически чистый текст"
  }
  
  # Собираем результат
  result <- list(
    chunk_id = chunk_id_value,
    status = status_tag,
    interpretation = interpretation,
    top1_author = names(top1),
    top1_prob = round(top1 * 100, 1),
    top2_author = names(top2),
    top2_prob = round(top2 * 100, 1),
    ratio = round(ratio, 2),
    all_probs = round(prob_row * 100, 1)
  )
  
  # Если передан настоящий автор — добавим сравнение
  if (!is.null(true_author_value)) {
    result$true_author <- true_author_value
    result$correct <- (names(top1) == true_author_value)
    if (!result$correct && top1 >= threshold) {
      result$warning <- "⚠️ Модель уверена, но ошиблась — возможно, спорный текст"
    }
  }
  
  return(result)
}


# Применяем диагностику
dubia_results <- list()
for (i in 1:nrow(dubia_matrix)) {
  dubia_results[[i]] <- diagnose_text(
    prob_row = dubia_prob[i, ],
    chunk_id_value = dubia_matrix$chunk_id[i],
    true_author_value = "dubia",  # указываем, что это неизвестный текст
    threshold = 0.5
  )
}

# Создаем удобную таблицу результатов
dubia_table <- data.frame(
  chunk_id = character(),
  true_author = character(),
  predicted = character(),
  plehanov = numeric(),
  parvus = numeric(),
  ortodox = numeric(),
  zasulich = numeric(),
  martov = numeric(),
  krupskaya = numeric(),
  lenin = numeric(),
  trotsky = numeric(),
  status = character(),
  top1_prob = numeric(),
  top2_author = character(),
  top2_prob = numeric(),
  ratio = numeric()
)

for (i in 1:length(dubia_results)) {
  res <- dubia_results[[i]]
  
  # Извлекаем вероятности для всех авторов
  probs <- res$all_probs
  names(probs) <- gsub("\\.", " ", names(probs))  # восстанавливаем имена
  
  # Создаем строку таблицы
  row_data <- data.frame(
    chunk_id = res$chunk_id,
    true_author = "dubia",
    predicted = res$top1_author,
    plehanov = ifelse("plehanov" %in% names(probs), probs["plehanov"], NA),
    parvus = ifelse("parvus" %in% names(probs), probs["parvus"], NA),
    ortodox = ifelse("ortodox" %in% names(probs), probs["ortodox"], NA),
    zasulich = ifelse("zasulich" %in% names(probs), probs["zasulich"], NA),
    martov = ifelse("martov" %in% names(probs), probs["martov"], NA),
    krupskaya = ifelse("krupskaya" %in% names(probs), probs["krupskaya"], NA),
    lenin = ifelse("lenin" %in% names(probs), probs["lenin"], NA),
    trotsky = ifelse("trotsky" %in% names(probs), probs["trotsky"], NA),
    status = res$status,
    top1_prob = res$top1_prob,
    top2_author = res$top2_author,
    top2_prob = res$top2_prob,
    ratio = res$ratio
  )
  
  dubia_table <- rbind(dubia_table, row_data)
}

# Просматриваем результат
print(dubia_table)


# Полная таблица результатов (для копирования в Google Sheets)
cat("\n=== DUBIA RESULTS (скопируйте таблицу ниже) ===\n\n")
cat(capture.output(write.table(dubia_table, sep = "\t", row.names = FALSE)), sep = "\n")


# Сохраняем в CSV
write_csv(dubia_table, "/Users/anastasiabogdanova/R_directory/iskra-project/results/dubia_predictions_SVM_75.csv")
cat("\nРезультаты сохранены в: /Users/anastasiabogdanova/R_directory/iskra-project/results/dubia_predictions_SVM_75.csv\n")



# Краткий вывод
# Сводка по статусам чанков
cat("\n=== СВОДКА ПО DUBIA ===\n")
table(dubia_table$status)

# Какие чанки получили какой статус
cat("\n=== ДЕТАЛИ ПО ЧАНКАМ ===\n")
cm <- dubia_table |> 
  select(chunk_id, predicted, top1_prob, status) |> 
  print()

#выгрузка
cat(capture.output(write.table(cm, sep = "\t", row.names = FALSE)), sep = "\n")




# ============================================

# C.C3_MFW500: CF_SVM_MFW500_76, CF_SVM_MFW500_76_88

# ============================================


# 1. ЗАГРУЗКА ДАННЫХ
mfw_clean <- read_csv("/Users/anastasiabogdanova/R_directory/iskra-project/features/mfw_C.C3_500.csv")
dubia_matrix <- read_csv("/Users/anastasiabogdanova/R_directory/iskra-project/features/dubia_C_mfw_500.csv")

# 2. ПОДГОТОВКА
X <- mfw_clean |> select(-chunk_id, -author)
y <- as.factor(mfw_clean$author)

# 3. TRAIN/TEST SPLIT
set.seed(818)  # можно поменять seed для разных реплик

df <- data.frame(X, author = y)

split <- initial_split(df, prop = 0.8, strata = author)

train_data <- training(split)
test_data <- testing(split)

# 4. SCALING
train_X <- train_data |> select(-author)
test_X <- test_data |> select(-author)

train_X_scaled <- scale(train_X)
scale_center <- attr(train_X_scaled, "scaled:center")
scale_scale <- attr(train_X_scaled, "scaled:scale")

test_X_scaled <- scale(test_X, center = scale_center, scale = scale_scale)
train_data2 <- data.frame(train_X_scaled, author = train_data$author)
test_data2 <- data.frame(test_X_scaled, author = test_data$author)

# 5. SVM МОДЕЛЬ - LINEAR
set.seed(818)

svm_model <- svm(author ~ ., data = train_data2, kernel = "linear", probability = TRUE, cost = 1)


# !! SVM RADIAL !!!
set.seed(818)

n_features <- 500

svm_model <- svm(
  author ~ .,
  data = train_data2,
  kernel = "radial",        # RBF/радиальное ядро
  probability = TRUE,       # для predict_proba
  cost = 1,                 # регуляризация
  gamma = 1 / n_features    # гамма = 1 / число признаков
)


# 6. ОЦЕНКА НА ТЕСТЕ
pred_class <- predict(svm_model, test_data2)
pred_prob <- attr(predict(svm_model, test_data2, probability = TRUE), "probabilities")

accuracy <- mean(pred_class == test_data2$author)
df_results <- data.frame(truth = test_data2$author, pred = pred_class)
f1_macro <- f_meas(df_results, truth, pred, estimator = "macro")

cat("\n=== C2_300 РЕЗУЛЬТАТЫ ===\n")
cat("Accuracy:", round(accuracy, 4), "\n")
cat("F1 (macro):", round(f1_macro$.estimate, 4), "\n")


# Confusion matrix
cat("\nConfusion matrix:\n")
conf_matrix <- table(Predicted = pred_class, Actual = test_data2$author)
print(conf_matrix)

# Выгрузка
cm <- table(Predicted = pred_class, Actual = test_data2$author)
cat(capture.output(write.table(cm, sep = "\t", col.names = NA)), sep = "\n")


# 7. ПОДГОТОВКА DUBIA
dubia_X <- dubia_matrix |> select(-chunk_id, -author)

# Добавляем недостающие колонки (если есть)
missing_cols <- setdiff(colnames(train_X), colnames(dubia_X))

if (length(missing_cols) > 0) {
  for (col in missing_cols) {
    dubia_X[[col]] <- 0
  }
}

dubia_X <- dubia_X |> select(all_of(colnames(train_X)))
dubia_X_scaled <- scale(dubia_X, center = scale_center, scale = scale_scale)

# 8. ПРЕДСКАЗАНИЕ DUBIA
dubia_prob <- attr(predict(svm_model, dubia_X_scaled, probability = TRUE), "probabilities")

# ==============================

#  9. Определяем функцию DIAGNOSE_TEXT

# =============================

# Полная версия функции
diagnose_text <- function(prob_row, 
                          chunk_id_value,      # id текста
                          true_author_value = NULL,  # можно передать, если знаем
                          threshold = 0.5) {
  
  sorted <- sort(prob_row, decreasing = TRUE)
  top1 <- sorted[1]
  top2 <- sorted[2]
  ratio <- top1 / top2
  
  # Определяем статус
  if (top1 < threshold) {
    status_tag <- "COLLABORATIVE / UNKNOWN"
    interpretation <- "Низкая уверенность — возможна редактура или чужой автор"
  } else if (ratio < 3) {
    status_tag <- "MIXED"
    interpretation <- "Топ-2 автора слишком близки — следы чужого влияния"
  } else {
    status_tag <- "CLEAN"
    interpretation <- "Стилистически чистый текст"
  }
  
  # Собираем результат
  result <- list(
    chunk_id = chunk_id_value,
    status = status_tag,
    interpretation = interpretation,
    top1_author = names(top1),
    top1_prob = round(top1 * 100, 1),
    top2_author = names(top2),
    top2_prob = round(top2 * 100, 1),
    ratio = round(ratio, 2),
    all_probs = round(prob_row * 100, 1)
  )
  
  # Если передан настоящий автор — добавим сравнение
  if (!is.null(true_author_value)) {
    result$true_author <- true_author_value
    result$correct <- (names(top1) == true_author_value)
    if (!result$correct && top1 >= threshold) {
      result$warning <- "⚠️ Модель уверена, но ошиблась — возможно, спорный текст"
    }
  }
  
  return(result)
}


# Применяем диагностику
dubia_results <- list()
for (i in 1:nrow(dubia_matrix)) {
  dubia_results[[i]] <- diagnose_text(
    prob_row = dubia_prob[i, ],
    chunk_id_value = dubia_matrix$chunk_id[i],
    true_author_value = "dubia",  # указываем, что это неизвестный текст
    threshold = 0.5
  )
}

# Создаем удобную таблицу результатов
dubia_table <- data.frame(
  chunk_id = character(),
  true_author = character(),
  predicted = character(),
  plehanov = numeric(),
  parvus = numeric(),
  ortodox = numeric(),
  zasulich = numeric(),
  martov = numeric(),
  krupskaya = numeric(),
  lenin = numeric(),
  trotsky = numeric(),
  status = character(),
  top1_prob = numeric(),
  top2_author = character(),
  top2_prob = numeric(),
  ratio = numeric()
)

for (i in 1:length(dubia_results)) {
  res <- dubia_results[[i]]
  
  # Извлекаем вероятности для всех авторов
  probs <- res$all_probs
  names(probs) <- gsub("\\.", " ", names(probs))  # восстанавливаем имена
  
  # Создаем строку таблицы
  row_data <- data.frame(
    chunk_id = res$chunk_id,
    true_author = "dubia",
    predicted = res$top1_author,
    plehanov = ifelse("plehanov" %in% names(probs), probs["plehanov"], NA),
    parvus = ifelse("parvus" %in% names(probs), probs["parvus"], NA),
    ortodox = ifelse("ortodox" %in% names(probs), probs["ortodox"], NA),
    zasulich = ifelse("zasulich" %in% names(probs), probs["zasulich"], NA),
    martov = ifelse("martov" %in% names(probs), probs["martov"], NA),
    krupskaya = ifelse("krupskaya" %in% names(probs), probs["krupskaya"], NA),
    lenin = ifelse("lenin" %in% names(probs), probs["lenin"], NA),
    trotsky = ifelse("trotsky" %in% names(probs), probs["trotsky"], NA),
    status = res$status,
    top1_prob = res$top1_prob,
    top2_author = res$top2_author,
    top2_prob = res$top2_prob,
    ratio = res$ratio
  )
  
  dubia_table <- rbind(dubia_table, row_data)
}

# Просматриваем результат
print(dubia_table)


# Полная таблица результатов (для копирования в Google Sheets)
cat("\n=== DUBIA RESULTS (скопируйте таблицу ниже) ===\n\n")
cat(capture.output(write.table(dubia_table, sep = "\t", row.names = FALSE)), sep = "\n")


# Сохраняем в CSV
write_csv(dubia_table, "/Users/anastasiabogdanova/R_directory/iskra-project/results/dubia_predictions_SVM_76.csv")
cat("\nРезультаты сохранены в: /Users/anastasiabogdanova/R_directory/iskra-project/results/dubia_predictions_SVM_76.csv\n")



# Краткий вывод
# Сводка по статусам чанков
cat("\n=== СВОДКА ПО DUBIA ===\n")
table(dubia_table$status)

# Какие чанки получили какой статус
cat("\n=== ДЕТАЛИ ПО ЧАНКАМ ===\n")
cm <- dubia_table |> 
  select(chunk_id, predicted, top1_prob, status) |> 
  print()

#выгрузка
cat(capture.output(write.table(cm, sep = "\t", row.names = FALSE)), sep = "\n")




# ==============================

#  УСРЕДНЯЕМ РЕЗУЛЬТАТЫ ПО 6 МОДЕЛЯМ C1, C2, C3 * MFW (300/500)

# =============================

library(tidyverse)

# 1. ЗАГРУЗКА ВСЕХ ТАБЛИЦ

results_dir <- "/Users/anastasiabogdanova/R_directory/iskra-project/results/"

C1_300 <- read_csv(paste0(results_dir, "dubia_predictions_SVM_71.csv"))
C2_300 <- read_csv(paste0(results_dir, "dubia_predictions_SVM_72.csv"))
C3_300 <- read_csv(paste0(results_dir, "dubia_predictions_SVM_73.csv"))
C1_500 <- read_csv(paste0(results_dir, "dubia_predictions_SVM_74.csv"))
C2_500 <- read_csv(paste0(results_dir, "dubia_predictions_SVM_75.csv"))
C3_500 <- read_csv(paste0(results_dir, "dubia_predictions_SVM_76.csv"))


# Добавляем колонку с идентификатором модели
C1_300 <- C1_300 |>  mutate(model = "C1_300")
C2_300 <- C2_300 |>  mutate(model = "C2_300")
C3_300 <- C3_300 |>  mutate(model = "C3_300")
C1_500 <- C1_500 |>  mutate(model = "C1_500")
C2_500 <- C2_500 |>  mutate(model = "C2_500")
C3_500 <- C3_500 |>  mutate(model = "C3_500")


# Объединяем все
all_predictions <- bind_rows(C1_300, C2_300, C3_300, C1_500, C2_500, C3_500)


# ============================================
# 2. АГРЕГАЦИЯ ПО ЧАНКУ (среднее по 6 моделям)
# ============================================

chunk_summary <- all_predictions |> 
  group_by(chunk_id) |> 
  summarise(
    # Частота предсказаний (какой автор победил в скольких моделях)
    predictions_list = list(predicted),
    n_predictions = n(),
    
    # Самый частый предсказанный автор (консенсус)
    consensus_author = names(sort(table(predicted), decreasing = TRUE))[1],
    consensus_strength = max(sort(table(predicted), decreasing = TRUE)) / n(),
    
    # Средняя уверенность
    mean_top1_prob = mean(top1_prob, na.rm = TRUE),
    sd_top1_prob = sd(top1_prob, na.rm = TRUE),
    
    # Статус (наиболее частый)
    consensus_status = names(sort(table(status), decreasing = TRUE))[1],
    
    # Все предсказания (для детального анализа)
    predictions_detail = paste(predicted, collapse = " | ")
  ) |> 
  arrange(chunk_id)

print(chunk_summary)



# ============================================
# 3. АГРЕГАЦИЯ ПО MFW (300 vs 500)
# ============================================

mfw_summary <- all_predictions |> 
  mutate(mfw_type = ifelse(grepl("300", model), "MFW300", "MFW500")) |> 
  group_by(chunk_id, mfw_type) |> 
  summarise(
    consensus_author = names(sort(table(predicted), decreasing = TRUE))[1],
    consensus_strength = max(sort(table(predicted), decreasing = TRUE)) / n(),
    mean_confidence = mean(top1_prob),
    .groups = "drop"
  ) |> 
  pivot_wider(id_cols = chunk_id, 
              names_from = mfw_type, 
              values_from = c(consensus_author, consensus_strength, mean_confidence))

print(mfw_summary)





# ============================================
# 4. АГРЕГАЦИЯ ПО РЕПЛИКАМ (C1, C2, C3) внутри каждого MFW
# ============================================

replica_summary <- all_predictions |> 
  mutate(
    mfw_type = ifelse(grepl("300", model), "MFW300", "MFW500"),
    replica = case_when(
      grepl("C1", model) ~ "C1",
      grepl("C2", model) ~ "C2",
      grepl("C3", model) ~ "C3"
    )
  ) |> 
  group_by(chunk_id, mfw_type, replica) |> 
  summarise(
    predicted_author = predicted[1],
    confidence = top1_prob[1],
    status = status[1],
    .groups = "drop"
  ) |> 
  pivot_wider(id_cols = chunk_id,
              names_from = c(mfw_type, replica),
              values_from = c(predicted_author, confidence, status))

print(replica_summary)




# ============================================
# 5. ИТОГОВАЯ ТАБЛИЦА (для диссертации)
# ============================================

final_table <- chunk_summary |> 
  select(chunk_id, consensus_author, consensus_strength, mean_top1_prob, consensus_status) |> 
  left_join(mfw_summary, by = "chunk_id") |> 
  left_join(replica_summary, by = "chunk_id")

print(final_table)

# Сохраняем
write_csv(final_table, paste0(results_dir, "dubia_aggregated_results.csv"))
write_csv(chunk_summary, paste0(results_dir, "dubia_chunk_consensus.csv"))



#==============

# Создаем интерпретируемую таблицу
interpretation <- final_table %>%
  select(chunk_id, consensus_author, consensus_strength, mean_top1_prob) %>%
  mutate(
    reliability = case_when(
      consensus_strength >= 0.83 ~ "Высокая (5-6/6 моделей)",  # 5 или 6 из 6
      consensus_strength >= 0.67 ~ "Средняя (4/6 моделей)",
      consensus_strength >= 0.5  ~ "Низкая (3/6 моделей)",
      TRUE                       ~ "Нет консенсуса"
    ),
    recommendation = case_when(
      consensus_strength >= 0.67 ~ paste("Атрибутировать", consensus_author),
      consensus_strength >= 0.5  ~ "Требуется экспертная проверка",
      TRUE                       ~ "Недостаточно данных для атрибуции"
    )
  )

print(interpretation)

# Сохраняем
write_csv(interpretation, "/Users/anastasiabogdanova/R_directory/iskra-project/results/dubia_interpretation.csv")





# ============================================
# МЕТОД ГОЛОСОВАНИЯ ДЛЯ КАЖДОГО ЧАНКА
# ============================================


# Более простой способ без сложных вычислений внутри summarise
voting_results <- all_predictions |>
  group_by(chunk_id, predicted) |>
  summarise(
    votes = n(),
    mean_confidence = mean(top1_prob, na.rm = TRUE),
    .groups = "drop"
  ) |>
  group_by(chunk_id) |>
  mutate(
    total_votes = sum(votes),
    agreement_ratio = votes / 6  # 6 моделей всего
  ) |>
  arrange(chunk_id, desc(votes)) |>
  slice(1) |>
  select(
    chunk_id,
    winner = predicted,
    winner_votes = votes,
    agreement_ratio,
    mean_winner_confidence = mean_confidence
  )

# Смотрим результат
print(voting_results)

# Сохраняем
write_csv(voting_results, "/Users/anastasiabogdanova/R_directory/iskra-project/results/dubia_voting_results.csv")

# Краткая сводка
cat("\n=== СВОДКА ПО ГОЛОСОВАНИЮ ===\n")
voting_results |>
  group_by(winner) |>
  summarise(
    n_chunks = n(),
    mean_agreement = mean(agreement_ratio),
    mean_confidence = mean(mean_winner_confidence)
  ) |>
  arrange(desc(n_chunks)) |>
  print()



# Корреляция между согласием и средней уверенностью
cor(voting_results$agreement_ratio, voting_results$mean_winner_confidence)
# Ожидается положительная корреляция

# График
library(ggplot2)

voting_results |>
  ggplot(aes(x = agreement_ratio, y = mean_winner_confidence, color = winner)) +
  geom_point(size = 3) +
  geom_text(aes(label = substr(chunk_id, 1, 30)), hjust = -0.1, size = 2) +
  labs(title = "Согласие моделей vs Уверенность предсказания",
       x = "Доля моделей, согласных с победителем", 
       y = "Средняя уверенность победителя (%)") +
  theme_minimal() +
  scale_color_brewer(palette = "Set2")



# Финальная интерпретация для каждого текста (по первому чанку)
text_summary <- voting_results |>
  mutate(
    text_name = str_extract(chunk_id, "^[^_]+_[^_]+_[^_]+"),
    chunk_num = str_extract(chunk_id, "chunk[0-9]+$")
  ) |>
  group_by(text_name) |>
  summarise(
    consensus_author = names(sort(table(winner), decreasing = TRUE))[1],
    n_chunks = n(),
    agreement = mean(agreement_ratio),
    confidence = mean(mean_winner_confidence),
    status = case_when(
      mean(agreement_ratio) >= 0.833 ~ "Высокая надежность",
      mean(agreement_ratio) >= 0.667 ~ "Средняя надежность",
      TRUE ~ "Требуется проверка"
    )
  )

print(text_summary)


# Сохраняем
write_csv(text_summary, "/Users/anastasiabogdanova/R_directory/iskra-project/results/dubia_voting_results_text_summary.csv")


