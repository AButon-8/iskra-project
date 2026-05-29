# ==========
# С MFW НОРМАЛИЗАЦИЕЙ: считаем MFW, строим матрицы частотностей, сохраняем
# ==========

library(tidyverse)
library(tidytext)

# читаем
df <- read_csv("/iskra-project/data/texts_with_metadata_20260401.csv")

# токенизация
tokens <- df |>
  unnest_tokens(word, text)


# функция для создания MFW матриц с нормализацией
get_mfw_matrix <- function(n_words, normalize = TRUE) {
  
  # Получаем список самых частотных слов
  mfw <- freq |>
    slice_head(n = n_words) |>
    pull(word)
  
  # Создаем матрицу с сырыми частотами
  matrix_raw <- tokens |>
    filter(word %in% mfw) |>
    count(file_name, word) |>
    pivot_wider(names_from = word,
                values_from = n,
                values_fill = 0)
  
  # Добавляем метаданные и информацию о длине
  matrix_with_meta <- matrix_raw |>
    left_join(df |> 
                select(file_name, author_folder, n_words), 
              by = "file_name")
  
  # Если нужна нормализация
  if (normalize) {
    # Определяем колонки с частотными признаками (все, кроме служебных)
    feature_cols <- setdiff(
      names(matrix_with_meta), 
      c("file_name", "author_folder", "n_words")
    )
    
    # Нормализуем: частота на 1000 слов
    matrix_result <- matrix_with_meta |>
      mutate(across(all_of(feature_cols), 
                    ~ . / n_words * 1000)) |>
      select(-n_words)  # удаляем временную колонку
    
  } else {
    # Без нормализации просто удаляем n_words
    matrix_result <- matrix_with_meta |>
      select(-n_words)
  }
  
  return(matrix_result)
}


# Создаем матрицу с нормализацией (по умолчанию)
mfw_100_norm <- get_mfw_matrix(100, normalize = TRUE)

# Создаем матрицу без нормализации (если нужно сравнить)
mfw_100_raw <- get_mfw_matrix(100, normalize = FALSE)

# Для других размеров
mfw_300_norm <- get_mfw_matrix(300, normalize = TRUE)
mfw_500_norm <- get_mfw_matrix(500, normalize = TRUE)
mfw_1000_norm <- get_mfw_matrix(1000, normalize = TRUE)


# Сохраняем
write_csv(mfw_100_norm, "/iskra-project/features/exploratory/mfw_100_norm.csv")
write_csv(mfw_300_norm, "/iskra-project/features/exploratory/mfw_300_norm.csv")
write_csv(mfw_500_norm, "/iskra-project/features/exploratory/mfw_500_norm.csv")
write_csv(mfw_1000_norm, "/iskra-project/features/exploratory/mfw_1000_norm.csv")



# ==========
# char_3/4gram С НОРМАЛИЗАЦИЕЙ
# ==========

library(tidyverse)
library(tidytext)
library(stringi)

# Загружаем данные
df <- read_csv("/iskra-project/data/texts_with_metadata_20260401.csv")


# функция для создания матрицы символьных n-грамм с нормализацией
get_char_ngram_matrix <- function(df, n = 3, 
                                  min_occurrence = 3,
                                  max_features = NULL,
                                  normalize = TRUE) {
  
  # Шаг 1: Создаем n-граммы символов для каждого текста
  char_ngrams <- df |>
    select(file_name, text, n_chars) |>  # добавляем n_chars для нормализации
    mutate(
      # Очищаем текст: приводим к нижнему регистру, убираем пунктуацию
      text_clean = str_to_lower(text),
      text_clean = str_replace_all(text_clean, "[[:punct:]]", " "),
      text_clean = str_replace_all(text_clean, "\\s+", " "),
      text_clean = str_trim(text_clean),
      
      # Создаем n-граммы символов (скользящее окно)
      char_ngrams = map(text_clean, function(x) {
        chars <- str_split_1(x, "")
        if (length(chars) >= n) {
          # Создаем n-граммы
          ngrams <- sapply(1:(length(chars) - n + 1), function(i) {
            paste(chars[i:(i + n - 1)], collapse = "")
          })
          return(ngrams)
        } else {
          return(character(0))
        }
      })
    ) |>
    unnest(char_ngrams) |>
    # Убираем n-граммы, содержащие только пробелы или начинающиеся/заканчивающиеся пробелами
    filter(!str_detect(char_ngrams, "^\\s+$"),
           !str_detect(char_ngrams, "^\\s"),
           !str_detect(char_ngrams, "\\s$")) |>
    count(file_name, char_ngrams, n_chars)  # сохраняем n_chars для нормализации
  
  # Шаг 2: Фильтруем редкие n-граммы
  char_ngrams_filtered <- char_ngrams |>
    group_by(char_ngrams) |>
    filter(sum(n) >= min_occurrence) |>
    ungroup()
  
  # Шаг 3: Если нужно, добавляем нормализацию
  if (normalize) {
    # Нормализуем: частота на 1000 символов
    char_ngrams_filtered <- char_ngrams_filtered |>
      mutate(n_normalized = n / n_chars * 1000)
    
    value_col <- "n_normalized"
  } else {
    # Без нормализации используем сырые частоты
    value_col <- "n"
  }
  
  # Шаг 4: Если указан max_features, выбираем топ-признаки
  if (!is.null(max_features)) {
    # Определяем важность признаков по сумме нормализованных/сырых частот
    top_ngrams <- char_ngrams_filtered |>
      group_by(char_ngrams) |>
      summarise(importance = sum(!!sym(value_col))) |>
      arrange(desc(importance)) |>
      slice_head(n = max_features) |>
      pull(char_ngrams)
    
    char_ngrams_filtered <- char_ngrams_filtered |>
      filter(char_ngrams %in% top_ngrams)
  }
  
  # Шаг 5: Создаем широкую матрицу
  matrix_result <- char_ngrams_filtered |>
    select(file_name, char_ngrams, value = all_of(value_col)) |>
    pivot_wider(
      names_from = char_ngrams,
      values_from = value,
      values_fill = 0
    ) |>
    left_join(df |> select(file_name, author_folder), 
              by = "file_name")
  
  # Выводим информацию о созданной матрице
  cat("Создана матрица", n, "-gram:\n")
  cat("- Признаков:", ncol(matrix_result) - 2, "\n")
  cat("- Фильтр min_occurrence =", min_occurrence, "\n")
  if (normalize) {
    cat("- Нормализация: частота на 1000 символов\n")
  } else {
    cat("- Нормализация: нет (сырые частоты)\n")
  }
  if (!is.null(max_features)) {
    cat("- Ограничение признаков: топ-", max_features, "\n")
  }
  cat("- Размер матрицы:", nrow(matrix_result), "текстов x", 
      ncol(matrix_result) - 2, "признаков\n\n")
  
  return(matrix_result)
}


# Char 3-gram с нормализацией и ограничением признаков (топ-500)
char_3gram_norm <- get_char_ngram_matrix(
  df,
  n = 3,
  min_occurrence = 5,
  max_features = 500,
  normalize = TRUE
)

# Char 4-gram с нормализацией и ограничением признаков (топ-500)
char_4gram_norm <- get_char_ngram_matrix(
  df,
  n = 4,
  min_occurrence = 5,       # более строгий порог для 4-грамм
  max_features = 500,       # ограничиваем до 500 лучших
  normalize = TRUE
)


# Сохраняем в CSV
write_csv(char_3gram_norm, 
          "/iskra-project/features/exploratory//char_3gram_norm.csv")
write_csv(char_4gram_norm, 
          "/iskra-project/features/exploratory/char_4gram_norm.csv")

