library(tidyverse)
library(tidytext)
library(e1071)
library(caret)
library(yardstick)
library(ggplot2)


# Примеры имен файлов
"mfw_C.C1_300.csv"
"mfw_C.C1_500.csv"
"dubia_C_mfw_300.csv"
"dubia_C_mfw_500.csv"
"mfw_300_C_master.txt" # фиксированный словарь 300
"mfw_500_C_master.txt" # фиксированный словарь 500

# Загружаем C1_300 для получения списка всех нужных колонок
C1_300 <- read_csv("/Users/anastasiabogdanova/R_directory/iskra-project/features/mfw_C.C1_300.csv")
needed_cols <- colnames(C1_300)  # все колонки, включая chunk_id и author

# Проверяем структуру C1_300
glimpse(C1_300)
needed_cols
names(C1_300)
dim(C1_300)
head(C1_300, 2)

# Загружаем фиксированные словари (из мастер-корпуса C)
mfw_300 <- read_lines(paste0(features_dir, "mfw_300_C_master.txt"))
mfw_500 <- read_lines(paste0(features_dir, "mfw_500_C_master.txt"))

# Загружаем dubia
dubia <- read_csv("/Users/anastasiabogdanova/R_directory/iskra-project/data/processed/corpus_dubia.csv")

# Проверяем, что все колонки есть
cat("Колонки в dubia:", paste(colnames(dubia), collapse=", "), "\n")




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

# # C.C1 с MFW300 == CF_SVM_MFW300_71

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
# 3. SCALING (для SVM с радиальным ядром)
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




# ===============================================

# ============================================================
# СРАВНЕНИЕ LINEAR vs RADIAL KERNEL НА МАСТЕР-КОРПУСЕ
# Кросс-валидация: 5 фолдов, стратифицированная
# Метрики: Log Loss + Macro AUC
# Данные: 6 датафреймов (C1-3, MFW300/500)
# ============================================================

library(tidymodels)
library(kernlab)
library(ggplot2)
library(patchwork)
library(yardstick)
library(purrr)
library(dplyr)
library(tidyr)
library(readr)

# ------------------------------------------------------------
# 1. НАСТРОЙКИ
# ------------------------------------------------------------
path_to_data <- "/Users/anastasiabogdanova/R_directory/iskra-project/features/"

# Список всех файлов
files <- c(
  "mfw_C.C1_300.csv", "mfw_C.C1_500.csv",
  "mfw_C.C2_300.csv", "mfw_C.C2_500.csv",
  "mfw_C.C3_300.csv", "mfw_C.C3_500.csv"
)

# Функция для извлечения имени датасета
parse_filename <- function(filename) {
  # "mfw_C.C1_300.csv" -> "C1_300"
  gsub("mfw_C\\.|\\.csv", "", filename)
}

# ------------------------------------------------------------
# 2. ФУНКЦИЯ ДЛЯ КРОСС-ВАЛИДАЦИИ С ЗАДАННЫМ ЯДРОМ
# ------------------------------------------------------------
run_cv_with_kernel <- function(file_path, dataset_name, kernel_type = "linear") {
  
  cat("\n🔹 Обработка:", dataset_name, "| Ядро:", kernel_type, "\n")
  
  # Загрузка данных
  data <- read_csv(file_path, show_col_types = FALSE)
  
  # Проверка наличия колонки author
  if(!"author" %in% names(data)) {
    stop(paste("В файле", dataset_name, "нет колонки 'author'"))
  }
  
  # Удаляем chunk_id если есть
  if("chunk_id" %in% names(data)) {
    data <- data %>% select(-chunk_id)
  }
  
  # Определяем количество признаков
  n_features <- ncol(data) - 1
  cat("   Признаков:", n_features, "\n")
  
  # Создаем спецификацию SVM в зависимости от ядра
  if(kernel_type == "linear") {
    svm_spec <- svm_linear(cost = 1) %>%
      set_engine("kernlab", probability = TRUE) %>%
      set_mode("classification")
    cat("   gamma: не используется\n")
  } else if(kernel_type == "radial") {
    svm_spec <- svm_rbf(cost = 1, rbf_sigma = 1/n_features) %>%
      set_engine("kernlab", probability = TRUE) %>%
      set_mode("classification")
    cat("   gamma =", 1/n_features, "\n")
  } else {
    stop("Неизвестный тип ядра")
  }
  
  # Рецепт
  recipe_svm <- recipe(author ~ ., data = data)
  
  # Создаем 5 стратифицированных фолдов
  set.seed(818)
  folds <- vfold_cv(data, v = 5, strata = author)
  
  # Воркфлоу
  svm_wflow <- workflow() %>%
    add_recipe(recipe_svm) %>%
    add_model(svm_spec)
  
  # Запускаем кросс-валидацию
  cv_results <- fit_resamples(
    svm_wflow,
    resamples = folds,
    metrics = metric_set(mn_log_loss, roc_auc)
  )
  
  # Извлекаем результаты
  results_summary <- cv_results %>%
    collect_metrics(summarize = TRUE) %>%
    mutate(
      dataset = dataset_name,
      kernel = kernel_type,
      n_features = n_features
    )
  
  # Сырые результаты для графиков
  results_raw <- cv_results %>%
    collect_metrics(summarize = FALSE) %>%
    mutate(
      dataset = dataset_name,
      kernel = kernel_type
    )
  
  return(list(
    summary = results_summary,
    raw = results_raw,
    cv_object = cv_results
  ))
}

# ------------------------------------------------------------
# 3. ЗАПУСК ДЛЯ ВСЕХ ДАТАСЕТОВ И ОБОИХ ЯДЕР
# ------------------------------------------------------------
all_results <- list()

for(file in files) {
  dataset_name <- parse_filename(file)
  file_path <- file.path(path_to_data, file)
  
  if(!file.exists(file_path)) {
    warning("Файл не найден: ", file_path)
    next
  }
  
  # Запускаем для linear ядра
  results_linear <- run_cv_with_kernel(file_path, dataset_name, "linear")
  all_results[[paste0(dataset_name, "_linear")]] <- results_linear
  
  # Запускаем для radial ядра
  results_radial <- run_cv_with_kernel(file_path, dataset_name, "radial")
  all_results[[paste0(dataset_name, "_radial")]] <- results_radial
}

# ------------------------------------------------------------
# 4. СВОДНАЯ ТАБЛИЦА РЕЗУЛЬТАТОВ
# ------------------------------------------------------------
summary_table <- map_dfr(all_results, ~ .x$summary) %>%
  select(dataset, kernel, n_features, .metric, mean, std_err) %>%
  mutate(
    result_display = paste0(round(mean, 4), " ± ", round(std_err, 4)),
    kernel_label = ifelse(kernel == "linear", "Linear", "Radial")
  ) %>%
  arrange(.metric, dataset, kernel)

# Выводим таблицу
cat("\n\n========================================\n")
cat("РЕЗУЛЬТАТЫ КРОСС-ВАЛИДАЦИИ (5 ФОЛДОВ)\n")
cat("СРАВНЕНИЕ LINEAR vs RADIAL KERNEL\n")
cat("========================================\n\n")

# Log Loss
cat("📉 LOG LOSS (меньше → лучше)\n")
cat("----------------------------------------\n")
summary_table %>%
  filter(.metric == "mn_log_loss") %>%
  select(dataset, kernel_label, result_display) %>%
  print(n = 20)

cat("\n📈 MACRO AUC (больше → лучше, 0.5 = случайно)\n")
cat("----------------------------------------\n")
summary_table %>%
  filter(.metric == "roc_auc") %>%
  select(dataset, kernel_label, result_display) %>%
  print(n = 20)

# ------------------------------------------------------------
# 5. СРАВНЕНИЕ ПО РЕПЛИКАМ (УСРЕДНЕНИЕ)
# ------------------------------------------------------------
comparison_by_replica <- summary_table %>%
  separate(dataset, into = c("replica", "mfw"), sep = "_", remove = FALSE) %>%
  group_by(replica, mfw, kernel_label, .metric) %>%
  summarise(
    mean_metric = mean(mean),
    se_metric = mean(std_err),
    .groups = "drop"
  ) %>%
  arrange(.metric, mfw, replica, kernel_label)

cat("\n\n========================================\n")
cat("СРАВНЕНИЕ ПО РЕПЛИКАМ (C1, C2, C3)\n")
cat("========================================\n\n")

# Log Loss по репликам
cat("📉 LOG LOSS\n")
comparison_by_replica %>%
  filter(.metric == "mn_log_loss") %>%
  select(replica, mfw, kernel_label, mean_metric) %>%
  pivot_wider(names_from = kernel_label, values_from = mean_metric) %>%
  mutate(Difference = Linear - Radial) %>%
  print()

# Macro AUC по репликам
cat("\n📈 MACRO AUC\n")
comparison_by_replica %>%
  filter(.metric == "roc_auc") %>%
  select(replica, mfw, kernel_label, mean_metric) %>%
  pivot_wider(names_from = kernel_label, values_from = mean_metric) %>%
  mutate(Difference = Linear - Radial) %>%
  print()

# ------------------------------------------------------------
# 6. ВИЗУАЛИЗАЦИЯ
# ------------------------------------------------------------

# ============================================================
# ПРОСТАЯ И НАДЕЖНАЯ ВЕРСИЯ ГРАФИКОВ (БЕЗ КОНФЛИКТОВ)
# ============================================================

# Функция для создания графика сравнения ядер (упрощенная)
plot_kernel_comparison_simple <- function(results_list, metric_name, metric_title) {
  
  # Агрегируем данные для отображения средних и стандартных ошибок
  plot_data <- map_dfr(results_list, ~ .x$raw) %>%
    filter(.metric == metric_name) %>%
    mutate(
      mfw = str_extract(dataset, "300|500"),
      kernel_label = ifelse(kernel == "linear", "Linear", "Radial"),
      replica = str_extract(dataset, "^C[1-3]")
    ) %>%
    group_by(dataset, mfw, kernel_label, replica) %>%
    summarise(
      mean_val = mean(.estimate),
      se_val = sd(.estimate) / sqrt(n()),
      .groups = "drop"
    )
  
  # Настройка ylim в зависимости от метрики
  y_limits <- if(metric_name == "mn_log_loss") {
    c(NA, NA)  # автоматически для Log Loss
  } else {
    c(0.5, 1.0)  # для AUC от 0.5 до 1
  }
  
  # Создаем график
  p <- ggplot(plot_data, aes(x = dataset, y = mean_val, 
                             color = kernel_label, 
                             group = kernel_label)) +
    # Точки для средних значений
    geom_point(size = 3, position = position_dodge(width = 0.3)) +
    # Планки ошибок (±SE)
    geom_errorbar(aes(ymin = mean_val - se_val, ymax = mean_val + se_val),
                  width = 0.2, position = position_dodge(width = 0.3)) +
    # Горизонтальная линия для AUC (уровень случайности)
    {if(metric_name == "roc_auc") geom_hline(yintercept = 0.5, linetype = "dashed", alpha = 0.5, color = "gray50")} +
    labs(
      title = metric_title,
      subtitle = "Linear (синий) vs Radial (красный) | Точки = среднее по 5 фолдам, планки = ±SE",
      x = "Датафрейм", 
      y = metric_title, 
      color = "Ядро"
    ) +
    theme_light() +
    theme(
      axis.text.x = element_text(angle = 45, hjust = 1, size = 9),
      legend.position = "top",
      strip.background = element_rect(fill = "steelblue"),
      strip.text = element_text(color = "white", face = "bold")
    ) +
    coord_cartesian(ylim = y_limits) +
    facet_wrap(~ mfw, scales = "free_x", ncol = 2)
  
  return(p)
}

# Строим графики
p_logloss <- plot_kernel_comparison_simple(all_results, "mn_log_loss", "Log Loss (меньше → лучше)")
p_auc <- plot_kernel_comparison_simple(all_results, "roc_auc", "Macro AUC (больше → лучше, 0.5 = случайно)")

# Показываем
print(p_logloss)
print(p_auc)

# Сохраняем (если path_to_data определена)
if(exists("path_to_data")) {
  ggsave(file.path(path_to_data, "CV_comparison_logloss.png"), p_logloss, width = 12, height = 6)
  ggsave(file.path(path_to_data, "CV_comparison_auc.png"), p_auc, width = 12, height = 6)
  cat("\n✅ Графики сохранены в:", path_to_data, "\n")
} else {
  cat("\n⚠️ path_to_data не найдена. Графики не сохранены, но отображены на экране.\n")
}



# ==============================================
# ВИЗУАЛИЗАЦИЯ 2.
#

# Версия с точками для каждого фолда (как вы хотели изначально)
plot_kernel_comparison_with_folds <- function(results_list, metric_name, metric_title) {
  
  # Данные для каждого фолда
  fold_data <- map_dfr(results_list, ~ .x$raw) %>%
    filter(.metric == metric_name) %>%
    mutate(
      mfw = str_extract(dataset, "300|500"),
      kernel_label = ifelse(kernel == "linear", "Linear", "Radial"),
      fold_id = as.numeric(str_extract(id, "[0-9]+"))
    )
  
  # Агрегированные данные для средних и SE
  agg_data <- fold_data %>%
    group_by(dataset, mfw, kernel_label) %>%
    summarise(
      mean_val = mean(.estimate),
      se_val = sd(.estimate) / sqrt(n()),
      .groups = "drop"
    )
  
  y_limits <- if(metric_name == "mn_log_loss") {
    c(NA, NA)
  } else {
    c(0.5, 1.0)
  }
  
  ggplot() +
    # Точки для каждого фолда (с небольшим разбросом)
    geom_jitter(data = fold_data, 
                aes(x = dataset, y = .estimate, color = kernel_label),
                width = 0.15, height = 0, alpha = 0.3, size = 1.5) +
    # Средние значения (большие точки)
    geom_point(data = agg_data,
               aes(x = dataset, y = mean_val, color = kernel_label),
               size = 4, position = position_dodge(width = 0.3)) +
    # Планки ошибок
    geom_errorbar(data = agg_data,
                  aes(x = dataset, ymin = mean_val - se_val, 
                      ymax = mean_val + se_val, color = kernel_label),
                  width = 0.2, position = position_dodge(width = 0.3)) +
    {if(metric_name == "roc_auc") geom_hline(yintercept = 0.5, linetype = "dashed", alpha = 0.5, color = "gray50")} +
    labs(
      title = metric_title,
      subtitle = "Мелкие точки = каждый фолд (5 шт.), крупные = среднее ± SE | Синий = Linear, Красный = Radial",
      x = "Датафрейм", y = metric_title, color = "Ядро"
    ) +
    theme_light() +
    theme(
      axis.text.x = element_text(angle = 45, hjust = 1, size = 9),
      legend.position = "top"
    ) +
    coord_cartesian(ylim = y_limits) +
    facet_wrap(~ mfw, scales = "free_x", ncol = 2)
}

# Используем версию с фолдами
p_logloss_folds <- plot_kernel_comparison_with_folds(all_results, "mn_log_loss", "Log Loss (меньше → лучше)")
p_auc_folds <- plot_kernel_comparison_with_folds(all_results, "roc_auc", "Macro AUC (больше → лучше, 0.5 = случайно)")

print(p_logloss_folds)
print(p_auc_folds)




# ------------------------------------------------------------
# 7. ОБЩАЯ СТАБИЛЬНОСТЬ ПО ЯДРАМ
# ------------------------------------------------------------
overall_kernel_summary <- summary_table %>%
  group_by(kernel_label, .metric) %>%
  summarise(
    overall_mean = mean(mean),
    overall_sd = sd(mean),
    display = paste0(round(overall_mean, 4), " ± ", round(overall_sd, 4)),
    .groups = "drop"
  )

cat("\n\n========================================\n")
cat("ОБЩАЯ СТАБИЛЬНОСТЬ ПО ВСЕМ ДАТАСЕТАМ\n")
cat("========================================\n\n")
overall_kernel_summary %>%
  mutate(metric = ifelse(.metric == "mn_log_loss", "Log Loss", "Macro AUC")) %>%
  select(metric, kernel_label, display) %>%
  print()

# ------------------------------------------------------------
# 8. СОХРАНЕНИЕ РЕЗУЛЬТАТОВ
# ------------------------------------------------------------
save(all_results, summary_table, comparison_by_replica, overall_kernel_summary,
     file = file.path(path_to_data, "CV_kernel_comparison.RData"))

# Экспорт таблиц в CSV
write.csv(summary_table, 
          file.path(path_to_data, "CV_kernel_comparison_summary.csv"), 
          row.names = FALSE)

write.csv(comparison_by_replica, 
          file.path(path_to_data, "CV_kernel_comparison_by_replica.csv"), 
          row.names = FALSE)

# Сохраняем графики
ggsave(file.path(path_to_data, "CV_comparison_logloss.png"), p_logloss, width = 12, height = 6)
ggsave(file.path(path_to_data, "CV_comparison_auc.png"), p_auc, width = 12, height = 6)

cat("\n\n✅ ГОТОВО!\n")
cat("Результаты сохранены в:", path_to_data, "\n")
cat("- CV_kernel_comparison_summary.csv (все метрики)\n")
cat("- CV_kernel_comparison_by_replica.csv (по репликам)\n")
cat("- CV_comparison_logloss.png и CV_comparison_auc.png\n")
cat("- CV_kernel_comparison.RData (полные результаты)\n")



# ============================================================
# ГРАФИК В СТИЛЕ autoplot ДЛЯ СРАВНЕНИЯ ЯДЕР
# ============================================================

library(ggplot2)
library(patchwork)

# Функция для создания autoplot-стиля
make_autoplot_style <- function(results_list, metric_name, metric_title) {
  
  # Собираем данные для каждого фолда
  plot_data <- map_dfr(results_list, ~ .x$raw) %>%
    filter(.metric == metric_name) %>%
    mutate(
      mfw = str_extract(dataset, "300|500"),
      kernel_label = ifelse(kernel == "linear", "Linear", "Radial"),
      # Создаем wflow_id для подписей
      wflow_id = paste(kernel_label, dataset, sep = " | ")
    )
  
  # Вычисляем среднее и std_err для каждой группы
  summary_data <- plot_data %>%
    group_by(dataset, kernel_label, mfw) %>%
    summarise(
      mean = mean(.estimate),
      std_err = sd(.estimate) / sqrt(n()),
      .groups = "drop"
    )
  
  # Объединяем
  plot_data <- plot_data %>%
    left_join(summary_data, by = c("dataset", "kernel_label", "mfw"))
  
  # Настройка ylim
  if(metric_name == "mn_log_loss") {
    y_min <- min(plot_data$.estimate) - 0.1
    y_max <- max(plot_data$.estimate) + 0.1
  } else {
    y_min <- 0.5
    y_max <- 1.0
  }
  
  # Создаем график
  ggplot(plot_data, aes(x = dataset, y = .estimate, color = kernel_label)) +
    # Точки для каждого фолда
    geom_point(size = 2, alpha = 0.5, position = position_dodge(width = 0.3)) +
    # Точка для среднего
    stat_summary(fun = mean, geom = "point", size = 4, 
                 position = position_dodge(width = 0.3)) +
    # Планки ошибок (±2*std_err)
    stat_summary(fun.data = function(x) {
      m <- mean(x)
      se <- sd(x) / sqrt(length(x))
      data.frame(y = m, ymin = m - 2*se, ymax = m + 2*se)
    }, geom = "errorbar", width = 0.2, position = position_dodge(width = 0.3)) +
    # Текст с подписями (как в вашем примере)
    geom_text(data = summary_data,
              aes(x = dataset, y = mean - 2*std_err, 
                  label = paste(kernel_label, dataset, sep = "\n"),
                  color = kernel_label),
              angle = 90, hjust = 1.2, size = 2.5,
              position = position_dodge(width = 0.3)) +
    labs(
      title = metric_title,
      subtitle = "Точки = каждый фолд (5 шт.), крупные точки = среднее, планки = ±2×SE",
      x = "Датафрейм", y = metric_title, color = "Ядро"
    ) +
    theme_light() +
    theme(
      legend.position = "top",
      axis.text.x = element_blank(),  # убираем метки x, т.к. есть подписи
      axis.ticks.x = element_blank()
    ) +
    coord_cartesian(ylim = c(y_min, y_max)) +
    facet_wrap(~ mfw, scales = "free_x", ncol = 2)
}

# Создаем графики
p_autoplot_logloss <- make_autoplot_style(all_results, "mn_log_loss", "Log Loss (меньше → лучше)")
p_autoplot_auc <- make_autoplot_style(all_results, "roc_auc", "Macro AUC (больше → лучше)")

# Показываем
print(p_autoplot_logloss)
print(p_autoplot_auc)

# Сохраняем
if(exists("path_to_data")) {
  ggsave(file.path(path_to_data, "CV_autoplot_logloss.png"), p_autoplot_logloss, width = 12, height = 6)
  ggsave(file.path(path_to_data, "CV_autoplot_auc.png"), p_autoplot_auc, width = 12, height = 6)
  cat("✅ Графики сохранены\n")
}



# ============================================================
# КОМПАКТНЫЙ ГРАФИК ДЛЯ MACRO AUC (БЕЗ ПУСТОТ)
# ============================================================

library(ggplot2)
library(patchwork)

# Функция для создания компактного графика
plot_compact_auc <- function(results_list) {
  
  # Собираем данные
  plot_data <- map_dfr(results_list, ~ .x$raw) %>%
    filter(.metric == "roc_auc") %>%
    mutate(
      mfw = str_extract(dataset, "300|500"),
      kernel_label = ifelse(kernel == "linear", "Linear", "Radial"),
      # Сортируем датасеты для лучшего отображения
      dataset = factor(dataset, levels = c("C1_300", "C2_300", "C3_300", 
                                           "C1_500", "C2_500", "C3_500"))
    )
  
  # Агрегированные данные для средних
  summary_data <- plot_data %>%
    group_by(dataset, mfw, kernel_label) %>%
    summarise(
      mean = mean(.estimate),
      std_err = sd(.estimate) / sqrt(n()),
      .groups = "drop"
    )
  
  # Компактный график
  ggplot() +
    # Точки для каждого фолда (меньший разброс)
    geom_jitter(data = plot_data,
                aes(x = dataset, y = .estimate, color = kernel_label),
                width = 0.1, height = 0, alpha = 0.3, size = 1.5) +
    # Средние значения (крупные точки)
    geom_point(data = summary_data,
               aes(x = dataset, y = mean, color = kernel_label),
               size = 3.5, position = position_dodge(width = 0.2)) +
    # Планки ошибок (±SE)
    geom_errorbar(data = summary_data,
                  aes(x = dataset, ymin = mean - std_err, 
                      ymax = mean + std_err, color = kernel_label),
                  width = 0.1, position = position_dodge(width = 0.2)) +
    # Горизонтальная линия случайности
    geom_hline(yintercept = 0.5, linetype = "dashed", alpha = 0.4, color = "gray40") +
    labs(
      title = "Macro AUC: Linear vs Radial Kernel",
      subtitle = "Точки = каждый фолд (5 шт.), крупные = среднее ± SE | Чем выше, тем лучше",
      x = "Признаки", y = "Macro AUC", color = "Ядро"
    ) +
    scale_color_manual(values = c("Linear" = "#1f77b4", "Radial" = "#d62728")) +
    scale_y_continuous(limits = c(0.85, 0.95), breaks = seq(0.85, 0.95, 0.02)) +
    theme_light() +
    theme(
      legend.position = "top",
      axis.text.x = element_text(angle = 45, hjust = 1, size = 10, face = "bold"),
      panel.grid.minor = element_blank(),
      panel.spacing = unit(1, "lines")
    ) +
    facet_wrap(~ mfw, scales = "free_x", ncol = 2)
}

# Строим график
p_auc_compact <- plot_compact_auc(all_results)
print(p_auc_compact)

# Сохраняем
if(exists("path_to_data")) {
  ggsave(file.path(path_to_data, "CV_auc_compact.png"), p_auc_compact, width = 12, height = 7)
  cat("✅ График сохранен\n")
}




# ============================================================
# ГРАФИК MACRO AUC В ЕДИНОМ СТИЛЕ (ВАША ТИПОГРАФИКА)
# ============================================================

library(ggplot2)
library(patchwork)

plot_elegant_auc <- function(results_list) {
  
  # Собираем агрегированные данные
  summary_data <- map_dfr(results_list, ~ .x$summary) %>%
    filter(.metric == "roc_auc") %>%
    mutate(
      mfw = str_extract(dataset, "300|500"),
      kernel_label = ifelse(kernel == "linear", "Linear", "Radial"),
      replica = str_extract(dataset, "^C[1-3]"),
      # Создаем метку MFW для фасетов
      MFW_label = paste0("MFW ", mfw),
      # Для порядка на графике
      dataset = factor(dataset, levels = c("C1_300", "C2_300", "C3_300", 
                                           "C1_500", "C2_500", "C3_500"))
    )
  
  # Сырые данные для точек фолдов
  raw_data <- map_dfr(results_list, ~ .x$raw) %>%
    filter(.metric == "roc_auc") %>%
    mutate(
      mfw = str_extract(dataset, "300|500"),
      kernel_label = ifelse(kernel == "linear", "Linear", "Radial"),
      replica = str_extract(dataset, "^C[1-3]"),
      MFW_label = paste0("MFW ", mfw),
      dataset = factor(dataset, levels = c("C1_300", "C2_300", "C3_300", 
                                           "C1_500", "C2_500", "C3_500"))
    )
  
  # Цвета (ваши любимые)
  kernel_colors <- c("Linear" = "#1f77b4", "Radial" = "#d62728")
  
  ggplot() +
    # Точки для каждого фолда (полупрозрачные, маленькие)
    geom_jitter(data = raw_data,
                aes(x = dataset, y = .estimate, color = kernel_label),
                width = 0.15, height = 0, alpha = 0.4, size = 2) +
    # Средние значения (крупные точки)
    geom_point(data = summary_data,
               aes(x = dataset, y = mean, color = kernel_label, group = kernel_label),
               size = 4, position = position_dodge(width = 0.2)) +
    # Планки ошибок (±SE)
    geom_errorbar(data = summary_data,
                  aes(x = dataset, ymin = mean - std_err, 
                      ymax = mean + std_err, color = kernel_label,
                      group = kernel_label),
                  width = 0.1, size = 0.8, position = position_dodge(width = 0.2)) +
    # Горизонтальная линия (уровень случайности)
    geom_hline(yintercept = 0.5, linetype = "dashed", alpha = 0.5, 
               color = "gray50", size = 0.5) +
    # Подписи над планками (как в вашем autoplot)
    #geom_text(data = summary_data,
    #          aes(x = dataset, y = mean + 0.012, 
    #             label = paste0(round(mean, 3), "±", round(std_err, 3)),
    #             color = kernel_label),
    #         size = 3, fontface = "bold", angle = 0,
    #         position = position_dodge(width = 0.2)) +
    labs(
      title = "Кросс-валидация — Macro AUC: SVM линейное vs радиальное ядро",
      subtitle = "Macro AUC (больше → лучше) | 5 фолдов, стратификация по автору",
      x = "Реплика корпуса и размер MFW", 
      y = "Macro AUC",
      color = "Ядро"
    ) +
    scale_color_manual(values = kernel_colors) +
    scale_y_continuous(limits = c(0.85, 0.96), 
                       breaks = seq(0.85, 0.96, 0.02),
                       labels = scales::number_format(accuracy = 0.01)) +
    facet_wrap(~ MFW_label, scales = "free_x", ncol = 2) +
    theme_minimal() +
    theme(
      # Текст осей
      axis.text.x = element_text(size = 11, face = "bold", angle = 45, hjust = 1),
      axis.text.y = element_text(size = 11, face = "bold"),
      axis.title.x = element_text(face = "bold", size = 13),
      axis.title.y = element_text(face = "bold", size = 13),
      
      # Заголовки
      plot.title = element_text(face = "bold", size = 18, hjust = 0.5),
      plot.subtitle = element_text(size = 13, color = "gray40", face = "italic", hjust = 0.5),
      
      # Фасеты
      strip.text = element_text(face = "bold", size = 13),
      strip.background = element_rect(fill = "gray95", color = NA),
      
      # Легенда
      legend.position = "top",
      legend.title = element_text(size = 12, face = "bold"),
      legend.text = element_text(size = 11),
      
      # Сетка
      panel.grid.major.x = element_blank(),
      panel.grid.minor = element_blank(),
      panel.grid.major.y = element_line(color = "gray90", size = 0.3),
      
      # Отступы
      plot.margin = unit(c(1, 1, 1, 1), "cm")
    )
}

# Строим
p_auc_elegant <- plot_elegant_auc(all_results)
print(p_auc_elegant)

# Сохраняем
if(exists("path_to_data")) {
  ggsave(file.path(path_to_data, "CV_auc_elegant.png"), 
         p_auc_elegant, width = 12, height = 7, dpi = 300)
  cat("✅ График сохранен\n")
}







# ============================================================
# ГРАФИК LOG LOSS В ЕДИНОМ СТИЛЕ (ВАША ТИПОГРАФИКА)
# ============================================================


plot_elegant_logloss <- function(results_list) {
  
  summary_data <- map_dfr(results_list, ~ .x$summary) %>%
    filter(.metric == "mn_log_loss") %>%
    mutate(
      mfw = str_extract(dataset, "300|500"),
      kernel_label = ifelse(kernel == "linear", "Linear", "Radial"),
      MFW_label = paste0("MFW ", mfw),
      dataset = factor(dataset, levels = c("C1_300", "C2_300", "C3_300", 
                                           "C1_500", "C2_500", "C3_500"))
    )
  
  raw_data <- map_dfr(results_list, ~ .x$raw) %>%
    filter(.metric == "mn_log_loss") %>%
    mutate(
      mfw = str_extract(dataset, "300|500"),
      kernel_label = ifelse(kernel == "linear", "Linear", "Radial"),
      MFW_label = paste0("MFW ", mfw),
      dataset = factor(dataset, levels = c("C1_300", "C2_300", "C3_300", 
                                           "C1_500", "C2_500", "C3_500"))
    )
  
  kernel_colors <- c("Linear" = "#1f77b4", "Radial" = "#d62728")
  
  ggplot() +
    geom_jitter(data = raw_data,
                aes(x = dataset, y = .estimate, color = kernel_label),
                width = 0.15, height = 0, alpha = 0.3, size = 2) +
    geom_point(data = summary_data,
               aes(x = dataset, y = mean, color = kernel_label),
               size = 4, position = position_dodge(width = 0.2)) +
    geom_errorbar(data = summary_data,
                  aes(x = dataset, ymin = mean - std_err, 
                      ymax = mean + std_err, color = kernel_label),
                  width = 0.1, size = 0.8, position = position_dodge(width = 0.2)) +
    #geom_text(data = summary_data,
    #          aes(x = dataset, y = mean - 0.025, 
    #              label = paste0(round(mean, 3), "±", round(std_err, 3)),
    #              color = kernel_label),
    #          size = 3, fontface = "bold", angle = 0,
    #          position = position_dodge(width = 0.2)) +
    labs(
      title = "Кросс-валидация — Log Loss: SVM линейное vs радиальное ядро",
      subtitle = "Log Loss (меньше → лучше) | 5 фолдов, стратификация по автору",
      x = "Реплика корпуса и размер MFW", 
      y = "Log Loss",
      color = "Ядро"
    ) +
    scale_color_manual(values = kernel_colors) +
    scale_y_continuous(limits = c(1.2, 1.6), 
                       breaks = seq(1.2, 1.6, 0.05)) +
    facet_wrap(~ MFW_label, scales = "free_x", ncol = 2) +
    theme_minimal() +
    theme(
      axis.text.x = element_text(size = 11, face = "bold", angle = 45, hjust = 1),
      axis.text.y = element_text(size = 11, face = "bold"),
      axis.title.x = element_text(face = "bold", size = 13),
      axis.title.y = element_text(face = "bold", size = 13),
      plot.title = element_text(face = "bold", size = 18, hjust = 0.5),
      plot.subtitle = element_text(size = 13, color = "gray40", face = "italic", hjust = 0.5),
      strip.text = element_text(face = "bold", size = 13),
      strip.background = element_rect(fill = "gray95", color = NA),
      legend.position = "top",
      legend.title = element_text(size = 12, face = "bold"),
      legend.text = element_text(size = 11),
      panel.grid.major.x = element_blank(),
      panel.grid.minor = element_blank(),
      panel.grid.major.y = element_line(color = "gray90", size = 0.3),
      plot.margin = unit(c(1, 1, 1, 1), "cm")
    )
}

p_logloss_elegant <- plot_elegant_logloss(all_results)
print(p_logloss_elegant)

# Сохраняем
if(exists("path_to_data")) {
  ggsave(file.path(path_to_data, "CV_logloss_elegant.png"), 
         p_logloss_elegant, width = 12, height = 7, dpi = 300)
}




# ============================================================
# ГРАФИК ДЛЯ ПЕЧАТИ (КРУПНО, ЧЕТКО, БЕЗ ПОЛУПРОЗРАЧНОСТИ)
# ============================================================

library(ggplot2)
library(patchwork)

plot_print_auc <- function(results_list) {
  
  # Агрегированные данные
  summary_data <- map_dfr(results_list, ~ .x$summary) %>%
    filter(.metric == "roc_auc") %>%
    mutate(
      mfw = str_extract(dataset, "300|500"),
      kernel_label = ifelse(kernel == "linear", "Linear", "Radial"),
      MFW_label = paste0("MFW ", mfw),
      dataset = factor(dataset, levels = c("C1_300", "C2_300", "C3_300", 
                                           "C1_500", "C2_500", "C3_500"))
    )
  
  # Сырые данные для точек фолдов
  raw_data <- map_dfr(results_list, ~ .x$raw) %>%
    filter(.metric == "roc_auc") %>%
    mutate(
      mfw = str_extract(dataset, "300|500"),
      kernel_label = ifelse(kernel == "linear", "Linear", "Radial"),
      MFW_label = paste0("MFW ", mfw),
      dataset = factor(dataset, levels = c("C1_300", "C2_300", "C3_300", 
                                           "C1_500", "C2_500", "C3_500"))
    )
  
  # Цвета (насыщенные, для печати)
  kernel_colors <- c("Linear" = "#1f77b4", "Radial" = "#d62728")
  
  ggplot() +
    # Точки для каждого фолда (НЕ полупрозрачные, большего размера)
    geom_jitter(data = raw_data,
                aes(x = dataset, y = .estimate, color = kernel_label),
                width = 0.12, height = 0, alpha = 0.7, size = 2.5) +
    # Средние значения (крупные точки, обводка)
    geom_point(data = summary_data,
               aes(x = dataset, y = mean, color = kernel_label, group = kernel_label),
               size = 5, stroke = 0.8, position = position_dodge(width = 0.25)) +
    # Планки ошибок (жирные, четкие)
    geom_errorbar(data = summary_data,
                  aes(x = dataset, ymin = mean - std_err, 
                      ymax = mean + std_err, color = kernel_label,
                      group = kernel_label),
                  width = 0.15, linewidth = 1.2, position = position_dodge(width = 0.25)) +
    # Горизонтальная линия (уровень случайности) - жирная, темная
    geom_hline(yintercept = 0.5, linetype = "dashed", alpha = 0.8, 
               color = "black", linewidth = 0.8) +
    # Подписи над планками (крупнее, жирнее)
    #geom_text(data = summary_data,
    #          aes(x = dataset, y = mean + 0.01, 
    #              label = paste0(round(mean, 3), "±", round(std_err, 3)),
    #              color = kernel_label),
    #          size = 4.5, fontface = "bold", angle = 0,
    #          position = position_dodge(width = 0.25)) +
    labs(
      title = "Кросс-валидация — Macro AUC: SVM линейное vs радиальное ядро",
      subtitle = "Macro AUC (больше → лучше) | 5 фолдов, стратификация по автору",
      x = "",  # убираем подпись оси, чтобы не давила
      y = "Macro AUC",
      color = "Ядро"
    ) +
    scale_color_manual(values = kernel_colors) +
    scale_y_continuous(limits = c(0.84, 0.97), 
                       breaks = seq(0.84, 0.97, 0.02),
                       labels = scales::number_format(accuracy = 0.01)) +
    facet_wrap(~ MFW_label, scales = "free_x", ncol = 2) +
    theme_bw() +  # вместо minimal - четкие границы
    theme(
      # Текст осей (крупнее)
      axis.text.x = element_text(size = 13, face = "bold", angle = 0, hjust = 0.5),
      axis.text.y = element_text(size = 12, face = "bold"),
      axis.title.x = element_blank(),  # убрали
      axis.title.y = element_text(face = "bold", size = 14, margin = margin(r = 10)),
      
      # Заголовки
      plot.title = element_text(face = "bold", size = 18, hjust = 0.5, margin = margin(b = 10)),
      plot.subtitle = element_text(size = 12, color = "gray30", face = "italic", hjust = 0.5, margin = margin(b = 20)),
      
      # Фасеты (планки с MFW300/500) — сделаем заметными
      strip.text = element_text(face = "bold", size = 14, color = "white"),
      strip.background = element_rect(fill = "#333333", color = "black", linewidth = 1),
      
      # Легенда
      legend.position = "top",
      legend.title = element_text(size = 13, face = "bold"),
      legend.text = element_text(size = 12),
      legend.key.size = unit(0.8, "cm"),
      
      # Сетка (только горизонтальные линии, жирные)
      panel.grid.major.x = element_blank(),
      panel.grid.minor = element_blank(),
      panel.grid.major.y = element_line(color = "gray70", linewidth = 0.5, linetype = "solid"),
      
      # Границы графика
      panel.border = element_rect(color = "black", fill = NA, linewidth = 1),
      
      # Отступы (увеличиваем, чтобы надписи не прижимались)
      plot.margin = margin(t = 20, r = 15, b = 15, l = 15, unit = "pt"),
      
      # Фон
      panel.background = element_rect(fill = "white"),
      plot.background = element_rect(fill = "white")
    )
}

# Строим
p_print_auc <- plot_print_auc(all_results)
print(p_print_auc)

# Сохраняем в высоком разрешении для печати
if(exists("path_to_data")) {
  ggsave(file.path(path_to_data, "CV_auc_print.png"), 
         p_print_auc, width = 10, height = 6, dpi = 600, bg = "white")
  cat("✅ График сохранен в высоком разрешении (600 dpi)\n")
}


# ============================================================
# ГРАФИК ДЛЯ ПЕЧАТИ (КРУПНО, БЕЗ ПОЛУПРОЗРАЧНОСТИ) LOG LOSS
# ============================================================

plot_print_logloss <- function(results_list) {
  
  summary_data <- map_dfr(results_list, ~ .x$summary) %>%
    filter(.metric == "mn_log_loss") %>%
    mutate(
      mfw = str_extract(dataset, "300|500"),
      kernel_label = ifelse(kernel == "linear", "Linear", "Radial"),
      MFW_label = paste0("MFW ", mfw),
      dataset = factor(dataset, levels = c("C1_300", "C2_300", "C3_300", 
                                           "C1_500", "C2_500", "C3_500"))
    )
  
  raw_data <- map_dfr(results_list, ~ .x$raw) %>%
    filter(.metric == "mn_log_loss") %>%
    mutate(
      mfw = str_extract(dataset, "300|500"),
      kernel_label = ifelse(kernel == "linear", "Linear", "Radial"),
      MFW_label = paste0("MFW ", mfw),
      dataset = factor(dataset, levels = c("C1_300", "C2_300", "C3_300", 
                                           "C1_500", "C2_500", "C3_500"))
    )
  
  kernel_colors <- c("Linear" = "#1f77b4", "Radial" = "#d62728")
  
  ggplot() +
    geom_jitter(data = raw_data,
                aes(x = dataset, y = .estimate, color = kernel_label),
                width = 0.12, height = 0, alpha = 0.7, size = 2.5) +
    geom_point(data = summary_data,
               aes(x = dataset, y = mean, color = kernel_label),
               size = 5, position = position_dodge(width = 0.25)) +
    geom_errorbar(data = summary_data,
                  aes(x = dataset, ymin = mean - std_err, 
                      ymax = mean + std_err, color = kernel_label),
                  width = 0.15, linewidth = 1.2, position = position_dodge(width = 0.25)) +
   # geom_text(data = summary_data,
  #            aes(x = dataset, y = mean - 0.025, 
   #               label = paste0(round(mean, 3), "±", round(std_err, 3)),
  #                color = kernel_label),
   #           size = 4.5, fontface = "bold", angle = 0,
  #            position = position_dodge(width = 0.25)) +
    labs(
      title = "Кросс-валидация — Log Loss: SVM линейное vs радиальное ядро",
      subtitle = "Log Loss (меньше → лучше) | 5 фолдов, стратификация по автору",
      x = "",
      y = "Log Loss",
      color = "Ядро"
    ) +
    scale_color_manual(values = kernel_colors) +
    scale_y_continuous(limits = c(1.2, 1.65), 
                       breaks = seq(1.2, 1.65, 0.05)) +
    facet_wrap(~ MFW_label, scales = "free_x", ncol = 2) +
    theme_bw() +
    theme(
      axis.text.x = element_text(size = 13, face = "bold", angle = 0, hjust = 0.5),
      axis.text.y = element_text(size = 12, face = "bold"),
      axis.title.x = element_blank(),
      axis.title.y = element_text(face = "bold", size = 14, margin = margin(r = 10)),
      
      plot.title = element_text(face = "bold", size = 18, hjust = 0.5, margin = margin(b = 10)),
      plot.subtitle = element_text(size = 12, color = "gray30", face = "italic", hjust = 0.5, margin = margin(b = 20)),
      
      strip.text = element_text(face = "bold", size = 14, color = "white"),
      strip.background = element_rect(fill = "#333333", color = "black", linewidth = 1),
      
      legend.position = "top",
      legend.title = element_text(size = 13, face = "bold"),
      legend.text = element_text(size = 12),
      legend.key.size = unit(0.8, "cm"),
      
      panel.grid.major.x = element_blank(),
      panel.grid.minor = element_blank(),
      panel.grid.major.y = element_line(color = "gray70", linewidth = 0.5),
      
      panel.border = element_rect(color = "black", fill = NA, linewidth = 1),
      
      plot.margin = margin(t = 20, r = 15, b = 15, l = 15, unit = "pt"),
      
      panel.background = element_rect(fill = "white"),
      plot.background = element_rect(fill = "white")
    )
}

p_print_logloss <- plot_print_logloss(all_results)
print(p_print_logloss)

# Сохраняем
if(exists("path_to_data")) {
  ggsave(file.path(path_to_data, "CV_logloss_print.png"), 
         p_print_logloss, width = 10, height = 6, dpi = 600, bg = "white")
}
