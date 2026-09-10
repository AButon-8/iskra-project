library(tidyverse)
library(textrecipes)
library(tidymodels)
library(tidytext)
library(stylo)
library(e1071)
library(caret)
library(yardstick)




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
