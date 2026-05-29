library(tidyverse)
library(textrecipes)
library(tidymodels)
library(tidytext)
library(stylo)
library(e1071)
library(caret)
library(yardstick)


# CF_LR_MFW500_59,	CF_LR_MFW500_60,	CF_SVM_MFW500_61,	CF_RF_MFW500_62
url_ch <- "https://raw.githubusercontent.com/AButon-8/iskra-project/refs/heads/main/features/mfw_C3_500.csv"


# Функция для диагностики "коллаборативности" (Улучшенная с chunk_id)
# Функция с chunk_id
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


# Применяем ко всем тестовым текстам

# Создаем диагностическую таблицу для всего теста
test_diagnostics <- list()

for (i in 1:nrow(prob_matrix)) {
  test_diagnostics[[i]] <- diagnose_text(
    prob_row = prob_matrix[i, ],
    chunk_id_value = mfw_ch$chunk_id[-train_idx][i],
    true_author_value = y_test[i],
    threshold = 0.4
  )
}

# Превращаем в data.frame для удобства
diag_df <- do.call(rbind, lapply(test_diagnostics, function(x) {
  data.frame(
    chunk_id = x$chunk_id,
    status = x$status,
    top1_author = x$top1_author,
    top1_prob = x$top1_prob,
    top2_author = x$top2_author,
    top2_prob = x$top2_prob,
    ratio = x$ratio,
    correct = x$correct,
    true_author = x$true_author,
    stringsAsFactors = FALSE
  )
}))
