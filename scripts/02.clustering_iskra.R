library(tidyverse)    # обработка данных
library(broom)        # работа с результатами моделей   
library(FactoMineR)   # PCA и визуализация
library(factoextra)   # PCA и визуализация
library(dendextend)   # кластеризация и дендрограммы
library(RColorBrewer)


# читаем данные — загружаются таблицы признаков (MFW, char, POS, lemma)
# фиксируем номера экспериментов возле ссылки на матрицы признаков
url <- "https://raw.githubusercontent.com/AButon-8/iskra-project/refs/heads/main/features/exploratory/mfw_100_norm.csv" # CL_MFW_01, DB_COS_MFW100_12, DB_COS_HC_MFW100_17, DB_TAN_MFW100_KNN_26, DB_TAN_BIN_MFW100_29

url <- "https://raw.githubusercontent.com/AButon-8/iskra-project/refs/heads/main/features/exploratory/mfw_300_norm.csv" # CL_MFW_02, DB_COS_HC_MFW300_18

url <- "https://raw.githubusercontent.com/AButon-8/iskra-project/refs/heads/main/features/exploratory/mfw_500_norm.csv" # CL_MFW_03, CL_MFW_03a, DB_COS_HC_MFW500_19

url <- "https://raw.githubusercontent.com/AButon-8/iskra-project/refs/heads/main/features/exploratory/mfw_1000_norm.csv" # CL_MFW_04, DB_COS_HC_MFW1000_20

url <- "https://raw.githubusercontent.com/AButon-8/iskra-project/refs/heads/main/features/exploratory/char_3gram_norm.csv" # CL_NGRAM_05, DB_COS_3NGRAM_13, DB_COS_HC_3GRAM_21, DB_TAN_CHAR3_KNN_27, DB_TAN_BIN_CHAR3_30

url <- "https://raw.githubusercontent.com/AButon-8/iskra-project/refs/heads/main/features/exploratory/char_4gram_norm.csv" # CL_NGRAM_06, DB_COS_HC_4GRAM_22

url <- "https://raw.githubusercontent.com/AButon-8/iskra-project/refs/heads/main/features/exploratory/pos_18_nodubia.csv" # CL_POS_07

url <- "https://raw.githubusercontent.com/AButon-8/iskra-project/refs/heads/main/features/exploratory/pos_18.csv" # DB_COS_POS18_14, DB_COS_HC_POS18_23

url <- "https://raw.githubusercontent.com/AButon-8/iskra-project/refs/heads/main/features/exploratory/pos_2gram.csv" # CL_POS_08

url <- "https://raw.githubusercontent.com/AButon-8/iskra-project/refs/heads/main/features/exploratory/pos_3gram.csv" # CL_POS_09

url <- "https://raw.githubusercontent.com/AButon-8/iskra-project/refs/heads/main/features/exploratory/lemma_3000.csv" # CL_L_10

url <- "https://raw.githubusercontent.com/AButon-8/iskra-project/refs/heads/main/features/exploratory/lemma_3000_full.csv" # DB_COS_L3000_15, DB_COS_HC_L3000_24, DB_TAN_L3000_KNN_28

url <- "https://raw.githubusercontent.com/AButon-8/iskra-project/refs/heads/main/features/exploratory/lemma_1000.csv" # CL_L_11, CL_L_11a

url <- "https://raw.githubusercontent.com/AButon-8/iskra-project/refs/heads/main/features/exploratorylemma_1000_full.csv" # DB_COS_L1000_16, DB_COS_HC_L1000_25



# Читаем файл
mfw <- read_csv(url) |> 
  mutate(author = str_extract(file_name, "^[^_]+"), .after = file_name) |> # создаётся колонка author, из имени файла извлекается автор (до первого _)
  select(-author_folder)                                                   # удаляется лишняя колонка

# сколько всего уникальных авторов / текстов?
mfw |> 
  count(author)

# =========
# БЛОК 1. Кластеризация k-means
# =========

# кластеризация по методу k-means
# mfw[-c(1,2)] == убираются первые две колонки (file_name, author)
# scale(...) == стандартизация: среднее = 0, дисперсия = 1 (важно для k-means). все признаки приводятся к одной шкале
# kmeans(...) == сам алгоритм: centers = 8 - заранее задано 8 кластеров, nstart = 20 - алгоритм запускается 20 раз, выбирается лучший результат

set.seed(818)
km.out <- kmeans(scale(mfw[-c(1,2)]), centers = 8, nstart = 40)


# PCA  (изображаем вместе с кластерами km.out)
# PCA снижает размерность, сохраняет максимум вариации
# scale == стандартизация перед PCA. все признаки приводятся к одной шкале

pca_fit <- prcomp(mfw[-c(1,2)], scale. = TRUE, center = TRUE)

# Визуализация
# augment(mfw) == добавляет PCA-координаты к данным
# .fittedPC1, .fittedPC2 == первые две главные компоненты
# color = km.out$cluster == цвет = кластер из k-means
# geom_text(label = author) == каждая точка подписана автором

pca_fit  |> 
  augment(mfw) |> 
  ggplot(aes(.fittedPC1, .fittedPC2,
             color = as.factor(km.out$cluster))) +
  geom_text(aes(label = author), size = 3, alpha = 0.7) +
  scale_color_discrete(name = "кластер") +
  theme_minimal()


# если нужно сохранить визуализацию
p <- pca_fit |> 
  augment(mfw) |> 
  ggplot(aes(.fittedPC1, .fittedPC2,
             color = as.factor(km.out$cluster))) +
  geom_text(aes(label = author), size = 3, alpha = 0.7) +
  scale_color_discrete(name = "кластер") +
  theme_minimal()

exp_id <- "CL_NGRAM_06"

ggsave(
  filename = paste0(exp_id, "_pca_plot.png"),
  plot = p,
  path = "results/viz",
  width = 8,
  height = 6,
  dpi = 300
)


# Посмотреть сколько объектов в каждом кластере
# Размер кластеров и пересечение авторов х класеров
table(km.out$cluster)
table(mfw$author, km.out$cluster)

# для таблицы с результатами
CL_NGRAM_06 <- table(mfw$author, km.out$cluster)
cat(capture.output(write.table(CL_NGRAM_06, sep = "\t", col.names = NA)), sep = "\n")


# Посчитаем purity
# Пересечение «автор × кластер» сохраняем в таблице
tab <- table(mfw$author, km.out$cluster)

# считаем purity
# apply(tab, 2, max) = максимум в каждом кластере
# sum(...) = сумма максимальных авторов
# / sum(tab) = делим на общее число текстов

purity <- sum(apply(tab, 2, max)) / sum(tab)

purity

# дополнительно для проверки
tab                       # пересечение
colSums(tab)              # размер кластеров
apply(tab, 2, max)        # максимумы по кластерам




# все слипается; что это значит? все пишут под копирку?
# можно попробовать другие наборы предикторов 
# также имеет смысл поработать с леммами, а не словоформами


# =========
# БЛОК 2. МАТРИЦА РАССТОЯНИЙ (Косинусное сходство, NN (ближайшие соседи))
# =========


# ======= ПОСТРОЕНИЕ МАТРИЦЫ РАССТОЯНИЙ (или сходств)

# считаем косинусное расстояние
# на вход: матрица тексты × признаки | на выход: матрица текст × текст
# косинусное расстояние - мера сходства распределений, чем ближе тексты по стилю, тем выше значение

dist_mx <- mfw |> 
  column_to_rownames("file_name") |> # file_name превращаются в имена строк, дальше расстояния считаются между текстами
  select(-author) |>                 # удаляем меткe класса "author", это не признак, не нужна 
  philentropy::distance(method = "cosine", use.row.names = TRUE) # считается косинусное расстояние/сходство между всеми текстами


# ищем ближайших соседей (это сходство, а не расстояние! поэтому ищем max)
# col <- dist_mx[,id] = берём весь столбец для одного текста - это сходство этого текста со всеми остальными
# sort = сортируем: сначала самые похожие тексты
# [2:(n+1)] = [1] → это сам текст (сходство = 1), его пропускаем
# получаем: топ-n самых похожих текстов

find_nearest <- function(dist_mx, id, n) {
  col <- dist_mx[,id]
  sort(col, decreasing = TRUE)[2:(n+1)]
}

# Применение к dubia: ищем 5 текстов, которые максимально похожи на этот dubia
find_nearest(dist_mx, "dubia_finans_manifest.txt", 5)


# превращаем результат в таблицу
nearest_table <- function(dist_mx, mfw, id, k = 5) {
  
  res <- find_nearest(dist_mx, id, k)
  
  tibble(
    dubia = id,
    neighbor = names(res),
    similarity = as.numeric(res)
  ) |>
    left_join(
      mfw |> select(file_name, author),
      by = c("neighbor" = "file_name")
    ) |>
    arrange(desc(similarity))
}

# считаем для dubia
nearest_table(dist_mx, mfw, "dubia_finans_manifest.txt", 5)


# аналогично можно посмотреть для других dubia, но насколько вижу, расстояние между ближайшими соседями невелико (м.б. троцкий, м.б. ленин)

# считаем для всех dubia
dubia_files <- c(
  "dubia_finans_manifest.txt",
  "dubia_nasushnie_zadachi_I.txt",
  "dubia_novoe_poboishe_I.txt",
  "dubia_ot_red_iskry_I.txt",
  "dubia_ot_red_na_pismo_parvusa_I.txt",
  "dubia_poslednee_slovo_bund_I.txt",
  "dubia_priznaki_bankrotstva_I.txt",
  "dubia_slovo_mosc_vedomostyam_I.txt",
  "dubia_zakon_o_voznagr_I.txt"
)

all_results <- map_dfr(dubia_files, ~ nearest_table(dist_mx, mfw, .x, 5))

# сохраняем
write_csv(all_results, "results/db/nearest_neighbors_l1000.csv")


# считаем Majority vote
# для кадого dubia считаем, сколько соседей, выбираем автора с максимумом

majority_vote <- function(dist_mx, mfw, id, k = 5) {
  
  nearest <- nearest_table(dist_mx, mfw, id, k)
  
  nearest |>
    group_by(author) |>
    summarise(
      votes = n(),
      weight = sum(similarity),
      .groups = "drop"
    ) |>
    arrange(desc(votes), desc(weight))
}


# считаем для одного dubia
majority_vote(dist_mx, mfw, "dubia_finans_manifest.txt", 5)


# ВАРИАНТ 1. Добавляем «предсказанного автора»
predict_author <- function(dist_mx, mfw, id, k = 5) {
  
  votes <- majority_vote(dist_mx, mfw, id, k)
  
  top_author <- votes$author[1]
  top_votes <- votes$votes[1]
  
  tibble(
    dubia = id,
    predicted_author = top_author,
    votes = top_votes
  )
}

# ВАРИАНТ 2. Добавляем «предсказанного автора» + «сила сигнала»
predict_author <- function(dist_mx, mfw, id, k = 5) {
  
  # расстояния до всех текстов
  sims <- dist_mx[, id]
  
  # убираем сам текст
  sims <- sims[names(sims) != id]
  
  # берём топ-k ближайших
  top_k <- sort(sims, decreasing = TRUE)[1:k]
  
  # авторы этих текстов
  authors <- mfw$author[match(names(top_k), mfw$file_name)]
  
  df <- tibble(
    file = names(top_k),
    author = authors,
    similarity = as.numeric(top_k)
  )
  
  # majority votes
  votes <- df |> count(author, name = "votes") |> arrange(desc(votes))
  
  top_author <- votes$author[1]
  top_votes <- votes$votes[1]
  
  # 👉 средняя similarity для top_author
  mean_sim <- df |>
    filter(author == top_author) |>
    summarise(mean_similarity = mean(similarity)) |>
    pull(mean_similarity)
  
  tibble(
    dubia = id,
    predicted_author = top_author,
    votes = top_votes,
    mean_similarity = mean_sim
  )
}


# считаем для всех dubia
predictions <- map_dfr(dubia_files, ~ predict_author(dist_mx, mfw, .x, 5))


# сохраняем
write_csv(predictions, "results/db/pred_nearest_neighbors_l1000.csv")



# =========
# БЛОК 3. СТРОИМ ДЕРЕВО - иерархическая кластеризация
# =========

# ===== 
# попробуем построить дерево (преобразуем сходство в расстояние для этого)
# hclust = алгоритм. as.dist(...) = преобразуем в формат расстояний
# 1-dist_mx = превращаем сходство в расстояние
# method = "ward.D2": метод минимизирует внутрикластерную дисперсию, дает “компактные” кластеры
# получаем иерархическое дерево кластеризации

hc <- hclust(as.dist(1-dist_mx), method = "ward.D2")

# Преобразование в дендрограмму
hcd <- as.dendrogram(hc)

# Окраска ветвей (по кластерам) - назначаем цвета ветвям по кластерам
# k = 8 - разрезаем дерево на 8 кластеров
# color_branches() - каждая группа получает свой цвет
# brewer.pal() - палитра
# branches_lwd - толщина линий
# пакет dendextend: "color_branches()" красит ветви по кластерам, set("labels_col", ...) красит подписи, set("labels", ...) меняет текст подписей

colored_dend <- hcd  |>
  color_branches(k = 8, col = brewer.pal(8, "Set2")) |> 
  set("branches_lwd", 1.5)

# Подготовка подписей: создаем короткие подписи (автор + сокращенное название). Без .txt, 25 символов
labels_info <- mfw  |> 
  select(file_name, author)  |> 
  mutate(short_label = paste0(str_remove(file_name, "\\.txt$") |> 
                                str_trunc(25, side = "right")))

# формируем вектор цветов строго в порядке следования листьев в дереве
current_labels <- labels(colored_dend)
ordered_authors <- labels_info$author[match(current_labels, labels_info$file_name)]

# вытаскиваем короткие имена и авторов в этом порядке
ordered_short_labels <- labels_info$short_label[match(current_labels, labels_info$file_name)]

# создаем палитру для авторов. Каждому автору соотв. свой цвет
unique_authors <- unique(mfw$author)
author_colors_map <- setNames(rainbow(length(unique_authors)), unique_authors)

# Применяем изменения к дереву. Ветви = кластеры. Каждому автору соотв. свой цвет. 0.8 = уменьшаем размер текста.
colored_dend <- colored_dend |> 
  set("labels", ordered_short_labels) |>              
  set("labels_col", author_colors_map[ordered_authors]) |> 
  set("labels_cex", 0.8) 

# Отрисовка. Сохраняем в PNG, увеличиваем правое поле (для подписей)
# horiz = TRUE дает горизонтальное дерево
png("results/viz/dendrogram_25.png", width = 2000, height = 2000, res = 150)
par(mar = c(2, 1, 2, 15)) # увеличили правый отступ для длинных имен
plot(colored_dend, horiz = TRUE, main = "Иерархическая кластеризация")
dev.off()


# Считаем Purity
# получаем кластеры
clusters <- cutree(hc, k = 8)

# таблица: автор × кластер
tab <- table(mfw$author, clusters)

# считаем purity
purity <- sum(apply(tab, 2, max)) / sum(tab)

purity

# для таблицы с результатами
DB_hc <- table(mfw$author, clusters)
cat(capture.output(write.table(DB_hc, sep = "\t", col.names = NA)), sep = "\n")


# =========
# БЛОК 4. расстояние ТАНИМОТО
# =========


# Танимото - учитывает пересечение признаков, лучше чувствует различия, чем cosine
# === ВАРИАНТ 1. Tanimoto на частотах (как в COSINE)

library(philentropy)

# Считаем расстояние Tanimoto. philentropy::distance() возвращает расстояние, не сходство
# берём только числовые признаки

dist_mx <- mfw |> 
  column_to_rownames("file_name") |> 
  select(-author) |> 
  philentropy::distance(method = "tanimoto", use.row.names = TRUE) |> 
  as.matrix()



# Поиск ближайших соседей. 
# отличие от cosine: раньше было decreasing = TRUE, теперь FALSE, потому что это расстояние
find_nearest <- function(dist_mx, id, n) {
  col <- dist_mx[, id]
  sort(col, decreasing = FALSE)[2:(n+1)]  # меньше = ближе
}


# Применение к dubia: ищем 5 текстов, которые максимально похожи на этот dubia
find_nearest(dist_mx, "dubia_finans_manifest.txt", 5)


# превращаем результат в таблицу
nearest_table <- function(dist_mx, mfw, id, k = 5) {
  
  res <- find_nearest(dist_mx, id, k)
  
  tibble(
    dubia = id,
    neighbor = names(res),
    distance = as.numeric(res)   # ← было similarity
  ) |>
    left_join(
      mfw |> select(file_name, author),
      by = c("neighbor" = "file_name")
    ) |>
    arrange(distance)  # ← было desc(similarity)
}


# считаем для dubia
nearest_table(dist_mx, mfw, "dubia_finans_manifest.txt", 5)


# считаем для всех dubia
dubia_files <- c(
  "dubia_finans_manifest.txt",
  "dubia_nasushnie_zadachi_I.txt",
  "dubia_novoe_poboishe_I.txt",
  "dubia_ot_red_iskry_I.txt",
  "dubia_ot_red_na_pismo_parvusa_I.txt",
  "dubia_poslednee_slovo_bund_I.txt",
  "dubia_priznaki_bankrotstva_I.txt",
  "dubia_slovo_mosc_vedomostyam_I.txt",
  "dubia_zakon_o_voznagr_I.txt"
)

all_results <- map_dfr(dubia_files, ~ nearest_table(dist_mx, mfw, .x, 5))

# сохраняем
write_csv(all_results, "results/db/tan_nearest_neighbors_l3000.csv")



# Предсказанный автор + сила сигнала
predict_author_tanimoto <- function(dist_mx, mfw, id, k = 5) {
  
  # расстояния до всех текстов
  dists <- dist_mx[, id]
  
  # убираем сам текст
  dists <- dists[names(dists) != id]
  
  # берём k ближайших (минимальное расстояние!)
  top_k <- sort(dists, decreasing = FALSE)[1:k]
  
  # авторы этих текстов
  authors <- mfw$author[match(names(top_k), mfw$file_name)]
  
  df <- tibble(
    file = names(top_k),
    author = authors,
    distance = as.numeric(top_k)   # ← было similarity
  )
  
  # majority vote
  votes <- df |> count(author, name = "votes") |> arrange(desc(votes))
  
  top_author <- votes$author[1]
  top_votes <- votes$votes[1]
  
  # 👉 среднее расстояние для top_author (меньше = лучше)
  mean_dist <- df |>
    filter(author == top_author) |>
    summarise(mean_distance = mean(distance)) |>
    pull(mean_distance)
  
  tibble(
    dubia = id,
    predicted_author = top_author,
    votes = top_votes,
    mean_distance = mean_dist
  )
}


# Применяем ко всем dubia
predictions_tan <- map_dfr(dubia_files, ~ predict_author_tanimoto(dist_mx, mfw, .x, 5))


# сохраняем
write_csv(predictions_tan, "results/db/tan_pred_nearest_neighbors_l3000.csv")




# === ВАРИАНТ 2: бинарный Tanimoto

library(philentropy)
library(dplyr)
library(tibble)
library(purrr)
library(readr)

# матрица признаков (с именами строк!)
mfw_matrix <- mfw |> 
  column_to_rownames("file_name") |> 
  select(-author) |> 
  as.matrix()


# Бинаризация
mfw_bin <- mfw_matrix > 0      # TRUE/FALSE
mfw_bin <- mfw_bin * 1         # → 1/0


# Танимото бинарный
dist_mx_bin <- distance(mfw_bin, method = "tanimoto", use.row.names = TRUE) |> 
  as.matrix()


# проверка
rownames(dist_mx_bin)[1:5] 


# Функция предсказания (бинарный вариант)
predict_author_tanimoto_bin <- function(dist_mx, mfw, id, k = 5) {
  
  dists <- dist_mx[, id]
  dists <- dists[names(dists) != id]
  
  top_k <- sort(dists, decreasing = FALSE)[1:k]
  
  authors <- mfw$author[match(names(top_k), mfw$file_name)]
  
  df <- tibble(
    file = names(top_k),
    author = authors,
    distance = as.numeric(top_k)
  )
  
  votes <- df |> count(author, name = "votes") |> arrange(desc(votes))
  
  top_author <- votes$author[1]
  top_votes <- votes$votes[1]
  
  mean_dist <- df |>
    filter(author == top_author) |>
    summarise(mean_distance = mean(distance)) |>
    pull(mean_distance)
  
  tibble(
    dubia = id,
    predicted_author = top_author,
    votes = top_votes,
    mean_distance = mean_dist
  )
}


# считаем для всех dubia
dubia_files <- c(
  "dubia_finans_manifest.txt",
  "dubia_nasushnie_zadachi_I.txt",
  "dubia_novoe_poboishe_I.txt",
  "dubia_ot_red_iskry_I.txt",
  "dubia_ot_red_na_pismo_parvusa_I.txt",
  "dubia_poslednee_slovo_bund_I.txt",
  "dubia_priznaki_bankrotstva_I.txt",
  "dubia_slovo_mosc_vedomostyam_I.txt",
  "dubia_zakon_o_voznagr_I.txt"
)


# Предсказание
predictions_tan_bin <- map_dfr(
  dubia_files, 
  ~ predict_author_tanimoto_bin(dist_mx_bin, mfw, .x, k = 5)
)


# сохранение
write_csv(predictions_tan_bin, "results/db/tan_bin_pred_3ngram.csv")


