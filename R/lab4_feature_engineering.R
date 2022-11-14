# Script to generate features for the MSIR data 

# read in data
feature_names <- c("x", "y", "label", "NDAI", "SD","CORR", "DF", "CF", "BF", "AF", "AN")
img1 <- read.table("data/image_data/image1.txt")
colnames(img1) <- feature_names
img2 <- read.csv("data/image_data/image2.txt")
colnames(img2) <- feature_names
img3 <- read.csv("data/image_data/image3.txt")
colnames(img3) <- feature_names

img1$label_final <- replace(img1$label, img1$label == 0, 1)
img2$label_final <- replace(img2$label, img2$label == 0, 1)
img3$label_final <- replace(img3$label, img3$label == 0, 1)


# converted the dataframe to matrices for speed-up
data <- as.matrix(img1[c("x", "y", "NDAI")])
get_average_matrix <- function(index){

  this_y <- data[index, 2]
  this_x <- data[index, 1]
  # get all of the columns with x and y values in the range we want. This works even if the area we find is not square (because the data isn't #   square). Then find the mean of the NDAI column for this region

  avg <- mean((data[data[,1] >= this_x - 4 & data[,1] <= this_x + 4 & data[,2] >= this_y - 4 & data[,2] <= this_y + 4, ])[,3])

  return(avg)
}

avg_NDAI_img1 <- lapply(1:nrow(img1), get_average_matrix)


img2_matrix <- as.matrix(img2[c("x", "y", "NDAI")])
get_average_matrix_img2 <- function(index){

  this_y <- img2_matrix[index, 2]
  this_x <- img2_matrix[index, 1]

  avg <- mean((img2_matrix[img2_matrix[,1] >= this_x - 4 & img2_matrix[,1] <= this_x + 4 & img2_matrix[,2] >= this_y - 4 & img2_matrix[,2] <= this_y + 4, ])[,3])

  return(avg)
}

img3_matrix <- as.matrix(img3[c("x", "y", "NDAI")])
get_average_matrix_img3 <- function(index){

  this_y <- img3_matrix[index, 2]
  this_x <- img3_matrix[index, 1]

  avg <- mean((img3_matrix[img3_matrix[,1] >= this_x - 4 & img3_matrix[,1] <= this_x + 4 & img3_matrix[,2] >= this_y - 4 & img3_matrix[,2] <= this_y + 4, ])[,3])

  return(avg)
}

avg_NDAI_img2 <- lapply(1:nrow(img2), get_average_matrix_img2)

avg_NDAI_img3 <- lapply(1:nrow(img3), get_average_matrix_img3)

img1$Avg_NDAI <- unlist(avg_NDAI_img1)

img2$Avg_NDAI <- unlist(avg_NDAI_img2)

img3$Avg_NDAI <- unlist(avg_NDAI_img3)


modify_rad <- function(DF, CF, AF, AN){
  return((DF*CF - AN*AF)/(DF*CF + AN*AF))
}

mult_rad <- function(DF, CF, BF, AF, AN){
  return(DF * CF * AF * AN)
}

mult_ndai_sd <- function(ndai, sd){
  return(ndai * sd)
}

# new features for image 1

img1 <- img1 %>% mutate(Sub_Feat = unlist(pmap(img1[c("DF", "CF", "AF", "AN")], function(DF, CF, AF, AN) modify_rad(DF = DF, CF = CF, AF = AF, AN = AN))))

img1 <- img1 %>% mutate(Mult_Rads = unlist(pmap(img1[c("DF", "CF", "BF", "AF", "AN")], function(DF, CF, BF, AF, AN) mult_rad(DF = DF, CF = CF, BF = BF, AF = AF, AN = AN))))

img1 <- img1 %>% mutate(Mult_AvgNDAi_SD = unlist(pmap(img1[c("Avg_NDAI", "SD")], function(Avg_NDAI, SD) mult_ndai_sd(ndai = Avg_NDAI, sd = SD))))

# repeat for image 2
img2 <- img2 %>% mutate(Sub_Feat = unlist(pmap(img2[c("DF", "CF", "AF", "AN")], function(DF, CF, AF, AN) modify_rad(DF = DF, CF = CF, AF = AF, AN = AN))))

img2 <- img2 %>% mutate(Mult_Rads = unlist(pmap(img2[c("DF", "CF", "BF", "AF", "AN")], function(DF, CF, BF, AF, AN) mult_rad(DF = DF, CF = CF, BF = BF, AF = AF, AN = AN))))

img2 <- img2 %>% mutate(Mult_AvgNDAi_SD = unlist(pmap(img2[c("Avg_NDAI", "SD")], function(Avg_NDAI, SD) mult_ndai_sd(ndai = Avg_NDAI, sd = SD))))

# repeat for image 3
img3 <- img3 %>% mutate(Sub_Feat = unlist(pmap(img3[c("DF", "CF", "AF", "AN")], function(DF, CF, AF, AN) modify_rad(DF = DF, CF = CF, AF = AF, AN = AN))))

img3 <- img3 %>% mutate(Mult_Rads = unlist(pmap(img3[c("DF", "CF", "BF", "AF", "AN")], function(DF, CF, BF, AF, AN) mult_rad(DF = DF, CF = CF, BF = BF, AF = AF, AN = AN))))

img3 <- img3 %>% mutate(Mult_AvgNDAi_SD = unlist(pmap(img3[c("Avg_NDAI", "SD")], function(Avg_NDAI, SD) mult_ndai_sd(ndai = Avg_NDAI, sd = SD))))

# save data
write.csv(img1, "data/img1_features.csv", row.names = FALSE)
write.csv(img2, "data/img2_features.csv", row.names = FALSE)
write.csv(img3, "data/img3_features.csv", row.names = FALSE)

