# lab4 modeling part

library(dplyr)
library(ROCR)
library(glmnet)
library(e1071)
library(randomForest)
library(class)

# functions for future use
# Compute the ROC curve from the response
roc_perf <- function(response, label){
  pred <- prediction(response, label)
  perf <- performance(pred, "tpr", "fpr")
  return(perf)
}

# Compute error metrics from a given confusion matrix
err_metric <- function(confusion_mat) {

  # load the TN, TP, FP, and FN
  TN <- confusion_mat[1,1]
  TP <- confusion_mat[2,2]
  FP <- confusion_mat[1,2]
  FN <- confusion_mat[2,1]

  # compute the error matrics
  accuracy <- (TP + TN) / (TP + TN + FP + FN)
  precision <- TP / (TP + FP)
  recall <- FP / (FP + TN)
  fpr <- FP / (FP + TN)
  fnr <- FN / (FN + TP)
  f1_score <- 2 * ((precision * recall) / (precision + recall))

  # summarize the results into a dataframe
  err_metric_df <- data.frame(t(c(accuracy, precision, recall,
                                  fpr, fnr, f1_score)))
  colnames(err_metric_df) <- c("accuracy", "precision", "recall",
                               "fpr", "fnr", "f1_score")

  return(err_metric_df)
}


# load the processed cloud data
img1_raw <- read.csv("data/img1_features.csv")
img2_raw <- read.csv("data/img2_features.csv")
img3_raw <- read.csv("data/img3_features.csv")

# change unlabeled data to cloud
img1_raw$label_final <- replace(img1_raw$label, img1_raw$label == 0, 1)
img2_raw$label_final <- replace(img2_raw$label, img2_raw$label == 0, 1)
img3_raw$label_final <- replace(img3_raw$label, img3_raw$label == 0, 1)

# change the label to binary variables and select features
img1 <- img1_raw %>%
  mutate(label_final = as.integer((label_final + 1) / 2)) %>%
  select(-c(label, Sub_Feat, Mult_Rads, Sub_Avg_Mult, Mult_AvgNDAi_SD))
img2 <- img2_raw %>%
  mutate(label_final = as.integer((label_final + 1) / 2)) %>%
  select(-c(label, Sub_Feat, Mult_Rads, Mult_AvgNDAi_SD))
img3 <- img3_raw %>%
  mutate(label_final = as.integer((label_final + 1) / 2)) %>%
  select(-c(label, Sub_Feat, Mult_Rads, Mult_AvgNDAi_SD))

# --- specify the test set, training set, and validation set --- #
# divide each image into 4 blocks - 12 blocks in total
# randomly pick 4 of them to be the test set
# and split the remaining 8 to training and validation
img1$block <-
  1 * (img1$x <= median(img1$x) & img1$y <= median(img1$y)) +
  2 * (img1$x > median(img1$x) & img1$y <= median(img1$y)) +
  3 * (img1$x <= median(img1$x) & img1$y > median(img1$y)) +
  4 * (img1$x > median(img1$x) & img1$y > median(img1$y))
img2$block <-
  5 * (img2$x <= median(img2$x) & img2$y <= median(img2$y)) +
  6 * (img2$x > median(img2$x) & img2$y <= median(img2$y)) +
  7 * (img2$x <= median(img2$x) & img2$y > median(img2$y)) +
  8 * (img2$x > median(img2$x) & img2$y > median(img2$y))
img3$block <-
  9 * (img3$x <= median(img3$x) & img3$y <= median(img3$y)) +
  10 * (img3$x > median(img3$x) & img3$y <= median(img3$y)) +
  11 * (img3$x <= median(img3$x) & img3$y > median(img3$y)) +
  12 * (img3$x > median(img3$x) & img3$y > median(img3$y))
img1$image <- "image1"
img2$image <- "image2"
img3$image <- "image3"

# rbind images and blocks to a full_data dataframe
full_data <- rbind(img1, img2, img3)
head(full_data)

# randomly pick 2 of them to be the test set
set.seed(215)
perm <- sample(1:12, 12, replace=F)
test <- full_data[full_data$block %in% perm[1:4], ]
# the remaining 10 blocks: cross validation sample
training <- full_data[full_data$block %in% perm[5:8], ]
validation <- full_data[full_data$block %in% perm[9:12], ]
cv_data <- rbind(training, validation)

# --- modeling part --- #
err_metric_res <- data.frame()

# --- 1. logistic regression --- #
# 1.1 ndai + corr + sd, no regularization
logit1 <- glm(label_final ~ NDAI + SD + CORR, data = cv_data, family = "binomial")
# 1.2 ndai + corr + sd + avg_ndai, no regularization
logit2 <- glm(label_final ~ NDAI + SD + CORR + Avg_NDAI, data = cv_data, family = "binomial")
# 1.3 all features, no regularization
logit3 <- glm(label_final ~ NDAI + SD + CORR + DF + CF + BF + AF + AN,
              data = cv_data, family = "binomial")
# 1.4 all features + avg_ndai, no regularization
logit4 <- glm(label_final ~ NDAI + SD + CORR + Avg_NDAI + DF + CF + BF + AF + AN,
              data = cv_data, family = "binomial")
# 1.5 all features + avg_ndai, with regularization
cv.fit <- cv.glmnet(as.matrix(cv_data %>% select(-c("x", "y", "label_final", "block", "image"))),
                    cv_data$label_final)
plot(cv.fit)
cv.fit$lambda.min
# the result is 0.0001900063, suggesting the regularization does not help a lot
# this is due to our low dimension in features - we only have 9 in total
# happy with the logit regression without regularization

roc_perf1 <- roc_perf(predict.glm(logit1, test, type = "response"), test$label_final)
roc_perf2 <- roc_perf(predict.glm(logit2, test, type = "response"), test$label_final)
roc_perf3 <- roc_perf(predict.glm(logit3, test, type = "response"), test$label_final)
roc_perf4 <- roc_perf(predict.glm(logit4, test, type = "response"), test$label_final)

plot(roc_perf1, col = "red")
plot(roc_perf2, col = "blue",  add = TRUE)
plot(roc_perf3, col = "green", add = TRUE)
plot(roc_perf4, col = "black", add = TRUE)

legend("bottomright", c("Model: three main features",
                        "Model: three main features + avg_ndai",
                        "Model: all provided features",
                        "Model: all provided features + avg_ndai"),
       lty=1, col = c("red", "blue", "green", "black"), bty="n")
# this suggests that the avg_ndai feature helps much

logit_pred_binary <- ifelse(predict.glm(logit4, test, type = "response") > 0.5, 1, 0)
logit_confusion_mat <- table(logit_pred_binary, test$label_final)
err_metric_res <- rbind(err_metric_res,
                        cbind(err_metric(logit_confusion_mat), method = "logit"))
err_metric_res

# --- 2. KNN --- #
# predict the class of a point by its k nearest neighbors (need to be in the same graph)
set.seed(215)
random_sample_index <- sample(1:dim(cv_data)[1], 10000, F)

# scale the features
training_scale <- scale(cv_data[random_sample_index, c(3:10, 12)])
test_scale <- scale(test[, c(3:10, 12)])
# Fitting KNN Model
# to training dataset
knn_model <- knn(train = training_scale,
                 test = test_scale,
                 cl = cv_data[random_sample_index,]$label_final,
                 k = 5)
knn_pred_binary <- knn_model
knn_confusion_mat <- table(knn_pred_binary, test$label_final)
err_metric_res <- rbind(err_metric_res,
                        cbind(err_metric(knn_confusion_mat), method = "knn"))
err_metric_res

# --- 3. SVM --- #
svm_model <- svm(label_final ~ NDAI + SD + CORR + Avg_NDAI + DF + CF + BF + AF + AN,
                 data = cv_data[random_sample_index,], probability = TRUE)
svm_pred <- predict(svm_model,
                    test %>% select(-c("x", "y", "label_final", "block", "image")))
roc_svm <- roc_perf(svm_pred, test$label_final)
# plot(roc_svm, add=T)
svm_pred_binary <- ifelse(svm_pred > 0.5, 1, 0)
svm_confusion_mat <- table(svm_pred_binary, test$label_final)
err_metric_res <- rbind(err_metric_res,
                        cbind(err_metric(svm_confusion_mat), method = "svm"))
err_metric_res

# --- 4. QDA --- #
qda_model <- MASS::qda(label_final ~ NDAI + SD + CORR + Avg_NDAI + DF + CF + BF + AF + AN,
                       data = cv_data)
qda_pred <- data.frame(predict(qda_model, newdata = test)$posterior)[2]
roc_qda <- roc_perf(qda_pred, test$label_final)
# plot(roc_qda, add=T)
qda_pred_binary <- ifelse(qda_pred > 0.5, 1, 0)
qda_confusion_mat <- table(qda_pred_binary, test$label_final)
err_metric_res <- rbind(err_metric_res,
                        cbind(err_metric(qda_confusion_mat), method = "qda"))
err_metric_res

# --- 5. RF --- #
cv_data$label_final_factor <- as.factor(cv_data$label_final)
rf_model <- randomForest(label_final_factor ~ NDAI + SD + CORR + Avg_NDAI + DF + CF + BF + AF + AN,
                         data = cv_data[random_sample_index,], mtry = 5, importance = TRUE)
importance(rf_model)
varImpPlot(rf_model)
# summary(rf_model)

rf_pred <- predict(rf_model, newdata = test)
rf_confusion_mat = table(rf_pred, test$label_final)
err_metric_res <- rbind(err_metric_res,
                        cbind(err_metric(rf_confusion_mat), method = "rf"))
err_metric_res

write.csv(err_metric_res, "err_metric_res.csv")

# warnings:
# 1. the knn, svm, and rf are slow, so only used 10,000 subsamples for now - this affects the performance
# 2. for knn and rf, should do more thorough cross-validation to tune the hyperparameters
# 3. discuss about the model assumptions? e.g., logit & some other models assume independence, which is unlikely

# takeaway:
# from the result, we see the logit and qda models are working the best in terms of accuracy


# cross-validation to tune hyper parameters
random_sample_index_training <- sample(1:dim(training)[1], 10000, F)

# KNN
k_list <- c(3, 5, 10, 20, 50, 100)
knn_accuracy_list <- data.frame()
# scale the features
training_scale <- scale(training[random_sample_index_training, c(3:10, 12)])
validation_scale <- scale(validation[, c(3:10, 12)])
for (k in k_list) {
  print(k)
  knn_model <- knn(train = training_scale,
                   test = validation_scale,
                   cl = training[random_sample_index_training,]$label_final,
                   k = k)
  knn_confusion_mat <- table(knn_model, validation$label_final)
  accuracy_res <- data.frame(k = k, accuracy = err_metric(knn_confusion_mat)$accuracy)
  knn_accuracy_list <- rbind(knn_accuracy_list, accuracy_res)
}


# RF
mtry_list <- 2:9
rf_accuracy_list <- data.frame()

for (mtry in mtry_list) {
  print(mtry)
  training$label_final_factor <- as.factor(training$label_final)
  validation$label_final_factor <- as.factor(validation$label_final)

  rf_model <- randomForest(label_final_factor ~ NDAI + SD + CORR + Avg_NDAI + DF + CF + BF + AF + AN,
                           data = training[random_sample_index_training,], mtry = mtry, importance = FALSE)
  rf_pred <- predict(rf_model, newdata = validation)
  rf_confusion_mat = table(rf_pred, validation$label_final)
  accuracy_res <- data.frame(mtry = mtry, accuracy = err_metric(rf_confusion_mat)$accuracy)
  rf_accuracy_list <- rbind(rf_accuracy_list, accuracy_res)
}







