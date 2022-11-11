# lab4 modeling part

# load packages
library(dplyr)
library(ROCR)
library(glmnet)
library(e1071)
library(randomForest)
library(class)

# define functions for future use
# Compute the ROC and precision-recall curves from the response
roc_pr_curve <- function(response, label) {
  pred <- prediction(response, label)
  roc <- performance(pred, "tpr", "fpr")
  pr <- performance(pred, "prec", "rec")
  return(list(roc = roc, pr = pr))
}

# Compute error metrics from a given confusion matrix
err_metric <- function(confusion_mat) {

  # load the TN, TP, FP, and FN
  TN <- confusion_mat[1, 1]
  TP <- confusion_mat[2, 2]
  FP <- confusion_mat[1, 2]
  FN <- confusion_mat[2, 1]

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

  # return the summary of error metrics
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

# randomly pick 4 of them to be the test set
set.seed(215)
perm <- sample(1:12, 12, replace=F)
test <- full_data[full_data$block %in% perm[1:4], ]
# the remaining 8 blocks: cross validation sample
training <- full_data[full_data$block %in% perm[5:8], ]
validation <- full_data[full_data$block %in% perm[9:12], ]
cv_data <- rbind(training, validation)

# --------------------- #
# --- modeling part --- #
# --------------------- #

# ------------- 1. logistic regression ------------- #

# 1.1 evaluate the feature importance using AIC
# start with an intercept-only model
logit_null_model <- glm(label_final ~ 1, data = cv_data, family = "binomial")
# then a logit model using all features
logit_all_model <- glm(label_final ~ NDAI + SD + CORR + Avg_NDAI + DF + CF + BF + AF + AN,
                       data = cv_data, family = "binomial")
# next do the forward stepwise regressions
logit_forward <- step(logit_null_model, direction = "forward",
                      scope = formula(logit_all_model), trace = 0)

# summarize the feature importance results
logit_forward_select <- logit_forward$anova
# logit_forward_select[c('Step', 'AIC')]

# save the feature importance results
write.csv(logit_forward_select, "results/logit_feature_importance.csv")

# 1.2 compare logit regression with and without regularization
# without regularization: logit_all_model
# with regularization: use the glmnet function (cross fit to tune hyper-parameter lambda)
cv.fit <- cv.glmnet(as.matrix(cv_data %>%
                                select(-c("x", "y", "label_final", "block", "image"))),
                    cv_data$label_final)

# plot the MSE vs log(lambda)
plot(cv.fit)

# find the best lambda -
#   the result is 0.00019, suggesting the regularization does not help a lot
#   this is due to our low dimension in features - we only have 9 in total
cv.fit$lambda.min

# 1.3 choose the logit model without regularization and get results

# get the prediction on the test set
logit_pred <- predict(logit_all_model, test, type = "response")
logit_pred_binary <- ifelse(logit_pred > 0.5, 1, 0)

# compute the confusion matrix
logit_confusion_mat <- table(logit_pred_binary, test$label_final)

# compute the roc and precision-recall curve
roc_pr_logit <- roc_pr_curve(logit_pred, test$label_final)

# save the prediction results
write.csv(cbind(logit_pred, logit_pred_binary, test$label_final),
          "results/logit_predictions.csv")


# ------------- 2. K Nearest Neighbors -------------#
# predict the class of a point by its k nearest neighbors (need to be in the same graph)

# 2.1 cross-validation to tune hyper parameter: k
# specify a list of k values to choose from
k_list <- c(2, 3, 5, 7, 10, 20, 50, 75, 100)
# save the results in a dataframe
knn_accuracy_list <- data.frame()
# scale the features in training and validation sets
training_scale <- scale(training[, c(3:10, 12)])
validation_scale <- scale(validation[, c(3:10, 12)])

# loop over all possible choices of k value
for (k in k_list) {
  # print(k)

  # run knn model on the training set and get prediction on the validation set
  knn_model <- knn(train = training_scale, test = validation_scale,
                   cl = training$label_final, k = k)

  # compute the confusion matrix
  knn_confusion_mat <- table(knn_model, validation$label_final)

  # get th accuracy from err_metric function
  accuracy_res <- data.frame(k = k, accuracy = err_metric(knn_confusion_mat)$accuracy)

  # save the result to the list
  knn_accuracy_list <- rbind(knn_accuracy_list, accuracy_res)
}
# plot the accuracy vs k values
plot(knn_accuracy_list)
# save the results
write.csv(knn_accuracy_list, "results/knn_cross_validation.csv")

# 2.2 fit the model using all cv_data and k = 50
# we use k = 50 because it gives the best accurcy from the cross-validation results

# scale the features
cv_scale <- scale(cv_data[, c(3:10, 12)])
test_scale <- scale(test[, c(3:10, 12)])

# run knn model on the whole cross validation set and get predictions on the test
knn_pred <- knn(train = cv_scale, test = test_scale,
                cl = cv_data$label_final, k = 50, prob = TRUE)
knn_pred_binary <- knn(train = cv_scale, test = test_scale,
                       cl = cv_data$label_final, k = 50)

# compute the confusion matrix
knn_confusion_mat <- table(knn_pred_binary, test$label_final)

# compute the roc and precision-recall curve
roc_pr_knn <- roc_pr_curve(as.numeric(knn_pred), test$label_final)

# save the prediction results
write.csv(cbind(knn_pred, knn_pred_binary, test$label_final),
          "results/knn_predictions.csv")


# ------------- 3. Random Forest ------------- #

# 3.1 cross-validation to tune hyper parameter: mtry
# specify a list of mtry values to choose from
mtry_list <- 2:9
# save the results in a dataframe
rf_accuracy_list <- data.frame()
# covert the label to factors
training$label_final_factor <- as.factor(training$label_final)
validation$label_final_factor <- as.factor(validation$label_final)

# loop over all possible choices of mtry value
for (mtry in mtry_list) {
  # print(mtry)

  # run random forest model on the training set
  rf_model <- randomForest(label_final_factor ~ NDAI + SD + CORR + Avg_NDAI + DF + CF + BF + AF + AN,
                           data = training, mtry = mtry, importance = FALSE)

  # get the prediction on the validation set
  rf_pred <- predict(rf_model, newdata = validation)

  # compute the confusion matrix
  rf_confusion_mat = table(rf_pred, validation$label_final)

  # get the accuracy from err_metric function
  accuracy_res <- data.frame(mtry = mtry, accuracy = err_metric(rf_confusion_mat)$accuracy)

  # save the result to the list
  rf_accuracy_list <- rbind(rf_accuracy_list, accuracy_res)
}
# plot the accuracy vs mtry values
plot(rf_accuracy_list)
# save the results
write.csv(rf_accuracy_list, "results/rf_cross_validation.csv")

# 3.2 fit the model using all cv_data and mtry = 2
# we use mtry = 2 because it gives the best accuracy from the cross-validation results

# convert the label to factors
cv_data$label_final_factor <- as.factor(cv_data$label_final)

# run random forest model on the whole training + validation set
rf_model <- randomForest(label_final_factor ~ NDAI + SD + CORR + Avg_NDAI + DF + CF + BF + AF + AN,
                         data = cv_data, mtry = 2, importance = TRUE)

# summarize feature importance and plot it
write.csv(importance(rf_model), "results/rf_feature_importance.csv")
varImpPlot(rf_model)

# get the prediction on the test set
rf_pred <- predict(rf_model, newdata = test, type = "prob")[, 2]
rf_pred_binary <- predict(rf_model, newdata = test, type = "response")

# compute the confusion matrix
rf_confusion_mat <-  table(rf_pred_binary, test$label_final)

# compute the roc and precision-recall curve
roc_pr_rf <- roc_pr_curve(rf_pred, test$label_final)

# save the prediction results
write.csv(cbind(rf_pred, rf_pred_binary, test$label_final),
          "results/rf_predictions.csv")

# ------------- 4. QDA ------------- #
# there is no hyper-parameter for QDA model, so no need to do cross-validation tuning

# run qda model on the whole training + validation set
qda_model <- MASS::qda(label_final ~ NDAI + SD + CORR + Avg_NDAI + DF + CF + BF + AF + AN,
                       data = cv_data)

# get the prediction on the test set
qda_pred <- data.frame(predict(qda_model, newdata = test)$posterior)[2]
qda_pred_binary <- ifelse(qda_pred > 0.5, 1, 0)

# compute the confusion matrix
qda_confusion_mat <- table(qda_pred_binary, test$label_final)

# compute the roc and precision-recall curve
roc_pr_qda <- roc_pr_curve(qda_pred, test$label_final)

# save the prediction results
write.csv(cbind(qda_pred, qda_pred_binary, test$label_final),
          "results/qda_predictions.csv")


# ------------- 5. SVM ------------- #
# SVM takes very long time to fit, so we pick a random sample of size 50,000
#   and did not conduct cross-validation, use the default values for hyper paras

# generate a random sample of size 50,000
set.seed(215)
random_sample_index <- sample(1:dim(cv_data)[1], 50000, F)

# scale the training and test sets
cv_scale <- data.frame(scale(cv_data[random_sample_index, c(3:10, 12)]))
cv_scale$label_final <- cv_data[random_sample_index,]$label_final
test_scale <- scale(test[, c(3:10, 12)])

# run the svm model on the random sample
svm_model <- svm(label_final ~ NDAI + SD + CORR + Avg_NDAI + DF + CF + BF + AF + AN,
                 data = cv_scale, probability = TRUE)

# get the prediction on the test set
svm_pred <- predict(svm_model, test_scale %>% select(c("NDAI", "SD", "CORR", "DF", "CF",
                                                 "BF", "AF", "AN", "Avg_NDAI")))
svm_pred_binary <- ifelse(svm_pred > 0.5, 1, 0)

# compute the confusion matrix
svm_confusion_mat <- table(svm_pred_binary, test$label_final)

# compute the roc and precision-recall curve
roc_pr_svm <- roc_pr_curve(svm_pred, test$label_final)

# save the prediction results
write.csv(cbind(svm_pred, svm_pred_binary, test$label_final),
          "results/svm_predictions.csv")


# all sample
cv_scale <- data.frame(scale(cv_data[, c(3:10, 12)]))
cv_scale$label_final <- cv_data$label_final
test_scale <- data.frame(scale(test[, c(3:10, 12)]))

start <- Sys.time()
print(start)
svm_model <- svm(label_final ~ NDAI + SD + CORR + Avg_NDAI + DF + CF + BF + AF + AN,
                 data = cv_scale, probability = TRUE)
end <- Sys.time()
print(end)
print(end - start)


# ------------- compare results ------------- #

# plot the roc curves
plot(roc_pr_logit$roc, col = "red")
plot(roc_pr_knn$roc, col = "blue",  add = TRUE)
plot(roc_pr_rf$roc, col = "green", add = TRUE)
plot(roc_pr_qda$roc, col = "black", add = TRUE)
plot(roc_pr_svm$roc, col = "yellow", add = TRUE)

legend("bottomright", c("Logistic Regression",
                        "k-Nearest Neighbour",
                        "Random Forest",
                        "Quadratic Discriminant Analysis",
                        "Support Vector Machines"),
       lty=1, col = c("red", "blue", "green", "black", "yellow"),
       bty = "n")

title("ROC curves")

# plot the precision-recall curves
plot(roc_pr_logit$pr, col = "red")
plot(roc_pr_knn$pr, col = "blue",  add = TRUE)
plot(roc_pr_rf$pr, col = "green", add = TRUE)
plot(roc_pr_qda$pr, col = "black", add = TRUE)
plot(roc_pr_svm$pr, col = "yellow", add = TRUE)

legend("bottomleft", c("Logistic Regression",
                       "k-Nearest Neighbour",
                       "Random Forest",
                       "Quadratic Discriminant Analysis",
                       "Support Vector Machines"),
       lty=1, col = c("red", "blue", "green", "black", "yellow"),
       bty = "n")

title("PR curves")

# summarize and save the error metrics
err_metric_res <- rbind(cbind(err_metric(logit_confusion_mat), method = "logit"),
                        cbind(err_metric(knn_confusion_mat), method = "knn"),
                        cbind(err_metric(rf_confusion_mat), method = "rf"),
                        cbind(err_metric(qda_confusion_mat), method = "qda"),
                        cbind(err_metric(svm_confusion_mat), method = "svm"))
write.csv(err_metric_res, "results/error_metrics.csv")

