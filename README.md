# stat215-lab4

# EDA: Kaitlin 

# Modelling: Ghafar & Sizhu 

Sizhu's update 11/12

Discussion on model descriptions and assumptions:
1. Logistic regression model: logistic regression is a standard and canonical choice of classification problems when the outcome is binary. In the logistic model, we assume the data points are independently and identically distributed from a model where the probability of outcome being 1 is equal to a non-linear (sigmoid) transform of some linear combinations of the covariates. Both the generalized linearity and the IID assumptions are very strong. Specifically, the independence assumption is likely to be violated in our dataset due to spatial correlation. We need to be careful when interpreting the regression coefficients. However, in this lab, our main goal is the prediction problem instead of estimating the coefficients in the specified logistic model, we focus less on the model assumption verification and evaluate the models mainly from the perspective of prediction performance. We also fit logistic regression with l1 regularization penalty to avoid over-fitting. 
2. KNN: KNN is a classification model where we predict the binary outcome of a single point by referring to the values of its k nearest neighbors, where k is a hyper-parameter we need to tune. This non-parametric model does not make assumptions about independence between data points or the specific probability distributions of the data. Therefore, we are able to use it even though we observe high spatial correlations in our cloud data.
3. Random forest: random forest is an ensemble machine learning model for classification and prediction. We construct various decision trees from the bootstrapped subsample of training data and summarize the results to a predicted value of the outcome from the random forest. For each decision tree, we can train it on only a subset of all covariates to determine splits that best separate the two classes. Random forest is also a non-parametric model that makes minimal assumptions on the distribution of the data points. 
4. Quadratic Discriminant Analysis (QDA): QDA is a version of Bayesian classification model which finds the linear separation in the second-order polynomial basis expansion of the covariates. It captures the lower-order linear interactions between covariates that separate the positive vs negative outcomes. The main model assumption QDA makes is that the covariates from the two classes have normal probability distributions with (possibly) different covariance matrices. As shown in our ggpair plot, not all covariates are obviously normally distributed thus it is likely the assumption is not perfectly satisfied in our data.
5. Support Vector Machines (SVM): SVM efficiently performs a non-linear classification by implicitly mapping the inputs into high-dimensional covariate spaces and then constructing a hyperplane separating the two classes of data points. It also makes the independence assumption between observations, which is likely to be violated in our dataset. Moreover, it assumes that in the true underlying data generating process, the two classes are linearly separable, which is another strong assumption in our dataset.

Sizhu's update 11/11

R codes and results updated in GitHub. 

1. Compute the feature importance using: 
    (1). Forward stepwise regression in logistic regression models; results in “results/logit_feature_importance.csv”; can select only the AIC measure when reporting: logit_forward_select[c('Step', 'AIC')]
    (2). Feature importance measure provided in the random forest function; results in “results/rf_feature_importance.csv”. 
    
2. Will write
    (1). Description for each model
    (2). What are the assumptions
    (3). Whether they are satisfied or not
    
3. To summarize the results:
    (1). Plot the ROC curves and precision-recall curves, saved in “figures/roc_curves” and “figures/pr_curves”;  (could possibly change color and/or style?)
    (2). error metrics matrix is in “results/error_metrics_res.csv”, can use different metrics to evaluate models - logistic regression is the best in terms of many metrics, e.g., accuracy; caveat: svm runs very slowly, so we only fit it on a random subsample of size 50,000 of our sample - the performance is also based on model fitted using subsample;
    (3). for cross-validation part, describe how we divide the data into blocks and randomly split into three parts: test + training + validation; 
    (4). we use cross-validation to tune hyper parameters in knn and random forest models; use cross-fit to tune hyper parameter in the regularized logistic regression; ada does not have hyper-parameter so no need to do cross-validation tuning; svm is so slow so we didn’t do cross validation and just used the default values
    
4. We can pick random forest. Feature importance figure in “figures/rf_feature_importance”. Learning curve to be plotted.

Besides, I’ve also saved the predictions of all 5 models in the “results/“ folder just in case, so are the cross-validation results.


# EDA 

How to deal with unlabelled data:

Dealing with Unlabelled data:
- based on NDAI, label this as cloud data

Feature Engineering:
- avg of NDAI score around each point 
 
Modelling Notes:
- KNN works well with NDAI (max or min for edge detection)

Train Test Split:
- split up images into blocks, and then use CV 
- leave out 15-20% for validation, don't touch it for training/test
