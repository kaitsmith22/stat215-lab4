# stat215-lab4

# EDA: Kaitlin 

# Modelling: Ghafar & Sizhu 

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
