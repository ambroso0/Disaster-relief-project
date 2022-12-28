**Multiple Logistic Regression**

The first model we are going to test is the multiple logistic regression where BlueTarp is going to be the response variable and Red, Green and Blue are going to be the predictors. To create the logistic regression models, I used the train() function from the Caret package and set the family to “binomial” and the method to “glm” for generalized linear model. The variable trainControl was also included as one of the parameters in the model to 10-fold cross-validation to obtain the values relatives to the accuracy and kappa statistics.

```{r warning=FALSE}
cl <- makePSOCKcluster(no_cores) 
registerDoParallel(cl)

glm.reg<-train(BlueTarp~Red+Green+Blue,
      data=data,
      family="binomial",
      method="glm",
      trControl=Control)

glm.reg
stopCluster(cl)
```

The model fit summary output indicates that the three predictors all have significant p-values, meaning that they all are contributing to the model.

Since this is a classification problem, the default metrics used by caret to evaluate the performance of the model are accuracy and kappa. Accuracy represents the percentage of correctly classified instances out of all instances. Kappa is like Accuracy expect that its score is normalized. The accuracy of the model is high, 99%. The Kappa score is also high for this model, showing a value of 92%.

**Determining the threshold**

With the function thresholder(), we are able to identify some key statistics for each fold and create a table summarizing these information. 

```{r}
threshold_glm<-thresholder(glm.reg,threshold = seq(0.05, 0.95, by = 0.05),statistics = "all")
threshold_glm$FNR <- 1 - threshold_glm$Sensitivity
threshold_glm$FPR <- 1 - threshold_glm$Specificity
colnames(threshold_glm)
threshold_glm %>% 
  select("prob_threshold", "Accuracy", "FNR", "FPR", "F1","Recall","Precision") %>%
  knitr::kable(digits=2,"pipe",align="c",caption = "Table 1. Table of threshold values for the regression model.") 
```

According to the table above (Tab.1) , the highest accuracy threshold for the logistic regression model was 0.05. The accuracy of the model is 99%. The False Negative Rate (FNR), the objects that were classified as non blue tarp by the model when they are in reality blue tarp, and the False Positive Rate (FPR), the objects that were classified as blue tarp by the model when they are not, are low.
Recall is the ratio of correctly predicted positive observations to the all observations in actual class - yes.
F1 Score is the weighted average of Precision and Recall providing info on the predictive performance of the model. This score takes both false positives and false negatives into account.
Both the Recall and the F1 Score for the logistic model with threshold of 0.05 are high.This is furthermore confirmed by the fact that a high area under the curve represents both high recall and high precision, where high precision relates to a low false positive rate, and high recall relates to a low false negative rate.

```{r}
selected_glm<-threshold_glm[2:9]%>%slice_max("Sensitivity")
selected_glm
```
**Confusion Matrix**
```{r}
set.seed(1)
glm_prob <- predict(glm.reg, newdata=data , type = "prob")
glm.thresh <- 0.05
glm.pred_thresh <- as.factor(ifelse(glm_prob$Yes>glm.thresh,
                                       "Yes", "No"))
glm_confusion <- confusionMatrix(factor(glm.pred_thresh),
                                    data$BlueTarp, 
                                    positive = "Yes") 
glm_confusion
```
**ROC curve**

The ROCurve() function was created to make ROC curve plotting easier for each model. This function takes in the model, the statistics of the selected “optimal” model, and a string to add to the title of the plot. 

```{r fig.cap = "Figure 4. ROC curve logistic regression model."}
ROCurve<-function(model,stats.selected, name){
       prob <- model$pred[order(model$pred$rowIndex),]
       rates <- prediction(prob$Yes,as.numeric(data$BlueTarp))
       roc <- performance(rates, measure="tpr", x.measure ="fpr")
       plot(roc, colorize=T,print.cutoffs.at=c(0, 0.1, 0.9, 1.0),
            main=paste("ROC Curve", name))
       lines(x=c(0,1),y=c(0,1),col="grey")
}
```

```{r fig.cap="Figure 4. ROC curve logistic regression model."}
ROCurve(glm.reg, selected_glm, "Logistic regression")
```

In Fig. 4 we use the ROC curve to visually represent the classification abilities of a model, plotting the TPR against the FPR at numerous threshold values. The ROC curve for the selected logistic regression shows that this model presented a near perfect fit. 
I used the ROCurve() function to plot the ROC curve and then I proceeded with calculating the AUROC value of the model. The AUROC for this model was 99% indicating that our model is able to distinguish between the classes almost perfectly.  

```{r}
prob <- glm.reg$pred[order(glm.reg$pred$rowIndex),]
rates <- prediction(prob$Yes,as.numeric(data$BlueTarp))
auc_glm<-performance(rates, measure = "auc") 
auc_glm@y.values
```
