To create the KNN models, I used the train() function. I created a list of k values, from 0 to 50 at intervals of 5, that was used for the tuneGrid parameter in the train() function. The tuneGrid acted as a list of options to train models on and determine from those models the best one. Adding this option made the train() function choose the optimal k-value for the model out of the ones given.

```{r warning=FALSE}
cl <- makePSOCKcluster(no_cores)
registerDoParallel(cl)

start.time <- Sys.time()

set.seed(1)

klist <- data.frame(k=seq(1,20,2))
klist[1,] <- 1

knn.mod <- train(BlueTarp~Red+Green+Blue, data=data,
                  tuneGrid = klist,
                  method="knn",
                  metric="Accuracy",
                  trControl=Control)
knn.mod
stopCluster(cl)
```


```{r fig.cap="Figure 7. Accuracy of KNN model at different k."}
#accuracy of KNN at different K
plot(knn.mod, main="Accuracy of KNN at different k")
```

Based on accuracy, the output of the train() function shows the optimal k value to be 5. Although the accuracy remained above 99% and Kappa above 95% for all k values tested.

**Determining the threshold**

```{r}
threshold_knn <- thresholder(knn.mod,threshold = seq(0.05, 0.95, by = 0.05),statistics = "all")
threshold_knn$FNR <- 1 - threshold_knn$Sensitivity
threshold_knn$FPR <- 1 - threshold_knn$Specificity
threshold_knn %>% 
  select("prob_threshold", "Accuracy", "FNR", "FPR", "F1","Recall","Precision") %>%
  knitr::kable(digits=2,"pipe",align="c",caption = "Table 4. Table of threshold values for the KNN model.") 
```

```{r}
selected_knn<-threshold_knn[1:9]%>%slice_max("Sensitivity")
selected_knn
```
**Confusion matrix**
```{r}
set.seed(1)
knn_prob <- predict(knn.mod, newdata=data , type = "prob")
knn.thresh <- 0.05
knn.pred_thresh <- as.factor(ifelse(knn_prob$Yes>knn.thresh,
                                       "Yes", "No"))
knn_confusion <- confusionMatrix(factor(knn.pred_thresh),
                                    data$BlueTarp, 
                                    positive = "Yes") 
knn_confusion
```


**ROC curve**
```{r fig.cap="Figure 7. ROC curve KNN model."}
#ROCurve(knn.mod, selected_knn)
pred_knn <- predict(knn.mod, data, type='prob')
predob_knn <- ROCR::prediction(pred_knn$Yes, data$BlueTarp, label.ordering=c('No', 'Yes'))
knn.roc <- ROCR::performance(predob_knn, measure='tpr', x.measure='fpr')
plot(knn.roc, colorize=T, print.cutoffs.at=c(0, 0.1, 0.9, 1.0),main="ROC Curve KNN Model")
lines(x=c(0,1), y=c(0,1), col='grey')
```

```{r}
auc_knn<-performance(predob_knn, measure = "auc") 
auc_knn@y.values
```

The ROC curve for the selected KNN model looks very good. The AUROC for the KNN model was 0.9998582.
