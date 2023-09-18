# Disaster-relief-project

In early 2010 Haiti was hit by a magnitude 7.0 earthquake. This catastrophe leveled many buildings, and resulted in numerous lives lost. In the wake of the earthquake, more than 50% of the population at the time, were displaced, with 1.5 million of them living in tent camps. This wide-spread displacement of people across the island, made relief efforts more difficult. 
In this scenario, we are working towards developing a predictive model which will help locating displaced people using RGB data extracted from aerial imagery of the country. It was known that displaced people were using blue tarp to create tent to shelter them. This distinguishable blue color was used to facilitate the identification of displaced people through aerial images.  For this purpose, we have a data set of RGB values and categorical variables denoting the structure types present in the images.  This dataset will be used to develop a model that has a high detection rate regarding true positives in order to maximize its capability to save lives in a precise manner. We will consider the accuracy and the False Negative Rate (FNR) and False Positive Rate (FPR) as metrics to determine the performance of the model selected. In particular the FPR has the potential to draw false conclusion and to jeopardize the rescue efforts in a context where time is the essence in terms of saving lives. 

# Project details

The objective of this disaster relief initiative is to assess the performance of various algorithms when applied to imagery data collected during the 2010 earthquake relief efforts in Haiti. The primary aim is to identify the most accurate and timely method for locating as many displaced individuals as possible, based on the imagery data. 
This project evaluates the effectiveness of several algorithms, including Logistic Regression, Linear Discriminant Analysis (LDA), Quadratic Discriminant Analysis (QDA), K-Nearest Neighbor (KNN), Penalized Logistic Regression, Random Forest, and Support Vector Machines (SVM). The evaluation is carried out using 10-fold cross-validation and a separate hold-out dataset. The dataset used for this analysis is sourced from the 'HaitiPixels.csv' file, which consists of three columns: 'Class', 'Red', 'Green', and 'Blue'. The 'Red', 'Green', and 'Blue' columns contain pixel data collected at various locations, while the 'Class' column categorizes objects found in the images, classifying them as 'vegetation', 'soil', 'rooftop', 'various non-tarp', or 'blue tarp'. In this project, the 'blue tarp' category from the 'Class' column serves as the response or outcome variable, while 'Red', 'Green', and 'Blue' are used as predictor variables. This dataset is utilized to assess the performance of the five different algorithms and identify the most effective one for predicting the presence of blue tarps.



