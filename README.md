# Machine_Learning_Credit_Risk_Analysis
## Project Overview
For this challenge, I am utilizing several models of supervised machine learning on credit loan data in order to predict credit risk. Credit risk is an inherently unbalanced classification problem, as good loans easily outnumber risky loans. I will be using Python, Scikit-learn libraries and several machine learning models to compare the strengths and weaknesses of various machine learning models and determine how well a model classifies and predicts data.

## Results
### 6 Supervised Machine Learning Algorithms
I utlized six different algorithms of supervised machine learning for this analysis. 
The first four algorithms I used are based on resampling techniques and the machine learing alogorithms are designed to deal with various class imbalance. 
After I resample the credit risk data, I ran a Logistic Regression to predict the outcome. Logistic regression predicts binary outcomes. 

It's important to note that the last two models are from the ensemble learning group. The concept of ensemble learning is the process of combining multiple models (i.e. decision tree algorithms) to help improve the accuracy and robustness of the data. Doing so will also decrease the variance of the model, and therefore increase the overall performance of the model.

From the results below, we can see how different machine learning models work on the same dataset. For better interpretation of the results, it's important to understand what each result shows:

- Accuracy score tells us what percentage of predictions the model gets it right. However, it is not enough just to see that results, especially with unbalanced data. 
Equation: accuracy score = number of correct prediction / total number of predictions

- Precision is the measure of how reliable a positive classification is. A low precision is indicative of a large number of false positives. 
Equation: Precision = TP/(TP + FP)

- Recall is the ability of the classifier to find all the positive samples. A low recall is indicative of a large number of false negatives. 
Equation: Recall = TP/(TP + FN)

- F1 Score is a weighted average of the true positive rate (recall) and precision, where the best score is 1.0 and the worst is 0.0. 
Equation: F1 score = 2(Precision * Sensitivity)/(Precision + Sensitivity)

### 1. Naive Random Oversampling and Logistic Regression
For random oversampling, instances of the minority class are randomly selected and added to the training set until the majority and minority classes are balanced.

![Naive_Random_Oversampling](/Images/Naive_Random_Oversampling.png)

Analyzing this supervised machine learning model, we can analyze the imbalanced classification report:
- Accuracy score: 0.64715
- Precision: 0.99
    - For high risk: 0.01
    - For low risk: 1.00
- Recall: 0.67
    - For high risk: 0.62
    - For low risk: 0.67

### 2. SMOTE Oversampling and Logistic Regression
The synthetic minority oversampling technique (SMOTE) is another oversampling approach where the minority class is increased. SMOTE differentiates itself from other oversampling methods in how SMOTE interpolates new instances. For an instance in the minority class, a number of its closest neighbors is chosen. Based on the values of these neighbors, new values are created.

![SMOTE_Oversampling](/Images/SMOTE_Oversampling.png)

- Accuracy score: 0.62515
- Precision: 0.99
    - For high risk: 0.01
    - For low risk: 1.00
- Recall: 0.63
    - For high risk: 0.62
    - For low risk: 0.63

### 3. Cluster Centroids Undersampling and Logistic Regression
Undersampling is essentially doing the opposite method of oversampling. Instead of increasing the number of the minority class like we did for oversampling, we decrease the size of the majority class.

![Cluster_Centroids_Undersampling](/Images/Cluster_Centroids_Undersampling.png)

- Accuracy score: 0.52073
- Precision: 0.99
    - For high risk: 0.01
    - For low risk: 1.00
- Recall: 0.47
    - For high risk: 0.57
    - For low risk: 0.47

### 4. SMOTEENN (Combination of Over and Under Sampling) and Logistic Regression
SMOTEENN is an method of resampling that combines aspects of both oversampling and undersampling. I have oversampled the minority class with SMOTE and cleaned the resulting data with an undersampling strategy 

![SMOTEENN_Combination_of_Over_and_Undersampling](/Images/SMOTEENN_Over_And_Undersampling.png)

- Accuracy score: 0.6481
- Precision: 0.99
    - For high risk: 0.01
    - For low risk: 1.00
- Recall: 0.56
    - For high risk: 0.74
    - For low risk: 0.56

### 5. Balanced Random Forest Classifier
The balanced random forest classifier differentiates itself as a model in how it processess decision trees. Instead of having a single, complex tree like the ones created by decision trees, a random forest algorithm will sample the data and build several smaller, simpler decision trees. Each tree is simpler because it is built from a random subset of features.

![Balanced_Random_Forest_Classifier](/Images/Balanced_Random_Forest_Classifier.png)

- Accuracy score: 0.7885
- Precision: 0.99
    - For high risk: 0.03
    - For low risk: 1.00
- Recall: 0.87
    - For high risk: 0.70
    - For low risk: 0.87

### 6. Easy Ensemble AdaBoost Classifier
In a Easy Ensemble AdaBoost Classifier, a model is trained and then evaluated. After evaluating the errors of the first model, another model is then trained. The model gives extra weight to the errors from the previous model. The purpose of this weighting is to minimize similar errors in subsequent models. This process is repeated until the error rate is minimized.

![Easy_Ensemble_AdaBoost_Classifier](/Images/Easy_Ensemble_AdaBoost_Classifier.png)

- Accuracy score: 0.9316
- Precision: 0.99
    - For high risk: 0.09
    - For low risk: 1.00
- Recall: 0.94
    - For high risk: 0.92
    - For low risk: 0.94

## Summary
#### First 4 Machine Learning Models on Resampling and Logistic Regression
To summarize these results, I have distinguished the first 4 models as they focus on resampling and performing a logistic regression.

From the results above, it's evident the first four models don’t perform well based off the accuracy scores. Those scores are 0.64715, 0.62515, 0.52073, 0.6481 for Naive Random Oversampling, SMOTE Oversampling, Cluster Centroids Undersampling and the SMOTEENN model respectively. This tells us that the models were accurate roughly a bit more than half of the time.

Precision for all four models is 0.01 for high risk loans and 1.00 for low risk loans. Low precision score for high risk loans is due to large number of false positives, meaning that too many of low risk loans were marked as high risk loans. A high score for low risk loans indicates that nearly all low risk scores were marked correctly. However, a lower recall score (i.e. 0.47 for Cluster Centroids Undersampling) indicates that there were quite a few low risk loans that were market as high risk, when they actually weren’t high risk loans. Actual high risk loans have slightly better scores on recall (i.e. 0.57 for Cluster Centroids Undersampling compared to low risk at 0.47) meaning that there weren’t as many false negatives or not too many high risk loans were marked as low risk loans.

#### Last 2 Machine Learning Ensemble Models
The last two models machine learning ensembe models utlizing a classifier both performed better. Their accuracy scores are 0.7885 and 0.9316 for Balanced Random Forest Classifier and Easy Ensemble AdaBoost Classifier respectively. 

Recall scores for both models (both low and high risk scores) were high, indicating very good accuracy for both models (0.87 and 0.94). It's important to note that recall for high risk loans in both models weren't as high as the recall for low risk loans. 

Precision for high risk loans in the Balanced Random Forest Classifier and Easy Ensemble AdaBoost Classifier models came out to 0.03 and 0.08 respectively. This indicates that there were large number of false positives, meaning that large number of low risk loans were marked as high risk. Comparing the high risk recision to the other four models, the other four models all had 0.01 precision for high risk loans. 

#### Personal Recommendation Regarding all the Machine Learning Models
It's evident the first three models didn’t do accurately perform on the test. Therefore, I would recommend not using them in the real-word testing without further fine-tuning. For example, we could train the model on a larger dataset. The other two classifier models showed better results, yet I would use exercise them with caution, since they might be prone to overfitting. If overfitting occurs and we don’t get our desired results when working with a new data set, we can do some further fine-tunning (pruning) to avoid the overfitting.

For all models, utlizing the Easy Ensemble AdaBoost Classifier is the most effective. It provides the highest Score for all risk loans. The precision is clearly worse for all the other models. The Easy Ensemble AdaBoost Classifier accurately represents and can predict the data with an accuracy of 0.9316. Using this model could provide value in determing high and low risk loans and overall loan analysis.