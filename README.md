> [!NOTE]
> If you encounter an error with the Jupyter Notebook on GitHub, please use this link [nbviewer](https://nbviewer.org/github/Agungvpzz/Telco-Churn-Analysis/blob/main/Telco%20Customer%20Churn%20Analysis.ipynb)


# Telco-Churn-Analysis

## 1. Introduction
In this repository, I will conduct churn analysis using Python and Plotly for interactive data visualization. The analysis will include examining the correlation of all features with the target variable 'Churn,' assessing the composition of categorical features relative to churn, and evaluating the distribution of numerical features relative to churn. Furthermore, I will perform statistical analysis and predictive modeling using logistic regression and XGBoost algorithms.


## 2. Data Understanding
The dataset can be downloaded with the following link [telco-customer-churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn/data).

## 3. Business Goals
Churn analysis is a technique used by businesses to understand why customers stop using their products or services, which is often referred to as "churn." The primary goal of churn analysis is to identify patterns and reasons behind customer attrition to take proactive measures to reduce it. Here’s an overview of the key aspects of churn analysis:

## 4. Objectives

1. <b>Which features are highly correlated to churn</b>: Understanding what are causes of the customers churn.
2. <b>Predict how likely a customer will churn in the future</b>: Informs business to determine which customer should get more attention.
3. <b>Analyze the impact of customer demographics on churn</b>: Identify demographic trends and their influence on customer attrition.
4. <b>Segment customers based on churn risk</b>: Create customer segments to better tailor retention strategies and marketing efforts.

## 5. Methodology
1. Data preparation and cleaning.
2. Feature Encoding
    - Conduct binary encoding for nominal data that consists of only two unique values.
    - Conduct target encoding for ordinal data that consists of more than two unique values.
3. Conduct chi-squared (chi²) tests for each feature against the target feature to determine significant correlations.
4. Build predictive models using Logistic Regression and XGBoost algorithms.
5. Assess model performance through various evaluation metrics: classification report, confusion matrix, TPR-FPR, ROC curves, and ROC area curve.

## 6. Results and Analysis

### Churn Compositions
<div align=center>

  ![image](https://github.com/Agungvpzz/Telco-Churn-Analysis/assets/48642326/aa82f4ce-f1a6-4ca2-8a6e-82fa95c342a6)
</div>

### Features Correlation Against Churn
Feature correlation in the following barplot informs us how each feature correlates to customer churn behaviour.
![corr_churn_features](https://github.com/Agungvpzz/Telco-Churn-Analysis/assets/48642326/cf57de7d-d9dc-4884-967a-89bcf009afcd)

Grouping features below allows for clear churn comparisons among unique values within each feature
![corr_churn_features_grouped](https://github.com/Agungvpzz/Telco-Churn-Analysis/assets/48642326/89748da4-5922-443f-8b67-fdab2e8af5f2)


### Comparison Across All Categorical Features in Relation to Churn
We can clearly compare each value across all categorical features with the help of this barplot below.
![compairson_across_categorical_features](https://github.com/Agungvpzz/Telco-Churn-Analysis/assets/48642326/139e8945-b03f-4cad-b43a-421310db135e)

### Churn Comparison Within Unique Values of Each Feature
- Each feature underwent chi-squared testing to evaluate churn comparisons among unique values
- The subplots are ordered in decreasing order of chi-squared values
- We can clearly identify churn value comparisons within unique values for each feature that significantly differ from other values.

#### Demographics Features Values Comparison by Churn
![categorical_features_demographics_by_churn](https://github.com/Agungvpzz/Telco-Churn-Analysis/assets/48642326/3d974671-eeb4-4128-ad21-66f6d6936805)
- As you can see above, only the 'Gender' feature does not have a significant p-value.
- Customers without dependents are likely to churn.
- Senior citizens tend to churn.
- Customers without partners tend to churn.

#### Payments Features Values Comparison by Churn
![categorical_features_payments_by_churn](https://github.com/Agungvpzz/Telco-Churn-Analysis/assets/48642326/be59ba31-fa1e-49f1-b88a-bc2bf88a5906)
- Customers who have contracts month-to-month are likely to churn.
- Customers with electronic check payment methods are likely to churn.
- Customers using paperless billing tend to churn.

#### Services Features Values Comparison by Churn
![categorical_features_services_by_churn](https://github.com/Agungvpzz/Telco-Churn-Analysis/assets/48642326/aeac8b1e-6778-4da5-a33d-a50ea9d5ff9d)
- Customers who don't subscribe to an additional online security service are likely to churn.
- Customers who don't subscribe to an additional tech support service are likely to churn.
- Customers who subscribe to fiber optic internet service tend to churn.
- Customers who don't subscribe to an additional online backup service are likely to churn.
- Customers who don't subscribe to an additional device protection service are likely to churn.
- Customers who didn't use their internet service to stream movies were likely to churn.
- Customers who didn't use their internet service to stream TV were likely to churn.
- Customers who subscribe to multiple telephone lines with the company tend to churn.
- Overall, customers who didn't subscribe to an internet service tend to be loyal.


### Churn Distributions in each Numerical Feature
The Mann-Whitney U test helps determine if there are significant differences in distribution values between churn values.
![numerical_distributions_against_churn](https://github.com/Agungvpzz/Telco-Churn-Analysis/assets/48642326/406bfbf8-a4eb-4f8d-9ac7-bdbc374ab6d8)
- From the

### Data Modeling
- We utilized Logistic Regression as our base model and compared its performance with XGBoost to determine which model was better.
- Logistic Regression Scores:
    - Training Score: 0.8050
    - Test Score: 0.8132
- XGBoost Scores:
    - Training Score: 0.8239
    - Test Score: 0.7995
- Since the Logistic Regression model has a higher test score compared to XGBoost, we chose to use Logistic Regression for the subsequent analysis.

### Model Evaluation
#### Classification Report
![image](https://github.com/Agungvpzz/Telco-Churn-Analysis/assets/48642326/7cbf7e0c-9d81-4084-9154-2bb617691612)

The classification report indicates that our model:
- Overall Accuracy: Achieves an accuracy of 81% (f1-score).
- Performance on Non-Churn Customers:
    - Accuracy: 88% f1-score.
    - Precision: 90% (higher than recall), indicating that the model tends to assume customers are loyal.
    - Recall: 85%, which, along with the higher precision, suggests an imbalance in the dataset where non-churn cases are more prevalent.
- Performance on Churn Customers:
    - Accuracy: 62% f1-score, indicating weaker performance.
    - Precision: 57% (lower than recall), suggesting the model is cautious in predicting churn, leading to fewer false positives.
    - Recall: 62%, which shows the model identifies more actual churn cases but at the cost of lower precision.
       
Overall, the model shows good performance in predicting non-churn customers but struggles with accurately identifying churn customers, highlighting areas for potential improvement

#### Confusion Matrix
![Confusion Matrix](https://github.com/Agungvpzz/Telco-Churn-Analysis/assets/48642326/9c224b58-2700-4430-aa79-b4c0c3d7d1d9)

Our confusion matrix shows the following:
- True Negative (1159), the model predicted negative and the actual was also negative.
- False Positive (123), the model predicted positive but the actual was negative.
- True Positive (273), the model predicted positive and the actual was also positive.
- False Negative (206), the model predicted negative but the actual was positive.

#### TPR-FPR at every Threshold
- True Positive Rate (also known as recall or sensitivity) measures the proportion of true positive cases correctly identified by the model among all actual positive cases. It is calculated as the ratio of true positives to the sum of true positives and false negatives.
- False Positive Rate measures the proportion of false positive cases incorrectly identified as positive by the model among all actual negative cases. It is calculated as the ratio of false positives to the sum of false positives and true negatives. 

![tpr_Fpr_threshold](https://github.com/Agungvpzz/Telco-Churn-Analysis/assets/48642326/02ddf4fd-9757-4903-a380-43b3ad63996d)

TPR and FPR are essential for evaluating the trade-off between sensitivity and specificity in classification models.
- Increasing the threshold will result in a lower FPR but also a lower TPR.
- Decreasing the threshold will result in a higher TPR but also a higher FPR.
- If we want to give more attention to customers that are likely to churn, we can decrease the threshold.
    - This approach is cost-effective, as providing special attention to customers likely to churn can prevent potential revenue loss.

#### Receiver Operating Characteristic (ROC) Curves
ROC curves are graphical representations of the true positive rate (TPR) versus the false positive rate (FPR) at various threshold settings. While TPR and FPR provide specific performance metrics at particular thresholds, the ROC curve offers a comprehensive visualization of the model's performance across all thresholds, facilitating a better understanding of the trade-offs and overall efficacy.
![roc_curves](https://github.com/Agungvpzz/Telco-Churn-Analysis/assets/48642326/9fd40b28-ae08-47ee-8fd1-fa9f965cdde7)

According to the ROC curves above, setting our model threshold at 0.26 allows us to achieve an 82% True Positive Rate (TPR) while maintaining a 25% False Positive Rate (FPR). However, it's important to note that adjusting the threshold in this way may lead to a reduction in the overall accuracy of our model. This trade-off between TPR and FPR should be carefully considered based on the specific requirements and priorities of the application

#### ROC Area Under Curve
The ROC curve allows for the calculation of the Area Under the Curve (AUC), a single scalar value that summarizes the overall ability of the model to discriminate between positive and negative cases.
A higher AUC indicates better overall performance of the model.
![roc_area_curve](https://github.com/Agungvpzz/Telco-Churn-Analysis/assets/48642326/61203082-60b6-411d-a519-089fd187d845)

An AUC score of 0.86 suggests that our model has strong predictive power and is highly effective at distinguishing between the classes. It reflects the model's robustness and its potential utility in practical applications.

## 8. Conclusion


## 9. Recommendation


