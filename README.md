> [!NOTE]
> If you encounter an error with the Jupyter Notebook on GitHub, please use the following links below:<br>
> [1. EDA](https://nbviewer.org/github/Agungvpzz/Telco-Churn-Analysis/blob/main/Telco%20Churn%20EDA.ipynb) <br>
> [2. Predictive Modeling](https://nbviewer.org/github/Agungvpzz/Telco-Churn-Analysis/blob/main/Telco%20Churn%20Predictive%20Modeling.ipynb) <br>
> [3. Model Comparisons using PyCaret](https://nbviewer.org/github/Agungvpzz/Telco-Churn-Analysis/blob/main/Telco%20Churn%20Find%20Best%20Model%20Using%20PyCaret.ipynb)

# Telco-Churn-Analysis

## 1. Introduction
In this repository, I will conduct churn analysis using Python and Plotly for interactive data visualization. The analysis will include examining the correlation of all features with the target variable 'Churn,' assessing the composition of categorical features relative to churn, and evaluating the distribution of numerical features relative to churn. Furthermore, I will perform statistical analysis and predictive modeling using logistic regression and XGBoost algorithms.

## 2. Business Understanding
### A. Business Goals
Churn analysis is a technique used by businesses to understand why customers stop using their products or services, which is often referred to as "churn." The primary goal of churn analysis is to identify patterns and reasons behind customer attrition to take proactive measures to reduce it. Here’s an overview of the key aspects of churn analysis:

### B. Key Questions to Answer
1. <b>Which features are highly correlated to churn</b>: Understanding what are causes of the customers churn.
2. <b>Predict how likely a customer will churn in the future</b>: Informs business to determine which customer should get more attention.
3. <b>Analyze the impact of customer demographics on churn</b>: Identify demographic trends and their influence on customer attrition.

## 3. Data Understanding
The dataset can be explored and downloaded with the following link [telco-customer-churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn/data).

## 4. Methodology
### A. Exploratory Data Analysis (EDA)
1. Conduct a visual inspection of the churn composition using a pie chart.
2. Calculate the Pearson correlation coefficient between each feature and the churn feature to assess their individual relationships, and visualize the results using a bar chart.
3. Conduct chi-squared (χ²) tests to evaluate whether the distribution of churn values differs significantly across the unique categories of each categorical feature, and visualize the results using a bar chart.
4. Conduct Mann-Whitney U tests to determine whether the distribution of a numerical feature differs significantly between churned and non-churned groups, and visualize the results using a line chart.

### B. Predictive Analysis
1. Build predictive models using Logistic Regression and XGBoost algorithms.
2. Assess model performance through various evaluation metrics: classification report, confusion matrix, TPR-FPR, ROC curves, and ROC area curve.

## 5. Data Preparation
### A. Data Cleaning for Exploratory Data Analysis (EDA)
- Decoding feature (for readability purposes)
- Encoding target value
- Replace inconsistency values

### B. Data Preprocessing for Modeling
- Impute outliers by grouping the data based on churn and no-churn values.
- Label Encoding for binary categorical features.
- One-Hot Encoding for categorical features with more than two unique values, and dropping the first category to avoid multicollinearity.
- Transform numerical features using the Power Transformer with the 'yeo-johnson' method to stabilize variance and make the data more Gaussian-like.
- Scaling numerical features using standard scale


## 6. Exploratory Data Analysis

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
- **Tenure**: The tenure value is directly tied to the churn status. For churned customers, tenure stops at the time of churn, resulting in lower values compared to non-churned customers, whose tenure continues to increase as they remain active.
- **MonthlyCharges**: Higher monthly charges are linked to a greater likelihood of churn, as churned customers tend to have higher monthly charges compared to non-churned customers (In general, short-term subscriptions tend to have higher prices compared to long-term subscriptions. Additionally, some customers who choose short-term contracts may do so as a way to test the services before committing to a longer-term plan).
- **TotalCharges**: TotalCharges is cumulative and reflects the combined impact of tenure and monthly charges. Churned customers exhibit lower TotalCharges because their tenure ends at the time of churn. In contrast, non-churned customers continue to accumulate TotalCharges over time, leading to higher overall values.

Overall, the Mann-Whitney U tests confirm significant differences in the distributions of these features between churned and non-churned customers, providing valuable insights for understanding and predicting customer churn.

## Model Development
### Model Characteristics

- Model
    - Handling Imbalanced Data:
        - Use the SMOTE (Synthetic Minority Over-sampling Technique) to balance the target classes.
    - Model Specification:
        - Use the XGBoost Classifier with the following settings:
            - eval_metric='aucpr' (Area Under the Precision-Recall Curve)
            - max_depth=5
            - max_leaves=5
- XGBoost Classifier Scores:
    - train score: 0.8413859745996687
    - test score: 0.7785139611926172
    - cross-val mean: 0.7878296146044624
    - roc-auc 0.8611882545895584


### Model Evaluation
#### Classification Report
![image](https://github.com/Agungvpzz/Telco-Churn-Analysis/assets/48642326/518111af-56ac-41c1-936d-70688eea1aff)

The classification report indicates that our model:
- Overall Accuracy: Achieves an accuracy of 78%.
- Performance on Non-Churn Customers:
    - Accuracy: 84% f1-score.
    - Precision: 90% (higher than recall), indicating that the model tends to assume customers are loyal.
    - Recall: 78%, which, along with the higher precision, suggests an imbalance in the dataset where non-churn cases are more prevalent.
- Performance on Churn Customers:
    - Accuracy: 65% f1-score, indicating weaker performance.
    - Precision: 56% (lower than recall), suggesting the model is cautious in predicting churn, leading to fewer false positives.
    - Recall: 77%, which shows the model identifies more actual churn cases but at the cost of lower precision.

Overall, the model shows good performance in predicting non-churn customers and churn customers

#### Confusion Matrix
![Confusion Matrix](https://github.com/Agungvpzz/Telco-Churn-Analysis/assets/48642326/736ccf53-4f43-4378-8401-cd407273def8)

Our confusion matrix shows the following:
- True Negative (1213), the model predicted negative and the actual was also negative.
- False Positive (339), the model predicted positive but the actual was negative.
- True Positive (432), the model predicted positive and the actual was also positive.
- False Negative (129), the model predicted negative but the actual was positive.

#### TPR-FPR at every Threshold
- True Positive Rate (also known as recall or sensitivity) measures the proportion of true positive cases correctly identified by the model among all actual positive cases. It is calculated as the ratio of true positives to the sum of true positives and false negatives.
- False Positive Rate measures the proportion of false positive cases incorrectly identified as positive by the model among all actual negative cases. It is calculated as the ratio of false positives to the sum of false positives and true negatives. 

![tpr_Fpr_threshold](https://github.com/Agungvpzz/Telco-Churn-Analysis/assets/48642326/ca54887e-5537-4152-8294-6d2e1b004434)

TPR and FPR are essential for evaluating the trade-off between sensitivity and specificity in classification models.
- Increasing the threshold will result in a lower FPR but also a lower TPR.
- Decreasing the threshold will result in a higher TPR but also a higher FPR.
- If we want to give more attention to customers that are likely to churn, we can decrease the threshold.
    - This approach is cost-effective, as providing special attention to customers likely to churn can prevent potential revenue loss.

#### Receiver Operating Characteristic (ROC) Curves
ROC curves are graphical representations of the true positive rate (TPR) versus the false positive rate (FPR) at various threshold settings. While TPR and FPR provide specific performance metrics at particular thresholds, the ROC curve offers a comprehensive visualization of the model's performance across all thresholds, facilitating a better understanding of the trade-offs and overall efficacy.

![roc_curves](https://github.com/Agungvpzz/Telco-Churn-Analysis/assets/48642326/6eda49a6-6a6b-4885-81b7-1f9e59043efe)


#### ROC Area Under Curve
The ROC curve allows for the calculation of the Area Under the Curve (AUC), a single scalar value that summarizes the overall ability of the model to discriminate between positive and negative cases.
A higher AUC indicates better overall performance of the model.

![roc_area_curve](https://github.com/Agungvpzz/Telco-Churn-Analysis/assets/48642326/3c3771b2-4359-4d95-bf5d-3e600757b775)

An AUC score of 0.8612 suggests that our model has strong predictive power and is highly effective at distinguishing between the classes. It reflects the model's robustness and its potential utility in practical applications.


## 8. The Best Model using PyCaret
### Model Comparisons
![image](https://github.com/Agungvpzz/Telco-Churn-Analysis/assets/48642326/e3b1ac88-3f54-4a2b-95e7-dc9b2caacccd)

### Adaptive Boosting (ADA) Model
![image](https://github.com/Agungvpzz/Telco-Churn-Analysis/assets/48642326/25686eac-f511-4db7-8525-89265103b53b)

### Model with Recall Optimization (Logistic Regression)
- Maximize recall score for the positive class (churned customers). 
- We assume that acquiring new customers costs more than retaining existing ones.

![image](https://github.com/Agungvpzz/Telco-Churn-Analysis/assets/48642326/72683d35-33a5-4315-a08b-e144cbc76bb6)


## 8. Conclusion
The analysis reveals several critical factors contributing to customer churn. Key patterns indicate that customers who are more likely to churn typically share the following characteristics:
- Contract Type:
    - Customers with a month-to-month contract are at a significantly higher risk of churning compared to those with longer-term commitments.
    - This suggests that the flexibility of a monthly contract may not foster long-term loyalty.
- Payment Methods:
    - A notable trend is observed among customers who use electronic check payment methods or opt for paperless billing.
    - These payment preferences are correlated with a higher churn rate.
- Demographic Characteristics:
    - Senior Citizens:
        - Older customers, specifically those identified as Senior Citizens, exhibit a higher likelihood of churning.
        - This may be due to factors such as changing service needs or financial considerations.
    - Marital and Family Status:
        - Single Customers (no partner) and have no dependents are more prone to churn.
        - This demographic might be more mobile and less tied down, making them more open to switching providers.


## 9. Recommendation
In our efforts to accurately predict customer churn, it is crucial to select a model that balances high performance with practical considerations specific to our business needs. Below is a detailed recommendation for model selection tailored to two different scenarios:

### General Case: Maximizing Accuracy
For the general scenario where our primary objective is to achieve the highest possible accuracy in predicting both churned and non-churned customers, we recommend utilizing the Adaptive Boosting (AdaBoost) model. This ensemble technique is known for its robust performance and ability to improve predictive accuracy by combining the outputs of multiple weak classifiers to form a strong one. AdaBoost effectively reduces bias and variance, making it an excellent choice for a balanced and accurate prediction model.

### Specific Case: Cost-Sensitive Prediction
In scenarios where the cost of acquiring new customers significantly outweighs the cost of retaining existing ones, our focus shifts toward optimizing for customer retention. In such cases, we recommend using Logistic Regression as the primary model. Logistic Regression offers a solid balance between precision and recall, ensuring that we effectively identify customers who are at risk of churning without compromising the other key metrics.

By prioritizing recall, we ensure that our model is sensitive to customers who are likely to churn, allowing us to take proactive measures to retain them. This approach helps in maximizing the return on investment by focusing on customer retention efforts.

### Summary of Recommendations
- General Case: Use Adaptive Boosting (AdaBoost) for its superior accuracy and robust performance across diverse data sets.
- Specific Case (Cost-Sensitive): Use Logistic Regression to achieve high recall, particularly when customer acquisition costs are a significant concern.

