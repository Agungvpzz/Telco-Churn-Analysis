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

### Feature Correlations Against Churn
![corr_churn_features](https://github.com/Agungvpzz/Telco-Churn-Analysis/assets/48642326/cf57de7d-d9dc-4884-967a-89bcf009afcd)
![corr_churn_features_grouped](https://github.com/Agungvpzz/Telco-Churn-Analysis/assets/48642326/131fed9d-581a-4f29-ad73-0dc805728b0e)

### Categorical Features Composition in Relation to Churn
![categorical_features_compositions_by_churn](https://github.com/Agungvpzz/Telco-Churn-Analysis/assets/48642326/783d24ca-16a9-471c-9c57-d5a3786219f2)

### Numerical Features Distributions in Relation to Churn
![numerical_distributions_against_churn](https://github.com/Agungvpzz/Telco-Churn-Analysis/assets/48642326/406bfbf8-a4eb-4f8d-9ac7-bdbc374ab6d8)




## 8. Conclusion

## 9. Recommendation


