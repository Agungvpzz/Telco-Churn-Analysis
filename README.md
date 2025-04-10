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
### Model Performance Summary
- We have developed three distinct models, each tailored with different optimization parameters: one focused on accuracy, one on balance, and one on recall.
- This segmentation enables us to strategically select the most suitable model to address specific business needs
![image](https://github.com/user-attachments/assets/d920a113-1d53-4f62-8faf-fad1d7436bd8)


### Model Evaluation and Interpretation
![image](https://github.com/user-attachments/assets/b010a31c-053b-4b05-815c-ca3e994d5ef9)
- The modest differences between CV, training and test metrics suggest minimal overfitting.
- The consistency of accuracy, balance, and ROC AUC across training, testing, and CV phases indicates that the models are well-calibrated and generalize effectively.
- The consistently high ROC AUC values (all above 0.84) across all evaluation methods indicate a strong ability of the models to differentiate between classes.
- **Optimized Recall Model – Prioritizing Customer Retention**
  - Performance Snapshot:
    - CV Accuracy: 74.4%
    - CV ROC AUC: 84.5%
    - CV Recall: 81.8%
  - This model aims to capture as many potential churners as possible — even at the cost of some false positives (predicting a customer will churn when they wouldn’t).
  - Why It’s Valuable:
    - Every customer has the potential to churn — even the seemingly loyal ones.
    - False positives (offering promotions to loyal customers) may still enhance satisfaction and loyalty.
    - High recall ensures very few at-risk customers are missed.
  - Best Fit For:
    - Businesses where acquiring new customers is costly.
    - Brands focused on customer lifetime value and retention.
    - Markets with high competition and many product/service alternatives.
- **Optimized Accuracy Model – Overall Prediction Quality**
  - Performance Snapshot:
    - CV Accuracy: 80.9%
    - CV ROC AUC: 84.9%
  - Accuracy is high, indicating strong overall predictive performance. But, this high accuracy driven by correctly predicting the majority (non-churn) class. Thus, the model doesn’t really help prevent churn.
- **Balanced Model – Middle-Ground Strategy**
  - This model attempts to treat both classes equally, balancing recall and precision without strongly favoring either. However, its recall is lower, which may reduce its effectiveness in proactively identifying churners.
  - Best Fit For:
    - Businesses that can tolerate some customer churn.
    - Scenarios where profit margins are low, and mass retention efforts are not economically justified.

#### Confusion Matrix
![Confusion Matrix](https://github.com/user-attachments/assets/b0287377-15d1-4596-8fe6-952dfd70d3f7)
- Using the recall model, we were able to predict churn for 461 customers, whereas the other models missed over 100 customers.
- The goal is to reduce churn, thus the recall-optimized model is the most effective, even at the cost of some incorrect targeting. It aligns best with customer-centric retention strategies, especially in competitive or high-value customer markets.

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
Below is a detailed recommendation for model selection tailored to two different scenarios:

### General Case: Maximizing Accuracy
For the general scenario where our primary objective is to achieve the highest possible accuracy in predicting both churned and non-churned customers, we recommend utilizing the accuracy-optimized model.

### Specific Case: Cost-Sensitive Prediction
In scenarios where the cost of acquiring new customers significantly outweighs the cost of retaining existing ones, our focus shifts toward optimizing for customer retention. In such cases, we recommend using recall-optimized model as the primary model. By prioritizing recall, we ensure that our model is sensitive to customers who are likely to churn, allowing us to take proactive measures to retain them. This approach helps in maximizing the return on investment by focusing on customer retention efforts.

### Summary of Recommendations
- General Case: Use Adaptive Boosting (AdaBoost) for its superior accuracy and robust performance across diverse data sets.
- Specific Case (Cost-Sensitive): Use Logistic Regression to achieve high recall, particularly when customer acquisition costs are a significant concern.

