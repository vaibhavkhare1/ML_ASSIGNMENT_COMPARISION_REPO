# ML Classification Model Comparision for Portuguese Bank Data
## Problem Statement
The primary objective of this project is to develop and evaluate various machine learning models to accurately predict whether a client will subscribe to a term deposit, based on the data from a direct marketing campaign conducted by a Portuguese banking institution. This prediction will enable the bank to optimize its marketing strategies, reduce campaign costs, and improve the overall efficiency of customer outreach.

Requirements for project:
  1. Implement 6 Machine Learning Models:
     * Logistic Regression
     * Desicion Tree Classifier
     * K-Nearest Neighbour Classifier
     * Naive Bias Classifier - Gaussian Model
     * Ensemble Model - Random Forest
     * Ensemble Model - XGBoost
       
  2. Evaluate Models using multiple marrix like
     * Accuracy
     * AUC Score
     * Precision
     * Recall
     * F1 Score
     * Matthews Correlation Coeffi cient (MCC Score)
  
  3.  Web Application to Compare the models and predictions
  
  4.  Deploying Models in Streamlit Cloud

## Dataset Description
**Dataset:** Portuguese Bank Marketing From UCI Machine Repository

**Source:** [UCI ML Repository - Portuguese Bank Marketing](https://archive.ics.uci.edu/dataset/222/bank+marketing)

**Description:** 
   The data is related with direct marketing campaigns of a Portuguese banking institution. 
   The marketing campaigns were based on phone calls. Often, more than one contact to the same client was required, 
   in order to access if the product (bank term deposit) would be (or not) subscribed. 

 **Dataset Dimensions**:
 * bank_train: Contains 45211 rows and 17 columns.
 * bank_test: Contains 4521 rows and 17 columns.

 **Features**:
  The dataset comprises both numerical and categorical features:
  * **Numerical Features**: 'age', 'balance', 'day', 'duration', 'campaign', 'pdays', 'previous'.
     * **age**: Age of the client.
     * **balance**: Average yearly balance of the client, in euros.
     * **day**: Last contact day of the month.
     * **duration**: Last contact duration, in seconds (highly influential, but should be removed for realistic prediction as it's known only after the call).
     * **campaign**: Number of contacts performed during this campaign and for this client.
     * **pdays**: Number of days that passed after the client was last contacted from a previous campaign (-1 means client was not previously contacted).
     * **previous**: Number of contacts performed before this campaign and for this client.
 
 *   **Categorical Features**: 'job', 'marital', 'education', 'default', 'housing', 'loan', 'contact', 'month', 'poutcome'.
        *   **job**: Type of job (e.g., admin., blue-collar, entrepreneur, management, etc.).
        *   **marital**: Marital status (e.g., married, divorced, single).
        *   **education**: Level of education (e.g., primary, secondary, tertiary, unknown).
        *   **default**: Has credit in default (yes/no).
        *   **housing**: Has housing loan (yes/no).
        *   **loan**: Has personal loan (yes/no).
        *   **contact**: Contact communication type (e.g., cellular, telephone, unknown).
        *   **month**: Last contact month of year (e.g., jan, feb, mar, ..., nov, dec).
        *   **poutcome**: Outcome of the previous marketing campaign (e.g., failure, other, success, unknown).

*   **Target Variable**:
    The target variable is 'y', which indicates whether the client subscribed to a term deposit. It originally contained 'yes' or 'no' values, which were subsequently mapped to **1** and **0** respectively for model training.

*   **Descriptive Statistics (bank_train)**:
    The numerical features in the bank_train dataset have the following key statistics:
    *   **age**: Mean age is approximately **40.94** years, with a standard deviation of **10.62**. Ages range from **18** to **95**.
    *   **balance**: The average balance is around **1362.27** euros. Notably, the minimum balance is **-8019.00**, indicating some clients have negative balances, and the maximum is **102127.00**, showing a wide range.
    *   **day**: The average contact day is around the **15th-16th** of the month.
    *   **duration**: The mean duration of the last contact is **258.16** seconds. The maximum duration is **4918** seconds. This feature was identified as highly predictive but also problematic for real-time predictions.
    *   **campaign**: On average, clients were contacted **2.76** times during the campaign, with a maximum of **63** contacts.
    *   **pdays**: The mean pdays is **40.20**, but the median is **-1**, indicating a large number of clients were not previously contacted.
    *   **previous**: The average number of previous contacts is **0.58**, with a maximum of **275**.

*   **Missing Values**:
    Both **bank_train** and **bank_test** datasets were confirmed to have **no missing values** across any of their columns, making them ready for immediate use after appropriate encoding and scaling.


## ML Models Used for this Project

### Model Comparision Table with Evaluation Matrix

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
| --- | --- | --- | --- | --- | --- | --- |
| Logistic Regression Model | 0.902 | 0.8966 | 0.8868 | 0.902 | 0.8878 | 0.4169 |
| Decision Tree Classifier | 0.9257 | 0.9205 | 0.9196 | 0.9257 | 0.9207 | 0.5972 |
| K-Nearest Neighbors Classifier | 0.9162 | 0.9526 | 0.9068 | 0.9162 | 0.9058 | 0.5178 |
| Naive Bayes - Gaussian Classifier | 0.8478 | 0.7926 | 0.8655 | 0.8478 | 0.8557 | 0.3371 |
| Ensemble Model - Random Forest | 0.9573 | 0.9793 | 0.9578 | 0.9573 | 0.9536 | 0.7741 |
| Ensemble Model - XGBoost | 0.9549 | 0.9787 | 0.953 | 0.9549 | 0.9533 | 0.7665 |

### Model Performance Observation

| ML Model Name | Observation about model performance |
| --- | --- |
| Logistic Regression Model | As a linear model, Logistic Regression provided a reasonable baseline (Accuracy: 0.9020, F1: 0.8878) but struggled to capture the full complexity of the data compared to the non-linear models. |
| Decision Tree Classifier | While better than simpler models, its performance (Accuracy: 0.9257, F1: 0.9207) indicates it captures some non-linear patterns but is surpassed by the more complex ensemble methods. |
| K-Nearest Neighbors Classifier | K-Nearest Neighbors showed a good AUC (0.9526) but slightly lower Accuracy (0.9162) and F1 (0.9058) compared to Decision Tree, suggesting it might be sensitive to the feature space or the chosen number of neighbors. |
| Naive Bayes - Gaussian Classifier  | Naive Bayes was the weakest performer among all (Accuracy: 0.8478, F1: 0.8557), suggesting that the assumption of conditional independence between features given the class might not hold strongly for this dataset. |
| Ensemble Model - Random Forest |  Random Forest model emerged as the best performer, achieving the highest accuracy (0.9573), AUC (0.9793), Precision (0.9578), Recall (0.9573), F1-score (0.9536), and MCC (0.7741). This highlights the benefit of combining multiple models or using boosting techniques for better generalization and accuracy. |
| Ensemble Model - XGBoost | XGBoost achieved the second-best performance accuracy (0.9541), AUC(0.9787), Precision (0.953), Recall (0.9549), F1-score (0.9533), and MCC (0.7665). It effectively captures complex patterns and feature interactions. While slightly lower than Random Forest in this case, it shows strong performance with good generalization. The model benefits from gradient boosting's ability to correct errors iteratively. |
