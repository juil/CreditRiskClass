# CreditRiskClass

Classification of credit risk using logistic regression models.

## Credit Risk Analysis Report 

### Overview of the Analysis

The purpose of this analysis was to build a model that predicts the creditworthiness of borrowers by analyzing **high-risk loans** in historical lending data from a peer-to-peer lending service company.

The analysis aimed to predict `loan_status` as **high** or **low** risk loans based on the characteristics of the loan: loan size, interest rate, borrower's income, debt-to-income ratio, number of accounts, derogatory marks, and total debt.

The machine learning process went through the following basic steps:

1. Split training and testing data.
2. Perform a `Logistic Regression` on the data.
3. Oversample the imbalanced data.
4. Perform another `Logistic Regression` on the oversampled data.

### Results

Using bulleted lists, describe the balanced accuracy scores and the precision and recall scores of all machine learning models.

* Machine Learning Model 1: `LogisticRegression` on Original Data 
  * The accuracy (`iba`) is good but not at the level needed for scoring creditworthiness. 0:`0.89`, 1:`0.87`
  * The precision and recall are perfect for healthy loans. 0:`1.0`, 1:`1.0` 
  * but lacking for high-risk loans, 0:`0.86`, 1:`0.89`

* Machine Learning Model 2: `LogisticRegression` on Oversampled Data
  * The accuracy (`iba`) has greatly improved 0:`0.99`, 1:`0.99`
  * The precision is about the same. 0:`1.0`, 1:`0.86`
  * The recall has improved significantly. 0:`0.99`, 1:`0.99`

### Summary

The `Logstic Regression` machine learning model is very good at labelling healthy loans with low fall positives (precision) and low false negatives (recall), but has some trouble with the imbalanced data when it comes to accuracy.

From a financial perspective, high accuracy in predicting high-risk loans is more important, as mislabeled healthy loans can be adjusted after the fact. However, for the purpose of customer satisfaction and overall good business, it is important to predict healthy loans as well.

By oversampling the data set, the accuracy of predicting high-risk and healhty loans both greatly increases.

## Code 

### Libraries

#### sklearn

```py
from sklearn.metrics import balanced_accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
```

#### imblearn

```py
from imblearn.metrics import classification_report_imbalanced
from imblearn.over_sampling import RandomOverSampler
```

*Starter code provided by edX.*
