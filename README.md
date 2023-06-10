# Case Study Loan Eligibility #
Welcome to the loan eligibility prediction problem! The objective is to determine whether a customer is eligible for a loan or not. Despite having a relatively small database with 614 values and 13 columns, I will leverage the "Loan_Data.csv" database, which has been made available for use. Overall, the notebook's goal is to develop an accurate loan eligibility prediction model by leveraging data analysis, preprocessing techniques, class balancing, and machine learning classification algorithms. The objective is to create a fair and reliable loan approval process that benefits both the institution and the customers.
First Part: To begin, let's analyze the data and process it for further exploration. The dataset contains various attributes that can help us make predictions. We will perform feature engineering, handle missing values, and perform any necessary transformations to ensure the data is suitable for our classification task.

## Data Preparation & Cleaning ##
<img src="/Users/nicorahn/Desktop/Ironhack/Mid_Project_final/1" width = 300  title="Data Set" />
**Here are the key points of the File:**

**Dataset Information**

- The dataset consists of 614 rows and 13 columns.
- The columns in the dataset are: Loan_ID, Gender, Married, Dependents, Education, Self_Employed, ApplicantIncome, CoapplicantIncome, LoanAmount, Loan_Amount_Term, Credit_History, Property_Area, and Loan_Status.
- Some columns have missing values: Gender (601 non-null), Married (611 non-null), Dependents (599 non-null), Self_Employed (582 non-null), LoanAmount (592 non-null), Loan_Amount_Term (600 non-null), and Credit_History (564 non-null).
- The data types of the columns include float64, int64, and object.

### Data Preparation Steps ###

1. Imported the loan dataset from a CSV file.
2. Checked the dataset's shape and data types using the info() and dtypes functions.
3. Removed duplicate rows using the drop_duplicates() function.
4. Standardized the column names by converting them to lowercase and replacing spaces with underscores.
5. Checked the percentage of missing values in each column using isna().sum() and isnull().sum()/len(df).
6. Replaced missing values in the "Gender" column with "unknown" as it doesn't influence the credit decision.
7. Created a list of columns with missing values to be imputed with the median.
8. Iterated over the columns and replaced the missing values with the column's median using the fillna() function.
9. Dropped rows with missing values in the "Married", "Dependents", and "Self_Employed" columns.
I have decided to delete the remaining empty lines, as the respective answers here can be decisive for a credit decision. Therefore, I did not choose to replace the "gender" column with the word "unknown". In total, 50 rows were deleted. 35 rows from the category "Y" and 15 rows from the category "N".

## EDA - Exploratory Data Analysis ##
10. Performed Exploratory Data Analysis (EDA) to analyze the distribution of numerical columns and visualize relationships between categorical columns and the loan status.
<img src="/Users/nicorahn/Desktop/Ironhack/Mid_Project_final/2" width = 300  title="distribution of numerical columns" />
11. Calculated the average income for each loan status category and created bar plots to compare them.
<img src="/Users/nicorahn/Desktop/Ironhack/Mid_Project_final/3" width = 300  title="Education vs Loan_Status" />
Education vs Loan_Status:
This table compares the education level ("Graduate" or "Not Graduate") with the loan status ("Y" for eligible or "N" for not eligible).
We can observe that among the individuals with a "Graduate" education, a higher number (314) were eligible for a loan compared to those who were not eligible (129).
On the other hand, among individuals with a "Not Graduate" education, the numbers are lower, with 76 being eligible and 48 being not eligible.
This suggests that having a graduate education may positively influence the chances of being eligible for a loan.

<img src="/Users/nicorahn/Desktop/Ironhack/Mid_Project_final/4" width = 300 title="DProperty_Area vs Loan_Status" />
Property_Area vs Loan_Status:
This table examines the property area ("Rural", "Semiurban", or "Urban") in relation to the loan status.
In the "Rural" area, the number of individuals eligible for a loan (102) is higher than those who are not eligible (65).
In the "Semiurban" area, the trend is even more pronounced, with a significantly higher number of individuals (167) being eligible compared to those who are not eligible (50).
In the "Urban" area, the numbers are more balanced, with 121 individuals eligible and 62 not eligible.
This suggests that individuals residing in semiurban areas have a higher chance of being eligible for a loan compared to rural or urban areas.

<img src="/Users/nicorahn/Desktop/Ironhack/Mid_Project_final/5" width = 300 title="Married vs Loan_Status" />
Married vs Loan_Status:
This table examines the marital status ("Yes" or "No") and its relationship with the loan status.
Among individuals who are married ("Yes"), a higher number (268) are eligible for a loan compared to those who are not eligible (104).
Among individuals who are not married ("No"), the numbers are relatively balanced, with 122 individuals eligible and 73 not eligible.
This indicates that being married may increase the likelihood of being eligible for a loan.


### Correlations ###
12. Created contingency tables and bar plots to examine relationships between categorical variables and loan status.
13. Explored correlations between variables using a correlation matrix and histograms.
<img src="/Users/nicorahn/Desktop/Ironhack/Mid_Project_final/6" width = 300 title="Correlations" />

## Modeling (logistic regression)
14. Prepared the data for modeling by splitting it into features (X) and target (y).
15. Applied one-hot encoding to categorical columns using pd.get_dummies() to convert them into numerical format.
Each categorical column has been replaced with binary columns, where a value of 1 indicates the presence of a particular category and 0 indicates its absence. By performing one-hot encoding, we have transformed the categorical variables into a numerical format that can be used by machine learning algorithms. This encoding enables the algorithms to effectively learn from the data and make predictions. The encoded feature variables will be used as inputs to train a logistic regression model for predicting the Loan_Status, which is the target variable.

16. Split the dataset into training and testing data using the train_test_split() function.
17. Trained a logistic regression model on the training data and evaluated its accuracy on the testing data.
18. Visualized the confusion matrix to analyze the model's performance in predicting loan status.
The result of the confusion matrix provides information about the performance of the classification model. The confusion matrix is a 2x2 matrix that represents the count of correctly and incorrectly classified instances for each class.
<img src="/Users/nicorahn/Desktop/Ironhack/Mid_Project_final/7" width = 300  title="Correlationsconfusion_matrix" />


- True Negative (TN): 15 - These are the cases that were actually classified as "N" (loan rejection) and were correctly predicted by the model.
- False Positive (FP): 22 - These are the cases that were actually classified as "N" (loan rejection), but were falsely predicted as "Y" (loan approval) by the model.
- False Negative (FN): 3 - These are the cases that were actually classified as "Y" (loan approval), but were falsely predicted as "N" (loan rejection) by the model.
- True Positive (TP): 74 - These are the cases that were actually classified as "Y" (loan approval) and were correctly predicted by the model.

## Optional: Performed data balancing using oversampling to address any class imbalance issues. ##
19. Repeated the logistic regression model training and evaluation with the balanced dataset.
20. Visualized the updated confusion matrix to compare the model's performance.