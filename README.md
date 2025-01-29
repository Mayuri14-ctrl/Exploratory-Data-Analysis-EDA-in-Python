# Exploratory-Data-Analysis-EDA-in-Python
Perform basic operations on data using Python

## 1. Introduction
Exploratory Data Analysis (EDA) is a crucial step in understanding the dataset before building models. It helps in discovering patterns, detecting outliers, and summarizing key characteristics of the data.

## 2. Importing Libraries
```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
```
## Loading Dataset
```
df = pd.read_csv("your_dataset.csv")
df.head()
```
## 4. Checking for Missing Values
```
df.isnull().sum()
# Display only columns with missing values
missing_columns = missing_values[missing_values > 0]
print(missing_columns)
```
## 5. Handling for Missing Values
```
# Convert blank values to NaN
df.replace('', np.nan, inplace=True)

# Convert all possible columns to numeric (forcing errors='coerce' converts non-convertible values to NaN)
df = df.apply(pd.to_numeric, errors='coerce')

# Fill NaN values with 0
df.fillna(0, inplace=True)
```
## 6. Feature Encoding

```
df['category'] = df['category'].astype('category').cat.codes

```
##7 # Exploratory-Data-Analysis-EDA-in-Python
Perform basic operations on data using Python

## 1. Introduction
Exploratory Data Analysis (EDA) is a crucial step in understanding the dataset before building models. It helps in discovering patterns, detecting outliers, and summarizing key characteristics of the data.

## 2. Importing Libraries
```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
```
## Loading Dataset
```
df = pd.read_csv("your_dataset.csv")
df.head()
```
## 4. Checking for Missing Values
```
df.isnull().sum()
# Display only columns with missing values
missing_columns = missing_values[missing_values > 0]
print(missing_columns)
```
## 5. Handling for Missing Values
```
# Convert blank values to NaN
df.replace('', np.nan, inplace=True)

# Convert all possible columns to numeric (forcing errors='coerce' converts non-convertible values to NaN)
df = df.apply(pd.to_numeric, errors='coerce')

# Fill NaN values with 0
df.fillna(0, inplace=True)
```
## 6. Feature Encoding
```
df['category'] = df['category'].astype('category').cat.codes

```
## 7. Handling Duplicates

```
# Check for duplicate rows
df.duplicated().sum()

# Remove duplicate rows
df = df.drop_duplicates()
```

## 8. Handling Outliers
```
Using IQR (Interquartile Range)
Q1 = df.quantile(0.25)
Q3 = df.quantile(0.75)
IQR = Q3 - Q1

# Remove outliers
df = df[~((df < (Q1 - 1.5 * IQR)) | (df > (Q3 + 1.5 * IQR))).any(axis=1)]
```
## 9. Normalization & Standardization
Min-Max Scaling (Normalization)

```
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
df[['column_name']] = scaler.fit_transform(df[['column_name']])
```
## Standardization (Z-Score)
```
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
df[['column_name']] = scaler.fit_transform(df[['column_name']])
```
## 10. Feature Engineering
Creating New Features
```
df['new_feature'] = df['feature1'] * df['feature2']
Extracting Date Features
df['date'] = pd.to_datetime(df['date_column'])
df['year'] = df['date'].dt.year
df['month'] = df['date'].dt.month
df['day'] = df['date'].dt.day
```
## 11. Binning (Converting Continuous to Categorical)
```
df['age_group'] = pd.cut(df['age'], bins=[0, 18, 35, 50, 100], labels=['Teen', 'Young Adult', 'Adult', 'Senior'])

```
## 12. Correlation Analysis
```
correlation_matrix = df.corr()
```
## 13. Plot heatmap
```
plt.figure(figsize=(10, 6))
sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm")
plt.show()
```
## 14. One-Hot Encoding for Categorical Variables
```
df = pd.get_dummies(df, columns=['category_column'], drop_first=True) 
```

## 15. Conclusion
Exploratory Data Analysis (EDA) is an essential step in understanding data before building machine learning models. This guide covered various transformations, including handling missing values, outlier detection, feature engineering, and data normalization. Implementing these techniques will help you preprocess data effectively and improve model performance.

If you found this helpful, consider starring ⭐ the repository!

