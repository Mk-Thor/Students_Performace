# Import all the Libraries
import pandas as pd
import numpy as np
import warnings
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from statsmodels.formula.api import ols
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

# Ignore the Warnings
warnings.filterwarnings("ignore")

# Read the Dataset
data = pd.read_csv("File Path")

# Make a copy of the dataset
data_copy = data

# Print the first five rows of the dataset
data_copy.head()

# Total no of Rows and Columns 
data_copy.shape

# Datatypes of every columns
data_copy.dtypes

# Check for the null values in every colums
data_copy.isna().sum()

# plot to check the null values places in the columns
sns.heatmap(data_copy.isnull(), cbar = False, cmap = 'viridis')

# Show the plot
plt.show()

# Data Cleaning
data_copy['parental_education_level'].mode()[0] # Checking the most repeated value

# Replace the null values with the most repeated values
data_copy['parental_education_level'] = data_copy['parental_education_level'].fillna(data['parental_education_level'].mode()[0])

# Again check the null values
data_copy.isna().sum()

# describe the dataset
data_copy.describe(include='all')

# Create a pairplot to show the linear relation
sns.pairplot(data_copy)

# Creating the list of Categorical columns
categorical_column = ['gender','part_time_job','diet_quality','parental_education_level',
                      'internet_quality','extracurricular_participation']
					  
					  
# Encode the categorical data with dummy variables
encoded_data = pd.get_dummies(data_copy, columns = categorical_column, drop_first = True)

# Print the first five rows of the encoded data
encoded_data.head()

# Total rows and columns in the encoded data
encoded_data.shape

# Setting X and Y variables
X = encoded_data.drop(columns = ['student_id','exam_score'])
y = encoded_data['exam_score']

# Set the Training and test data (Data size and Random state)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 4)

# Scale the data
scaler = StandardScaler()

# Scale the data
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Deploy regression model
lr_model = LinearRegression()

lr_model.fit(X_train_scaled, y_train)
y_pred_lr = lr_model.predict(X_test_scaled)

print("Linear Regression Results:")
print("R² Score:", r2_score(y_test, y_pred_lr))
print("RMSE:", np.sqrt(mean_squared_error(y_test, y_pred_lr)))

rf_model = RandomForestRegressor(n_estimators = 100, random_state = 4)
rf_model.fit(X_train, y_train)
y_pred_rf = rf_model.predict(X_test)

print("Random Forest Results:")
print("R² Score:", r2_score(y_test, y_pred_rf))
print("RMSE:", np.sqrt(mean_squared_error(y_test, y_pred_rf)))

# Getting most important features
feat_imp = pd.Series(rf_model.feature_importances_, index=X.columns)
feat_imp.nlargest(10).plot(kind='barh')
plt.title("Top 10 Important Features")
plt.show()

data_copy['gender'].value_counts()

data_copy.groupby(['gender','part_time_job'])['study_hours_per_day'].mean()

# Ordinary Least Squares (OLS)
ols_formula = "study_hours_per_day ~ exam_score"

OLS = ols(formula=ols_formula, data = data_copy)

model = OLS.fit()

model_results = model.summary()
model_results

# Regression plot
sns.regplot(x='exam_score', y='study_hours_per_day', data=data_copy)
plt.title("Study Hours vs Exam Score")
plt.show()

residuals = model.resid

# Creating a sub-plot
fig, axes = plt.subplots(1, 2, figsize = (8,4))

# Plotting a Histogram
sns.histplot(residuals, ax = axes[0]);

# Setting Title and X label
axes[0].set_title("Histogram of Residuals")
axes[0].set_xlabel("Resudual Value")

# QQ plot
sm.qqplot(residuals, line = 's', ax = axes[1]);

# QQ plot in axes 1
axes[1].set_title("QQ Plot")

plt.tight_layout()

# Plotting
plt.show()

# Plotting a scatter plot
fig = sns.scatterplot(x = model.fittedvalues, y = model.resid)

# Fixing X label
fig.set_xlabel("FittedValues")

# Fixing Y label
fig.set_ylabel("Residuals")

# Fixing Title
fig.set_title("Fittedvalues VS Residuals")

# Setting Axhline
fig.axhline(0)

# Plotting
plt.show()

model_results

sm.stats.anova_lm(model, typ = 2)

tukey_result = pairwise_tukeyhsd(
    endog = data_copy["study_hours_per_day"],
    groups = data_copy["parental_education_level"],  # must be categorical
    alpha = 0.05
)

print(tukey_result)

tukey_result.summary()
