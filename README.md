Project Definition:

This project investigates the relationship between students' daily habits and their academic performance, particularly their exam scores. Using a real-world dataset containing lifestyle, behavioral, and academic attributes of 1000 students, we explore how factors such as study hours, sleep patterns, social media usage, mental health, diet quality, parental education, and extracurricular participation impact a student's academic success.

The project follows a structured data science workflow:

Data Cleaning: Handling missing values and preparing the dataset for analysis.

Exploratory Data Analysis (EDA): Understanding the distributions, patterns, and correlations within the data.

Feature Engineering: Converting categorical variables into numerical formats through one-hot encoding.

Statistical Modeling (OLS Regression): Building a basic linear model to understand simple relationships between key variables.

Machine Learning Modeling: Implementing and evaluating models such as Linear Regression and Random Forest to predict student exam scores more accurately.

Model Evaluation: Using metrics like R² Score and Root Mean Squared Error (RMSE) to assess the performance of each model.

Feature Importance Analysis: Identifying which student habits have the most significant impact on academic outcomes.

Through this project, we not only aim to predict exam scores but also extract meaningful insights about student behavior patterns, helping educators and students make informed decisions to enhance academic performance.

🎯 Key Objectives:
Understand which lifestyle habits correlate with higher academic achievement.

Build predictive models to estimate exam scores based on various student attributes.

Compare traditional statistical modeling (OLS) with machine learning models.

Highlight the most important factors influencing student performance.

🛠 Tools & Libraries Used:
Python (Pandas, NumPy, Seaborn, Matplotlib)

Statsmodels (for OLS Regression)

Scikit-learn (for Machine Learning models)

Data Visualization Techniques

📈 Expected Outcomes:
A ranked list of the most impactful habits on student performance.

Accurate prediction models for exam scores.

Deeper insights into how digital behavior, mental health, and lifestyle choices affect academic outcomes.

----------------------------------------------------------------------------------------------------------------------------------------------------------------------------

Key Steps in the Project

🔍 1. Exploratory Data Analysis (EDA)
Exploratory Data Analysis (EDA) is the process of examining the data to understand its structure, spot patterns, detect anomalies, check assumptions, and test hypotheses.
EDA involves summarizing the main characteristics of the dataset using statistics and visualizations like histograms, pairplots, heatmaps, and boxplots.

✅ Purpose:

Understand distributions and relationships

Identify missing values, outliers, or errors

Guide decisions for cleaning and modeling

🛠 2. Feature Engineering
Feature Engineering is the process of transforming raw data into a format that is better suited for modeling.
It includes creating new features, modifying existing ones, encoding categorical variables, handling missing values, and scaling numeric data to improve model performance.

✅ Purpose:

Make data machine-readable

Highlight important patterns

Improve the predictive power of models

📈 3. Statistical Modeling
Statistical Modeling is the process of creating mathematical representations of real-world data relationships using formulas and statistics.
In this project, Ordinary Least Squares (OLS) Regression is used to model the linear relationship between study_hours_per_day and exam_score.

✅ Purpose:

Quantify and understand relationships between variables

Check assumptions like linearity and normality

Provide interpretable results (e.g., how much exam scores affect study hours)

🤖 4. Machine Learning Models
Machine Learning Models are algorithms that learn patterns from historical data to make predictions or classifications on new, unseen data.
In this project, Linear Regression and Random Forest Regressor models are built to predict students' exam scores based on multiple lifestyle features.

✅ Purpose:

Create predictive systems that generalize well

Capture complex patterns that may not be obvious through simple statistics

Compare different models to find the best performing one

📏 5. Model Evaluation
Model Evaluation is the process of measuring how well a model performs on unseen or test data.
It involves using metrics like R² Score (how much variance is explained) and Root Mean Squared Error (RMSE) (average prediction error) to judge the model’s accuracy.

✅ Purpose:

Ensure that the model is not underfitting or overfitting

Quantify the model’s predictive power

Compare multiple models based on performance metrics


-------------------------------------------------------------------------------------------------------
Step                 | What It Means                               | Purpose                          |
---------------------|---------------------------------------------|----------------------------------|
EDA                  | Explore and visualize the data              | Understand and clean data        |
Feature Engineering  | Transform data for modeling                 | Improve model performance        |
Statistical Modeling | Build interpretable mathematical models     | Find real-world relationships    |
Machine Learning     | Train models to predict outcomes            | Capture complex patterns         |
Model Evaluation     | Measure model success                       | Ensure reliability and accuracy  |
-------------------------------------------------------------------------------------------------------




