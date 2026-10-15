# Predicting Sales from Campaign Data 📊

## Project Overview

This project focuses on predicting product sales using marketing campaign data and machine learning regression techniques.

The dataset contains campaign-related features such as follower count, engagement rate, advertising spend, content quality, timestamps, and other metadata. The project demonstrates how raw and messy data can be transformed into a machine learning pipeline for sales prediction.

The primary objective is to identify useful sales predictors, compare regression models, tune the best-performing candidate, and generate predictions for unseen test data.

## Objectives

- Clean and preprocess messy real-world-style campaign data.
- Perform Exploratory Data Analysis (EDA).
- Investigate relationships between campaign features and sales.
- Compare multiple machine learning regression models.
- Identify important features using permutation importance.
- Optimize the Gradient Boosting model using RandomizedSearchCV.
- Generate sales predictions for the unseen test dataset.
- Translate model findings into actionable business insights.

## Dataset Description

The dataset consists of separate training and test files.

**Features include:**

| Feature | Description |
|---|---|
| `Followers` | Number of followers associated with a campaign or account |
| `EngagementRate (%)` | Audience engagement rate |
| `AdSpend (GBP)` | Advertising expenditure in GBP |
| `ContentQuality` | Content quality score |
| `Timestamp` | Campaign-related date/time information |
| `Notes` | Categorical metadata |
| `ID` | Record identifier |
| `Sales (Units)` | Target variable available in the training dataset |

**Dataset size:**
- Training data: 8,000 rows
- Test data: 2,000 rows
- Target variable: `Sales (Units)`

*Note: The test dataset does not contain the actual sales labels, so its true prediction accuracy cannot be calculated directly.*

## Technologies Used

- Python
- Pandas and NumPy
- Matplotlib and Seaborn
- Scikit-learn
- Jupyter Notebook / Kaggle Notebook

## Project Workflow

### 1. Data Inspection and Cleaning

- Inspected dataset dimensions, data types, missing values, and duplicates.
- Converted engagement rate and advertising spend into numeric values.
- Parsed timestamps into datetime format.
- Identified invalid and extreme values for data-quality handling.
- Handled missing values using a preprocessing pipeline.

### 2. Feature Engineering

Extracted the following date-based features:

- Year
- Month
- Day of Week

Categorical `Notes` values were imputed and one-hot encoded. Numeric missing values were handled using median imputation.

### 3. Exploratory Data Analysis (EDA)

Analyzed the relationships between sales and campaign features.

Key observations:

- Followers showed the strongest positive linear correlation with sales.
- Content quality showed a modest positive relationship with sales.
- Engagement rate had a relatively weak linear correlation.
- Advertising spend had almost zero linear correlation, although the trained model identified it as an important predictive feature.

### 4. Model Comparison

The following models were evaluated using a train-validation split:

- Dummy Regressor
- Random Forest Regressor
- Extra Trees Regressor
- Gradient Boosting Regressor

Gradient Boosting achieved the best initial validation performance among these candidates.

### 5. Hyperparameter Tuning

RandomizedSearchCV with five-fold cross-validation was used to tune the Gradient Boosting model.

The search explored parameters including:

- Number of estimators
- Learning rate
- Maximum tree depth
- Minimum samples per leaf
- Subsample ratio
- Loss function

The selected configuration used 400 estimators, a learning rate of 0.1, maximum depth of 1, a subsample ratio of 0.7, and Huber loss.

## Model Performance

The tuned model achieved the following validation results:

| Metric | Result |
|---|---:|
| R² Score | 0.4578 |
| RMSE | 2,075.66 units |
| MAE | 1,657.56 units |
| Best 5-Fold CV RMSE | 2,009.59 units |

**Interpretation:**

- The R² score indicates that the model explains approximately 45.78% of the variation in sales on the validation split.
- The MAE indicates an average absolute prediction error of approximately 1,658 units.
- RMSE penalizes larger prediction errors more strongly than MAE.

These are validation results, not measured accuracy on the unlabeled test dataset. R² should not be interpreted as classification accuracy.

## Feature Importance

Permutation importance was calculated using the validation dataset.

| Feature | Importance |
|---|---:|
| Followers | 834.22 |
| AdSpend (GBP) | 554.53 |
| ContentQuality | 98.07 |
| EngagementRate (%) | 23.90 |
| DayOfWeek | 0.07 |
| Year | -0.37 |
| Month | -0.50 |
| Notes | -0.50 |

Higher positive values indicate that shuffling a feature worsened the model's validation performance.

Followers was the most influential feature, followed by advertising spend. Calendar features and Notes showed little measurable predictive contribution in this evaluation.

Permutation importance measures predictive usefulness, not causation. Advertising spend's low linear correlation but relatively high permutation importance suggests that nonlinear relationships or feature interactions may warrant further investigation.

## Business Insights

1. **Audience size:** Follower count was the strongest linear correlate of sales and the most important feature in the model. It may be useful when evaluating campaign opportunities.
2. **Advertising expenditure:** Advertising spend contributed to predictions despite having almost zero linear correlation with sales. Outliers and nonlinear effects should be investigated before drawing business conclusions.
3. **Content quality:** Average sales generally increased across higher content quality scores in the training data. This is an observed association, not proof of causation.
4. **Engagement rate:** Engagement rate had limited predictive importance in this particular model and dataset.
5. **Business application:** Sales predictions can support campaign planning, inventory estimates, and comparisons of expected sales against advertising costs and profit margins.

## Test Predictions

After tuning, the selected model was fitted on the complete training dataset and used to generate predictions for the 2,000 test records.

The predictions were exported to:

`sales_predictions.csv`

The output contains the record `ID` and the predicted sales value.

Actual test performance remains unknown because the test dataset does not include ground-truth sales labels.

## Repository Structure

```text
predicting-sales-from-campaign-data/
│
├── README.md
├── requirements.txt
├── .gitignore
├── predicting_sales.ipynb
└── sales_predictions.csv
```

The notebook filename is a suggested name. Adjust it to match your actual uploaded notebook. Include the prediction CSV only if you want to share the generated output.

## How to Run

1. Clone this repository.
2. Install the required Python libraries.
3. Open the notebook in Jupyter or Google Colab.
4. Update the dataset paths to match your local environment.
5. Run the notebook cells in sequence.

Install dependencies:

```bash
pip install -r requirements.txt
```

## Limitations and Future Improvements

- Evaluate the model on an independent labeled test set when labels become available.
- Investigate extreme values and their impact on model performance.
- Explore nonlinear transformations and feature interactions.
- Compare additional models, such as XGBoost and HistGradientBoostingRegressor.
- Use time-based validation if the data's chronology and prediction objective justify it.
- Add model explainability visualizations and prediction error analysis.
- Evaluate campaign profitability using cost, revenue, and margin data where available.

## Conclusion

This project demonstrates a complete regression workflow, from data cleaning and exploratory analysis to model comparison, hyperparameter tuning, feature importance, and test prediction generation.

The tuned Gradient Boosting model achieved an R² score of 0.4578 on the validation split. The findings highlight follower count and advertising spend as the leading predictive features in the evaluated model.

**Disclaimer:** Predictive associations do not establish causation, and validation metrics do not guarantee equivalent performance on unseen test data.

---

**Author:** Ravi Kushwah

**Skills demonstrated:** Python, Data Cleaning, EDA, Feature Engineering, Scikit-learn, Regression, Model Evaluation, Hyperparameter Tuning, and Machine Learning.
