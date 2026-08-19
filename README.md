# Chemical Process Yield Prediction

## Overview

This project was developed as part of an ML Hackathon. The objective was to build a Machine Learning regression model to predict the overall yield of a chemical process based on its operating conditions.

During the initial analysis, we found that the input features had complex and highly nonlinear relationships with `overall_yield`. The relationships were not straightforward, which made it difficult for the models to capture the underlying patterns using only the original features. Therefore, feature engineering became one of the most important parts of the project.

## Problem Statement

The objective is to develop a Machine Learning regression model that can predict `overall_yield` from the available process parameters and generalize well to unseen data.

The primary evaluation metric used was Root Mean Squared Error (RMSE), where a lower value indicates better predictive performance.

## Dataset

The dataset contains process-related variables describing the operating conditions of the chemical process.

### Original Input Features

- `flow_rate_L_min`
- `concentration_mol_L`
- `inlet_temperature_K`
- `length_m`
- `jacket_temperature_K`

### Target Variable

- `overall_yield`

## Challenges Faced and How We Addressed Them

### 1. Complex Nonlinear Relationships

The original features showed complex and nonlinear relationships with `overall_yield`.

**Solution:** We used tree-based regression models and domain-inspired feature engineering to capture these relationships more effectively.

### 2. Poor Initial Model Performance

The initial models using the original features did not achieve the expected performance.

**Solution:** We explored meaningful derived features instead of relying only on the original variables.

### 3. Feature Engineering

It became clear that feature engineering would be important for improving model performance.

**Solution:** We developed features related to residence time, thermal exposure, inverse temperature, and kinetic behaviour.

### 4. Too Many Feature Experiments

Multiple engineered features and feature combinations made it difficult to identify the most useful ones.

**Solution:** We compared different feature combinations using cross-validation and RMSE to identify the most effective feature set.

### 5. Difficult Feature-Target Relationships

Visualizations did not always reveal clear patterns because of the nonlinear nature of the data.

**Solution:** We combined visualization with quantitative model evaluation instead of relying only on visual relationships.

### 6. Model Selection

Different regression models produced different levels of performance.

**Solution:** We compared Random Forest, Extra Trees, Gradient Boosting, and HistGradientBoosting using the same 5-fold cross-validation setup.

### 7. Hyperparameter Tuning

Model performance depended on several hyperparameters.

**Solution:** We tuned parameters such as `n_estimators`, `max_depth`, `min_samples_leaf`, `min_samples_split`, and `max_features`.

### 8. Fold-to-Fold Variability

RMSE varied across different cross-validation folds.

**Solution:** We used 5-fold cross-validation and considered both mean RMSE and standard deviation.

## Feature Engineering

Since the original features were not sufficient to capture the complex relationships in the data, several domain-inspired features were explored.

The final feature set used by the best-performing model includes:

- `flow_rate_L_min`
- `concentration_mol_L`
- `inlet_temperature_K`
- `length_m`
- `jacket_temperature_K`
- `residence_time_proxy`
- `thermal_exposure`
- `inv_T`
- `kinetic_exposure_proxy`

These engineered features were designed to capture process-related effects such as residence time, thermal exposure, inverse temperature behaviour, and temperature-dependent kinetics.

## Models Tested

We experimented with multiple regression models:

- Random Forest Regressor
- Extra Trees Regressor
- Gradient Boosting Regressor
- HistGradientBoostingRegressor

All models were evaluated using 5-fold cross-validation with RMSE as the primary evaluation metric.

## Model Performance

The initial Random Forest model achieved a mean 5-fold cross-validation RMSE of **20.2475**.

After feature engineering and model tuning, the Extra Trees Regressor achieved a mean CV RMSE of **18.0965**, giving a clear improvement over the initial model.

| Model / Stage | Mean CV RMSE |
|---|---:|
| Initial Random Forest | 20.2475 |
| Final Extra Trees | **18.0965** |

RMSE measures the average magnitude of prediction errors, with larger errors receiving greater weight. Therefore, the reduction from **20.2475 to 18.0965** indicates improved predictive performance after feature engineering and model optimization.

### Final Model Configuration

The final Extra Trees model used the following configuration:

~~~python
ExtraTreesRegressor(
    n_estimators=400,
    max_depth=10,
    min_samples_split=2,
    min_samples_leaf=1,
    max_features=0.8,
    bootstrap=False,
    random_state=42,
    n_jobs=-1
)
~~~

## Cross-Validation Results

Using **5-fold cross-validation**, the final model produced:

### Fold RMSE

~~~text
[18.0991, 18.2961, 22.2184, 14.7291, 17.1398]
~~~

### Mean CV RMSE

**18.0965**

### Standard Deviation

**2.4201**

The relatively low standard deviation indicates that the model's performance remained reasonably consistent across the five folds.

## Project Structure and Notebooks

The project contains both the final Jupyter Notebook and the collaborative Google Colab Notebook, along with the datasets and submission file.

~~~text
chemical-process-yield-prediction/
│
├── ml_hackathon_jupyter_notebook.ipynb
├── ml_hackathon_collab_notebook.ipynb
├── train_dataset.csv
├── test_dataset.csv
├── submission.csv
├── README.md
└── .gitignore
~~~

### Notebook Description

**`ml_hackathon_jupyter_notebook.ipynb`**

Contains the final analysis, feature engineering, model training, hyperparameter tuning, cross-validation, and final model evaluation.

**`ml_hackathon_collab_notebook.ipynb`**

Contains the collaborative development process, including initial data exploration, visualizations, feature experiments, and model experiments.

## Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- Jupyter Notebook
- Google Colab
- Matplotlib

## Team

This project was developed collaboratively as part of the **ML Hackathon**.

### Team Members

- **Shivam Ahirwar** — [GitHub Profile](https://github.com/ahirwarshivam)
- **Sudhanshu Pandey 2** — [GitHub Profile](#)
- **Priyanshu Gupta 3** — [GitHub Profile](https://github.com/Priyanshu-iitkgp)

## Future Improvements

Further improvements can be explored through:

- Additional domain-inspired feature engineering
- Feature selection
- Ensemble techniques
- Systematic hyperparameter optimization
- Advanced regression models
- Model interpretability and error analysis

These approaches may help further reduce the cross-validation RMSE and improve the model's generalization performance.

## Conclusion

The project demonstrates an end-to-end machine learning workflow for **chemical process yield prediction**, from data exploration and feature engineering to model development, hyperparameter tuning, and cross-validation.

The final Extra Trees model achieved a **mean cross-validation RMSE of 18.0965** across 5 folds, improving upon the initial Random Forest RMSE of **20.2475**.
