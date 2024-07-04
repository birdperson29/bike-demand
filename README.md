
# Project Report

## Overview
This project is about building a machine learning model to predict certain outcomes based on data. This report explains the steps we took, including preparing the data, choosing and setting up models, and making those models work better.

## Data Preparation and Understanding Data

### Checking Data Distribution
I first looked at how data points are spread across different features. This helps us understand our data better and see if there are any unusual points. This was done by plotting a histogram of all features 

```python
df_train.hist(bins=50, figsize=(20, 15))
plt.tight_layout()
plt.show()
```

### Reducing Data Complexity with PCA
I used a method called PCA to reduce the number of data features to the most important ones. This makes our model faster and still effective. PCA is a dimensionality reduction technique. 

```python
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

features = df_train.drop(['count', 'datetime'], axis=1)
scaler = StandardScaler()
features_scaled = scaler.fit_transform(features)

pca = PCA(n_components=features.shape[1])
pca.fit(features_scaled)
```

## Choosing and Setting Up Models
We used Autogluon for this purpose. 
### Initial Model Setup
I first used models with default settings to see how well they perform without any changes.

### Using AutoGluon for Model Setup
AutoGluon is a tool that helps us automatically find the best model and settings. It tries out many different models and settings for us.

#### Models Tested by AutoGluon
AutoGluon looked at different models including:
- Decision Trees
- Random Forest
- LightGBM
- Neural Networks

### Making Models Work Better

#### General Settings
I adjusted settings like how fast the model learns, how many decision points it considers, and how it avoids overfitting. This was done by setting specific hyperparameters. 

Autogluon found that Weighted Ensemble L2 produced the best results, but Kaggle scores favoured LightGBM, closely followed by the Weighted Ensemble L2. 

#### Random Forest Settings
I focused more on a model called Random Forest because it is generally promising. I changed settings like the number of decision trees, how deep these trees can grow, and how many samples they consider before making a decision. (Hyperparameter tuning)

```python
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV, KFold

param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

kf = KFold(n_splits=5, shuffle=True, random_state=0)
grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=kf, scoring='neg_mean_squared_error', verbose=2, n_jobs=-1)
grid_search.fit(X_train, y_train)

best_rf = grid_search.best_estimator_
y_pred = best_rf.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print("Best parameters found: ", grid_search.best_params_)
```

## Results and Conclusion
We checked how good our models are using measures like accuracy or error rate. AutoGluon really helped us find and set up the best model quickly. The Random Forest model did really well after we adjusted its settings.
