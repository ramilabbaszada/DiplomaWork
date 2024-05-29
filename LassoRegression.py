import pandas as pd
import numpy as np
from sklearn.linear_model import LassoCV
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

# Load the dataset
data_path = 'Final_Dataset.csv'
data = pd.read_csv(data_path)

# Define the target variables and their respective features
targets_and_features = \
    {
        'RH2M_next': ['QV2M', 'RH2M', 'T2M', 'T2MDEW', 'T2MWET', 'TS', 'T2M_MAX', 'T2M_MIN'],
        'TS_next': ['MO', 'QV2M', 'RH2M', 'T2M', 'T2MDEW', 'T2MWET', 'TS', 'T2M_RANGE', 'T2M_MAX', 'T2M_MIN', 'PS'],
        'PS_next': ['QV2M', 'RH2M', 'T2M', 'T2MDEW', 'T2MWET', 'TS', 'T2M_RANGE', 'T2M_MAX', 'T2M_MIN', 'PS', 'WS10M'],
        'WS1OM_next': ['WS10M', 'WS10M_MAX', 'WS10M_MIN', 'WS10M_RANGE', 'WS50M', 'WS50M_MAX', 'WS50M_MIN', 'WS50M_RANGE'],
        'WD1OM_next': ['WD10M', 'WD50M']
    }

# Function to train Lasso regression model and evaluate performance
def train_lasso_model(target, features):
    X = data[features]
    y = data[target]

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Create a pipeline that scales the data then applies LassoCV
    pipeline = make_pipeline(StandardScaler(), LassoCV(cv=5, random_state=42))

    # Train the model
    pipeline.fit(X_train, y_train)

    # Predict on the test set
    y_pred = pipeline.predict(X_test)

    # Evaluate the model
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print(f"Model for {target}:")
    print(f"Best alpha: {pipeline.named_steps['lassocv'].alpha_}")
    print(f"MSE: {mse}")
    print(f"R²: {r2}")
    print("-" * 40)

# Train and evaluate models for each target variable
for target, features in targets_and_features.items():
    train_lasso_model(target, features)