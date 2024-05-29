import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score

# Load the dataset
data = pd.read_csv('Final_Dataset.csv')

# Define the target variables and their corresponding features
targets_and_features = {
    'RH2M_next': ['QV2M', 'RH2M', 'T2M', 'T2MDEW', 'T2MWET', 'TS', 'T2M_MAX', 'T2M_MIN'],
    'TS_next': ['MO', 'QV2M', 'RH2M', 'T2M', 'T2MDEW', 'T2MWET', 'TS', 'T2M_RANGE', 'T2M_MAX', 'T2M_MIN', 'PS'],
    'PS_next': ['QV2M', 'RH2M', 'T2M', 'T2MDEW', 'T2MWET', 'TS', 'T2M_RANGE', 'T2M_MAX', 'T2M_MIN', 'PS', 'WS10M'],
    'WS1OM_next': ['WS10M', 'WS10M_MAX', 'WS10M_MIN', 'WS10M_RANGE', 'WS50M', 'WS50M_MAX', 'WS50M_MIN', 'WS50M_RANGE'],
    'WD1OM_next': ['WD10M', 'WD50M']
}

# Model training and evaluation
results = {}
for target, features in targets_and_features.items():
    X = data[features]
    y = data[target]

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Initialize the StandardScaler
    scaler = StandardScaler()

    # Fit the scaler on the training data and transform both the training and testing data
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Initialize and train the Ridge Regression model
    model = Ridge()
    model.fit(X_train_scaled, y_train)

    # Predict on the testing set
    y_pred = model.predict(X_test_scaled)

    # Calculate MSE and R2 score
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    # Store the results
    results[target] = {'MSE': mse, 'R2 Score': r2}

# Output the results for each model
for target, scores in results.items():
    print(f"Results for {target}:")
    print(f"MSE: {scores['MSE']:}")
    print(f"R2 Score: {scores['R2 Score']:.4f}\n")