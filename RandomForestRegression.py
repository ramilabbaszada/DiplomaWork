import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

# Load the dataset
data_path = 'Final_Dataset.csv'
data = pd.read_csv(data_path)

# Define target variables
target_variables = ['PRECTOTCORR_next', 'TS_next', 'RH2M_next', 'PS_next', 'WS1OM_next', 'WD1OM_next']

# Define all features except the target variables
features = data.columns.difference(target_variables)

results = {}

# Iterate through each target variable
for target in target_variables:
    # Split the data into features (X) and target (y)
    X = data[features]
    y = data[target]

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Define preprocessing for numerical and categorical features
    numerical_features = X.select_dtypes(include=['int64', 'float64']).columns
    categorical_features = X.select_dtypes(include=['object']).columns

    numerical_transformer = SimpleImputer(strategy='mean')
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numerical_features),
            ('cat', categorical_transformer, categorical_features)
        ])

    # Create a pipeline that first preprocesses the data and then fits the model
    model = Pipeline(steps=[('preprocessor', preprocessor),
                            ('regressor', RandomForestRegressor(random_state=42, n_estimators=100))])

    # Train the model
    model.fit(X_train, y_train)

    # Make predictions
    y_pred = model.predict(X_test)

    # Calculate MSE and R2 score
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    # Store the results
    results[target] = {'MSE': mse, 'R2': r2}

    print(f"{target} - MSE: {mse:}, R2: {r2:.4f}")

# Display the results
for target, metrics in results.items():
    print(f"Target: {target}")
    for metric, value in metrics.items():
        print(f"  {metric}: {value:.4f}")
    print()