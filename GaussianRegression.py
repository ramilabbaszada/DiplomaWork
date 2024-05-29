import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import DotProduct, WhiteKernel
from sklearn.metrics import mean_absolute_error, r2_score

# Load the dataset
data_path = 'Final_Dataset.csv'
data = pd.read_csv(data_path)

data = data[data['YEAR'] >= 2010]

# Define the target variables
target_variables = ['PRECTOTCORR_next', 'TS_next', 'RH2M_next', 'PS_next', 'WS1OM_next', 'WD1OM_next']

# Define features to exclude
exclude_features = target_variables

# Define all features except the excluded ones
features = data.columns.difference(exclude_features)

# Split the data into features (X) and targets (Y)
X = data[features]
Y = data[target_variables]

# Split the data into training and testing sets
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# Define preprocessing for numerical and categorical features
numerical_features = X.select_dtypes(include=['int64', 'float64']).columns
categorical_features = X.select_dtypes(include=['object']).columns

numerical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler())
])
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_features),
        ('cat', categorical_transformer, categorical_features)
    ])

# Define the kernel for Gaussian Process Regressor
kernel = DotProduct() + WhiteKernel()

# Initialize results dictionary
results = {}

# Train, predict, and evaluate for each target variable
for target in target_variables:
    # Create and train the Gaussian Process Regressor model
    model = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', GaussianProcessRegressor(kernel=kernel, random_state=42))
    ])

    model.fit(X_train, Y_train[target])

    # Make predictions
    y_pred = model.predict(X_test)

    # Evaluate predictions
    mae = mean_absolute_error(Y_test[target], y_pred)
    r2 = r2_score(Y_test[target], y_pred)

    # Store the results
    results[target] = {'MAE': mae, 'R2': r2}

# Print the performance metrics for each target variable
for target, metrics in results.items():
    print(f"{target} - MAE: {metrics['MAE']}, R²: {metrics['R2']:.4f}")