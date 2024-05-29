import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Load the updated dataset
updated_data = pd.read_csv("Final_Dataset.csv")

# Calculate the correlation matrix
correlation_matrix_full = updated_data.corr()

# Define the target variables
targets = ['RH2M_next', 'TS_next', 'PS_next', 'WS1OM_next', 'WD1OM_next']

# Filter columns based on a correlation threshold of 0.2 for each target variable
correlation_threshold = 0.2
filtered_columns_refined = {}

# Filtering columns for each target excluding other target variables
for target in targets:
    # Find columns where the absolute correlation with the target is greater than 0.2
    relevant_columns = correlation_matrix_full.index[abs(correlation_matrix_full[target]) > 0.2].tolist()
    # Remove the current target from the list and any other target variables
    relevant_columns = [col for col in relevant_columns if col not in targets and col != target]
    filtered_columns_refined[target] = relevant_columns



# Create a dictionary to store results
model_results = {}

# Iterate over each target to create and evaluate a model
for target, features in filtered_columns_refined.items():
    # Prepare the data
    X = updated_data[features]
    y = updated_data[target].dropna()  # Ensure target has no missing values
    X = X.loc[y.index]  # Align the features with the target

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Feature Scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Create and train the model
    model = LinearRegression()
    model.fit(X_train_scaled, y_train)

    # Predict and evaluate the model
    y_pred = model.predict(X_test_scaled)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    # Store the results
    model_results[target] = {'MSE': mse, 'R2 Score': r2}

print(model_results)