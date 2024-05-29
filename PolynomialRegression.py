import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Load the dataset
data = pd.read_csv('Final_Dataset.csv')

# Select the features and target for RH2M_next
features_rh2m = ['WD10M', 'WD50M']
target_rh2m = 'WD1OM_next'


# Prepare the data
X = data[features_rh2m]
y = data[target_rh2m]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Create polynomial features
poly = PolynomialFeatures(degree=2)
X_train_poly = poly.fit_transform(X_train_scaled)
X_test_poly = poly.transform(X_test_scaled)

# Fit the regression model
model = LinearRegression()
model.fit(X_train_poly, y_train)

# Predict on testing set
y_pred = model.predict(X_test_poly)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f'Mean Squared Error for WS1OM_next: {mse}')
print(f'R2 Score for WS1OM_next: {r2}')