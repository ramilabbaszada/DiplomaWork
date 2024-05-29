import pandas as pd
from matplotlib import pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import DotProduct, WhiteKernel
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge, RidgeCV, LinearRegression, LassoCV, Lasso, MultiTaskLassoCV
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor

# Load the dataset
df = pd.read_csv('dataset.csv')

# Assuming the dataset is ordered by date, shift the target variables by -1 to predict the next day's values
df['PRECTOTCORR_next'] = df['PRECTOTCORR'].shift(-1)
df['TS_next'] = df['TS'].shift(-1)

# Drop the last row as it will have NaN values for the shifted columns
df = df[:-1]

# Selecting features and target for the model
# Replace 'feature_columns' with the actual column names of your features
X = df[['YEAR','MO','DY','PS','WS10M','WS10M_MAX','WS10M_MIN','WS10M_RANGE','WD10M','WS50M','WS50M_MAX','WS50M_MIN','WS50M_RANGE','WD50M','QV2M','RH2M','PRECTOTCORR','T2M','T2MDEW','T2MWET','TS','T2M_RANGE','T2M_MAX','T2M_MIN']]
Y = df[['PRECTOTCORR_next', 'TS_next']]

# # Splitting the dataset into training and testing sets
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

#------------------------------------------------------------------------------------------------------ Ridge Model
# List of alphas to try out
alphas = [0.01, 0.1, 1.0, 10.0, 100.0]
# Using RidgeCV to find the best alpha
ridge_cv = RidgeCV(alphas=alphas, store_cv_values=True)
ridge_cv.fit(X_train, Y_train)
# The best alpha parameter found
best_alpha = ridge_cv.alpha_
# Training the Ridge Regression model
# You might want to adjust the alpha parameter based on your dataset's characteristics
ridge_model = Ridge(alpha=best_alpha)
ridge_model.fit(X_train, Y_train)
# Making predictions
Y_pred = ridge_model.predict(X_test)
#----------------------------------------------------------------------------------------------------------------------- Polynomial Regression
# # Split the data for PRECTOTCORR
# X_train_prectotcorr, X_test_prectotcorr, y_train_prectotcorr, y_test_prectotcorr = train_test_split(X, Y["PRECTOTCORR_next"], test_size=0.2, random_state=42)
#
# # Split the data for TS
# X_train_ts, X_test_ts, y_train_ts, y_test_ts = train_test_split(X, Y["TS_next"], test_size=0.2, random_state=42)
#
# # Create polynomial features
# degree = 2  # Degree of polynomial features
# poly_features = PolynomialFeatures(degree=degree)
#
# X_train_poly_prectotcorr = poly_features.fit_transform(X_train_prectotcorr)
# X_test_poly_prectotcorr = poly_features.transform(X_test_prectotcorr)
#
# X_train_poly_ts = poly_features.fit_transform(X_train_ts)
# X_test_poly_ts = poly_features.transform(X_test_ts)
#
# # Initialize the model
# model_prectotcorr = LinearRegression()
# model_ts = LinearRegression()
#
# # Train the model for PRECTOTCORR
# model_prectotcorr.fit(X_train_poly_prectotcorr, y_train_prectotcorr)
#
# # Train the model for TS
# model_ts.fit(X_train_poly_ts, y_train_ts)
#
# # Predictions
# y_pred_prectotcorr = model_prectotcorr.predict(X_test_poly_prectotcorr)
# y_pred_ts = model_ts.predict(X_test_poly_ts)
#
# # Evaluation for PRECTOTCORR
# mse_prectotcorr = mean_squared_error(y_test_prectotcorr, y_pred_prectotcorr)
# r2_prectotcorr = r2_score(y_test_prectotcorr, y_pred_prectotcorr)
#
# # Evaluation for TS
# mse_ts = mean_squared_error(y_test_ts, y_pred_ts)
# r2_ts = r2_score(y_test_ts, y_pred_ts)
#
# # Print performance metrics
# print("PRECTOTCORR - MSE:", mse_prectotcorr, "R2:", r2_prectotcorr)
# print("TS - MSE:", mse_ts, "R2:", r2_ts)
#----------------------------------------------------------------------------------------------------- Linear Regression
# # Training the Linear Regression model
# model_PRECTOTCORR = LinearRegression()
# model_PRECTOTCORR.fit(X_train, Y_train)
#
# # Predicting and Evaluating for PRECTOTCORR
# Y_pred = model_PRECTOTCORR.predict(X_test)
#----------------------------------------------------------------------------------------------------- Lasso Regression
# # Scaling features
# scaler = StandardScaler()
# X_scaled = scaler.fit_transform(X)
# # Splitting the dataset into training and testing sets
# X_train, X_test, Y_train, Y_test = train_test_split(X_scaled, Y, test_size=0.2, random_state=42)
# # LassoCV
# lasso_cv = MultiTaskLassoCV(alphas=None, cv=10, max_iter=10000)
# # Fitting the model
# lasso_cv.fit(X_train, Y_train)
# # Training a Lasso model with the best alpha on the training data
# lasso_best = Lasso(alpha=lasso_cv.alpha_)
# lasso_best.fit(X_train, Y_train)
# # Evaluating the model on the test set
# Y_pred = lasso_best.predict(X_test)
#------------------------------------------------------------------------------------------------------ Decision tree regression
# model = DecisionTreeRegressor(random_state=42)
# model.fit(X_train, Y_train)
#
# Y_pred = model.predict(X_test)
#------------------------------------------------------------------------------------------------------ Random Forest Regression
# model = RandomForestRegressor(random_state=42)
# model.fit(X_train, Y_train)
#
# Y_pred = model.predict(X_test)
#------------------------------------------------------------------------------------------------------- KNN Model
# model = KNeighborsRegressor(n_neighbors=5)
# model.fit(X_train, Y_train)
# Y_pred = model.predict(X_test)
#-------------------------------------------------------------------------------------------------------- Support Vector Machines
# Splitting the dataset into training and testing sets for both target variables
# X_train_PRECTOTCORR, X_test_PRECTOTCORR, y_train_PRECTOTCORR, y_test_PRECTOTCORR = train_test_split(X, Y["PRECTOTCORR_next"], test_size=0.2, random_state=42)
# X_train_TS, X_test_TS, y_train_TS, y_test_TS = train_test_split(X, Y["TS_next"], test_size=0.2, random_state=42)
#
# # Standardizing the features (important for SVM)
# scaler_X = StandardScaler().fit(X_train_PRECTOTCORR)
# X_train_PRECTOTCORR_scaled = scaler_X.transform(X_train_PRECTOTCORR)
# X_test_PRECTOTCORR_scaled = scaler_X.transform(X_test_PRECTOTCORR)
#
# X_train_TS_scaled = scaler_X.transform(X_train_TS)
# X_test_TS_scaled = scaler_X.transform(X_test_TS)
#
# # Initializing and training the SVR models
# svr_PRECTOTCORR = SVR(kernel='rbf')
# svr_TS = SVR(kernel='rbf')
#
# svr_PRECTOTCORR.fit(X_train_PRECTOTCORR_scaled, y_train_PRECTOTCORR)
# svr_TS.fit(X_train_TS_scaled, y_train_TS)
#
# # Making predictions
# y_pred_PRECTOTCORR = svr_PRECTOTCORR.predict(X_test_PRECTOTCORR_scaled)
# y_pred_TS = svr_TS.predict(X_test_TS_scaled)
#
# # Evaluating the models
# mse_PRECTOTCORR = mean_squared_error(y_test_PRECTOTCORR, y_pred_PRECTOTCORR)
# r2_PRECTOTCORR = r2_score(y_test_PRECTOTCORR, y_pred_PRECTOTCORR)
#
# mse_TS = mean_squared_error(y_test_TS, y_pred_TS)
# r2_TS = r2_score(y_test_TS, y_pred_TS)
#
# # Print performance metrics
# print(f"PRECTOTCORR - MSE: {mse_PRECTOTCORR}, R2: {r2_PRECTOTCORR}")
# print(f"TS - MSE: {mse_TS}, R2: {r2_TS}")
#----------------------------------------------------------------------------------------------------------------------- Gaussian Regression
# Define the kernel
kernel = DotProduct() + WhiteKernel()

# Initialize and train the Gaussian Process Regressors
gpr_PRECTOTCORR = GaussianProcessRegressor(kernel=kernel, random_state=42).fit(X_train, Y_train['PRECTOTCORR_next'])
gpr_TS = GaussianProcessRegressor(kernel=kernel, random_state=42).fit(X_train, Y_train['TS_next'])

# Make predictions
y_pred_PRECTOTCORR = gpr_PRECTOTCORR.predict(X_test)
y_pred_TS = gpr_TS.predict(X_test)

# Evaluate PRECTOTCORR predictions
mae_PRECTOTCORR = mean_absolute_error(Y_test['PRECTOTCORR_next'], y_pred_PRECTOTCORR)
r2_PRECTOTCORR = r2_score(Y_test['PRECTOTCORR_next'], y_pred_PRECTOTCORR)

# Evaluate TS predictions
mae_TS = mean_absolute_error(Y_test['TS_next'], y_pred_TS)
r2_TS = r2_score(Y_test['TS_next'], y_pred_TS)

# Print performance metrics
print(f"PRECTOTCORR - MAE: {mae_PRECTOTCORR}, R²: {r2_PRECTOTCORR}")
print(f"TS - MAE: {mae_TS}, R²: {r2_TS}")

#----------------------------------------------------------------------------------------------------------------------- Neural Network Regression
# # Evaluating the model performance for each target
# mse_prectotcorr = mean_squared_error(Y_test['PRECTOTCORR_next'], Y_pred[:, 0])
# r2_prectotcorr = r2_score(Y_test['PRECTOTCORR_next'], Y_pred[:, 0])
#
# mse_ts = mean_squared_error(Y_test['TS_next'], Y_pred[:, 1])
# r2_ts = r2_score(Y_test['TS_next'], Y_pred[:, 1])
#
# # Displaying the performance metrics
# print("Performance for PRECTOTCORR_next:")
# print(f"Mean Squared Error: {mse_prectotcorr}")
# print(f"R-squared: {r2_prectotcorr}")
#
# print("\nPerformance for TS_next:")
# print(f"Mean Squared Error: {mse_ts}")
# print(f"R-squared: {r2_ts}")
#




























# import pandas as pd
# import matplotlib.pyplot as plt
# import seaborn as sns

# dataset1 = pd.read_csv("1.csv")
# dataset2 = pd.read_csv("2.csv")
#
# finaldatet=dataset1.merge(dataset2, on=["YEAR","MO","DY"])
#
# finaldatet.to_csv("finaldataset.csv",index=False)


# Assuming 'df' is your DataFrame and 'column_name' is the column you want to plot
# Load your dataset into df
# df = pd.read_csv('finaldataset.csv')  # Example for loading dataset from a CSV file
#
# # Plotting
# plt.figure(figsize=(10, 6))  # Adjust the size of your plot
# sns.histplot(df['TS'], kde=True, stat='density', linewidth=0)
# plt.title('Density Plot for Temperature')
# plt.xlabel('column_name')
# plt.ylabel('Density')
#
# plt.show()
#
# # Plotting
# plt.figure(figsize=(10, 6))  # Adjust the size of your plot
# sns.histplot(df['PRECTOTCORR'], kde=True, stat='density', linewidth=0)
# plt.title('Density Plot for PRECTOTCORR')
# plt.xlabel('column_name')
# plt.ylabel('Density')
# plt.xlim(-10, 25)
# plt.show()
#
# # Calculate correlation matrix
# corr = df.corr()
#
# # Plot the heatmap
# plt.figure(figsize=(15, 12))  # Adjust the size as needed
# sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f", linewidths=.1)
# plt.title('Correlation Heatmap among Features')
# plt.show()

#df['MO'] = pd.Categorical(df['MO'], categories=range(1, 13), ordered=True)

# Plotting
# plt.figure(figsize=(12, 6))  # Adjust figure size as necessary
# sns.boxplot(x='MO', y='PRECTOTCORR', data=df)
# plt.title('Monthly Distribution of PRECTOTCORR')
# plt.xlabel('Month')
# plt.ylabel('PRECTOTCORR')
# plt.xticks(ticks=range(12), labels=['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])
# plt.show()
#
# plt.figure(figsize=(12, 6))  # Adjust figure size as necessary
# sns.boxplot(x='MO', y='TS', data=df)
# plt.title('Monthly Distribution of Temperature')
# plt.xlabel('Month')
# plt.ylabel('Temperature')
# plt.xticks(ticks=range(12), labels=['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])
# plt.show()