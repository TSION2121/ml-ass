import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
# Additional imports for handling the error and ensuring required packages are there

# Load the JSON data
url = "https://data.nasa.gov/resource/e6wj-e2uc.json"
data = pd.read_json(url)

# Display basic information about the dataset
print(data.info())

# Display summary statistics
print(data.describe())

# Display the first few rows of the dataset
print(data.head())

# Check for missing values
print(data.isnull().sum())

# Fill or drop missing values
data = data.dropna()  # Example: Dropping rows with missing values

# Remove duplicate rows
data = data.drop_duplicates()

print(data.head())

# Split the 'year_doy' column into 'year' and 'day_of_year'
data[['year', 'day_of_year']] = data['year_doy'].str.split('-', expand=True)

# Convert 'year' and 'day_of_year' to numeric values
data['year'] = pd.to_numeric(data['year'])
data['day_of_year'] = pd.to_numeric(data['day_of_year'])

# Drop the original 'year_doy' and 'hh_mm_ss' columns
data = data.drop(['year_doy', 'hh_mm_ss'], axis=1)

# Visualizing the distribution of the numerical features
features = ['date_doy', 'et1989', 'b0', 'b1', 'dc3']
for feature in features:
    plt.figure(figsize=(10, 6))
    sns.histplot(data[feature], kde=True)
    plt.title(f'Distribution of {feature}')
    plt.show()

# Visualize relationships between features
sns.pairplot(data, vars=features)
plt.show()

# Correlation matrix to understand relationships
corr_matrix = data.corr(numeric_only=True)  # Specify numeric_only to avoid errors
plt.figure(figsize=(10, 6))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
plt.title('Correlation Matrix')
plt.show()

# One-hot encode the 'orb' column
data = pd.get_dummies(data, columns=['orb'])
print(data.head())

# Scale numerical features
scaler = StandardScaler()
scaled_features = ['date_doy', 'et1989', 'b0', 'b1', 'year', 'day_of_year']
data[scaled_features] = scaler.fit_transform(data[scaled_features])
print(data.head())

# Define features (X) and target (y)
X = data.drop('dc3', axis=1)  # Assuming 'dc3' is our target variable
y = data['dc3']

# Split the data
X_train, X_val_test, y_train, y_val_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_val_test, y_val_test, test_size=0.5, random_state=42)

print(X_train.shape, X_val.shape, X_test.shape)
print(y_train.shape, y_val.shape, y_test.shape)

# Initialize models
linear_reg = LinearRegression()
lasso_reg = Lasso()
random_forest_reg = RandomForestRegressor()

# Train and evaluate models
models = {
    "Linear Regression": linear_reg,
    "Lasso Regression": lasso_reg,
    "Random Forest Regressor": random_forest_reg
}

results = {}

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_val)
    mse = mean_squared_error(y_val, y_pred)
    rmse = mse ** 0.5  # Manually calculate RMSE
    r2 = r2_score(y_val, y_pred)
    results[name] = {"MSE": mse, "RMSE": rmse, "R²": r2}
    print(f"{name} - MSE: {mse}, RMSE: {rmse}, R²: {r2}")

# # Save the processed data to an Excel file
# data.to_excel('processed_data1.xlsx', index=False)

# Save the Random Forest Regressor model along with scaler, scaled features, and feature columns
joblib.dump((random_forest_reg, scaler, scaled_features, X.columns), 'final_model1.pkl')
