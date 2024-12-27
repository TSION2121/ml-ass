import pandas as pd
import joblib
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV

class RandomForestRegressionModel:
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.scaled_features = None
        self.feature_columns = None

    def load_and_preprocess_data(self, url):
        data = pd.read_json(url)
        data = data.dropna()
        data = data.drop_duplicates()

        # Split the 'year_doy' column into 'year' and 'day_of_year'
        data[['year', 'day_of_year']] = data['year_doy'].str.split('-', expand=True)
        data['year'] = pd.to_numeric(data['year'])
        data['day_of_year'] = pd.to_numeric(data['day_of_year'])
        data = data.drop(['year_doy', 'hh_mm_ss'], axis=1)

        # One-hot encode the 'orb' column
        data = pd.get_dummies(data, columns=['orb'])
        self.feature_columns = data.drop('dc3', axis=1).columns

        # Generate synthetic data for augmentation
        synthetic_data = {
            'date_doy': np.random.uniform(data['date_doy'].min(), data['date_doy'].max(), 100),
            'et1989': np.random.uniform(data['et1989'].min(), data['et1989'].max(), 100),
            'b0': np.random.uniform(data['b0'].min(), data['b0'].max(), 100),
            'b1': np.random.uniform(data['b1'].min(), data['b1'].max(), 100),
            'year': np.random.randint(data['year'].min(), data['year'].max() + 1, 100),
            'day_of_year': np.random.randint(data['day_of_year'].min(), data['day_of_year'].max() + 1, 100),
            'dc3': np.random.uniform(data['dc3'].min(), data['dc3'].max(), 100)
        }

        # Include synthetic 'orb' columns
        for orb_column in data.columns:
            if orb_column.startswith('orb_'):
                synthetic_data[orb_column] = np.random.choice([0, 1], 100)

        synthetic_df = pd.DataFrame(synthetic_data)
        data = pd.concat([data, synthetic_df], ignore_index=True)

        # Scale numerical features
        self.scaled_features = ['date_doy', 'et1989', 'b0', 'b1', 'year', 'day_of_year']
        data[self.scaled_features] = self.scaler.fit_transform(data[self.scaled_features])

        # Define features (X) and target (y)
        X = data.drop('dc3', axis=1)
        y = data['dc3']

        # Split the data
        X_train, X_val_test, y_train, y_val_test = train_test_split(X, y, test_size=0.2, random_state=42)
        X_val, X_test, y_val, y_test = train_test_split(X_val_test, y_val_test, test_size=0.5, random_state=42)

        return X_train, X_val, X_test, y_train, y_val, y_test

    def train(self, X_train, y_train):
        model = RandomForestRegressor(random_state=42)
        param_grid = {
            'n_estimators': [100, 200, 300],
            'max_depth': [None, 10, 20, 30],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4]
        }
        grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=3, n_jobs=-1, verbose=2)
        grid_search.fit(X_train, y_train)
        self.model = grid_search.best_estimator_

        # Save the model, scaler, scaled_features, and feature_columns
        joblib.dump((self.model, self.scaler, self.scaled_features, self.feature_columns), 'final_model1.pkl')

    def predict(self, input_data):
        model, scaler, scaled_features, feature_columns = joblib.load('final_model1.pkl')
        input_df = pd.DataFrame([input_data], columns=input_data.keys())
        input_df = pd.get_dummies(input_df).reindex(columns=feature_columns, fill_value=0)
        input_df[scaled_features] = scaler.transform(input_df[scaled_features])
        prediction = model.predict(input_df)
        return prediction[0]

if __name__ == "__main__":
    url = "https://data.nasa.gov/resource/e6wj-e2uc.json"
    regression_model = RandomForestRegressionModel()
    X_train, X_val, X_test, y_train, y_val, y_test = regression_model.load_and_preprocess_data(url)
    regression_model.train(X_train, y_train)
