import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
import joblib

class RegressionModel:
    """
    A class used to represent the Regression Model for predicting house prices.

    Attributes
    ----------
    model : LinearRegression
        The linear regression model used for prediction.
    scaler : StandardScaler
        The scaler used to normalize/standardize features.

    Methods
    -------
    load_and_preprocess_data()
        Loads and preprocesses the California Housing dataset, including scaling and splitting the data.
    train(X_train, y_train)
        Trains the model using the training dataset and performs hyperparameter tuning.
    predict(input_data)
        Predicts house prices for the given input data.
    """

    def __init__(self):
        """
        Initializes the RegressionModel with a scaler and sets the model attribute to None.
        """
        self.model = None
        self.scaler = StandardScaler()

    def load_and_preprocess_data(self):
        """
        Loads and preprocesses the California Housing dataset.

        Returns
        -------
        tuple
            Contains the training, validation, and test datasets.
        """
        # Load and preprocess the California Housing dataset
        housing = fetch_california_housing()
        data = pd.DataFrame(housing.data, columns=housing.feature_names)
        data['MedHouseVal'] = housing.target

        # Define features and target
        X = data.drop(columns=['MedHouseVal'])
        y = data['MedHouseVal']

        # Normalize/Standardize features
        X = self.scaler.fit_transform(X)

        # Split the data into training, validation, and test sets
        X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
        X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

        return X_train, X_val, X_test, y_train, y_val, y_test

    def train(self, X_train, y_train):
        """
        Trains the model using the training dataset and performs hyperparameter tuning.

        Parameters
        ----------
        X_train : numpy.ndarray
            The training input samples.
        y_train : numpy.ndarray
            The target values for training.
        """
        model = LinearRegression()
        param_grid = {'fit_intercept': [True, False], 'positive': [True, False]}
        grid_search = GridSearchCV(LinearRegression(), param_grid, cv=5)
        grid_search.fit(X_train, y_train)
        self.model = grid_search.best_estimator_

        # Save the trained model
        joblib.dump(self.model, 'california_housing_model.pkl')

    def predict(self, input_data):
        """
        Predicts house prices for the given input data.

        Parameters
        ----------
        input_data : numpy.ndarray
            The input data for prediction.

        Returns
        -------
        float
            The predicted house price.
        """
        input_data = self.scaler.transform(input_data)
        prediction = self.model.predict(input_data)[0]
        return prediction

regression_model = RegressionModel()
