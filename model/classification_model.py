import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
import joblib
import numpy as np
from sklearn.impute import SimpleImputer

class ClassificationModel:
    """
    A class used to represent the Classification Model for predicting iris species.

    Attributes
    ----------
    model : RandomForestClassifier
        The Random Forest Classifier model used for prediction.
    scaler : StandardScaler
        The scaler used to normalize/standardize features.
    imputer : SimpleImputer
        The imputer used to handle missing values.
    species_map : dict
        A dictionary mapping species labels to their names.

    Methods
    -------
    load_and_preprocess_data()
        Loads and preprocesses the Iris dataset, including scaling and splitting the data.
    train(X_train, y_train)
        Trains the model using the training dataset and performs hyperparameter tuning.
    predict(input_data)
        Predicts the iris species for the given input data.
    """

    def __init__(self):
        """
        Initializes the ClassificationModel with a scaler, imputer, and sets the model attribute to None.
        """
        self.model = None
        self.scaler = StandardScaler()
        self.imputer = SimpleImputer(strategy='mean')
        self.species_map = {0: 'Setosa', 1: 'Versicolor', 2: 'Virginica'}

    def load_and_preprocess_data(self):
        """
        Loads and preprocesses the Iris dataset.

        Returns
        -------
        tuple
            Contains the training, validation, and test datasets.
        """
        iris = load_iris()
        data = pd.DataFrame(data=iris.data, columns=iris.feature_names)
        data['species'] = iris.target

        # Add synthetic data for augmentation
        synthetic_data = {
            'sepal length (cm)': np.random.uniform(4.3, 7.9, 50),
            'sepal width (cm)': np.random.uniform(2.0, 4.4, 50),
            'petal length (cm)': np.random.uniform(1.0, 6.9, 50),
            'petal width (cm)': np.random.uniform(0.1, 2.5, 50),
            'species': np.random.choice([0, 1, 2], 50)
        }
        synthetic_df = pd.DataFrame(synthetic_data)
        data = pd.concat([data, synthetic_df], ignore_index=True)
        # splitting features
        X = data.drop(columns=['species'])
        y = data['species']

        X.dropna(inplace=True)
        y.dropna(inplace=True)

        X = self.scaler.fit_transform(X)

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
        model = RandomForestClassifier(random_state=42)
        param_grid = {
            'n_estimators': [100, 200, 300],
            'max_depth': [None, 10, 20, 30],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4]
        }
        grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=3, n_jobs=-1, verbose=2)
        grid_search.fit(X_train, y_train)
        self.model = grid_search.best_estimator_

        # Print the best parameters, score, and detailed results
        print("Best parameters found: ", grid_search.best_params_)
        print("Best cross-validation score: ", grid_search.best_score_)
        print("Best estimator: ", grid_search.best_estimator_)

        # Optional: Detailed results
        results = grid_search.cv_results_
        for mean_score, params in zip(results["mean_test_score"], results["params"]):
            print(f"Mean test score: {mean_score:.3f} for parameters: {params}")

        # Save the trained model
        joblib.dump(self.model, 'iris_model.pkl')

    def predict(self, input_data):
        """
        Predicts the iris species for the given input data.

        Parameters
        ----------
        input_data : numpy.ndarray
            The input data for prediction.

        Returns
        -------
        str
            The predicted iris species.
        """
        input_data = self.imputer.fit_transform(input_data)
        input_data = self.scaler.transform(input_data)
        prediction = self.model.predict(input_data)
        predicted_species = self.species_map[int(prediction[0])]
        return predicted_species

# Train the classification model
classification_model = ClassificationModel()
X_train_cls, X_val_cls, X_test_cls, y_train_cls, y_val_cls, y_test_cls = classification_model.load_and_preprocess_data()
classification_model.train(X_train_cls, y_train_cls)
