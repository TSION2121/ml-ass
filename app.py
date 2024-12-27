from flask import Flask, render_template, request, jsonify, Blueprint
import pandas as pd
import numpy as np
import logging
import matplotlib.pyplot as plt
import seaborn as sns
import io
import base64
import joblib
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix, mean_absolute_error, mean_squared_error, r2_score
from sklearn.datasets import fetch_california_housing, load_iris
from model.classification_model import ClassificationModel
from model.regression_model import RegressionModel
from model.random_forest_regressor_model import RandomForestRegressionModel
from waitress import serve
from updated import scaled_features, features, data, corr_matrix

# Configure logging
logging.basicConfig(level=logging.INFO)

app = Flask(__name__)

# Initialize and load models
classification_model = ClassificationModel()
regression_model = RegressionModel()

# Train and load classification model
X_train_cls, X_val_cls, X_test_cls, y_train_cls, y_val_cls, y_test_cls = classification_model.load_and_preprocess_data()
classification_model.train(X_train_cls, y_train_cls)
logging.info("Classification model trained and saved.")

# Train and load regression model
X_train_reg, X_val_reg, X_test_reg, y_train_reg, y_val_reg, y_test_reg = regression_model.load_and_preprocess_data()
regression_model.train(X_train_reg, y_train_reg)
logging.info("Regression model trained and saved.")

# Load and train Random Forest Regression model
random_forest_model = RandomForestRegressionModel()
url = "https://data.nasa.gov/resource/e6wj-e2uc.json"
X_train, X_val, X_test, y_train, y_val, y_test = random_forest_model.load_and_preprocess_data(url)
random_forest_model.train(X_train, y_train)
model, scaler, scaled_features, feature_columns = joblib.load('final_model1.pkl')

# Blueprint for regression routes
regression_bp = Blueprint('regression', __name__, template_folder='templates')

# Blueprint for classification routes
classification_bp = Blueprint('classification', __name__, template_folder='templates')

# Blueprint for Random Forest Regression routes
random_forest_regression_bp = Blueprint('random_forest_regression', __name__, template_folder='templates')

@app.route('/')
def home():
    return render_template('home.html')

@random_forest_regression_bp.route('/')
def random_forest_regression_home():
    return render_template('random_forest_regression.html')

@random_forest_regression_bp.route('/predict', methods=['POST'])
def predict_random_forest_regression():
    try:
        data = request.get_json(force=True)
        input_data = {feature: 0 for feature in feature_columns}  # Initialize all features to 0
        input_data.update({feature: data['features'][i] for i, feature in enumerate(feature_columns[:len(data['features'])])})
        features_df = pd.DataFrame([input_data])
        features_df[scaled_features] = scaler.transform(features_df[scaled_features])
        prediction = model.predict(features_df)

        # Evaluation metrics on the test set
        y_pred = model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        rmse = mse ** 0.5  # Manually calculate RMSE
        r2 = r2_score(y_test, y_pred)

        # Plotting actual vs predicted values
        plt.figure(figsize=(10, 6))
        sns.scatterplot(x=y_test, y=y_pred)
        plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='green', linestyle='--')
        plt.xlabel('Actual Values')
        plt.ylabel('Predicted Values')
        plt.title('Actual vs. Predicted Values for Random Forest Regressor')

        img = io.BytesIO()
        plt.savefig(img, format='png')
        img.seek(0)
        plot_url = base64.b64encode(img.getvalue()).decode()

        return jsonify({
            'prediction': prediction[0],
            'mse': mse,
            'rmse': rmse,
            'r2': r2,
            'plot_url': f'data:image/png;base64,{plot_url}'
        })
    except Exception as e:
        return jsonify({'error': str(e)})

@random_forest_regression_bp.route('/graphs', methods=['GET'])
def get_graphs():
    try:
        # Generate and encode graphs
        graphs = []

        # Histogram
        for feature in features:
            plt.figure(figsize=(10, 6))
            sns.histplot(data[feature], kde=True)
            plt.title(f'Distribution of {feature}')
            img = io.BytesIO()
            plt.savefig(img, format='png')
            img.seek(0)
            graphs.append(base64.b64encode(img.getvalue()).decode())

        # Pairplot
        pairplot = sns.pairplot(data, vars=features)
        img = io.BytesIO()
        pairplot.savefig(img, format='png')
        img.seek(0)
        graphs.append(base64.b64encode(img.getvalue()).decode())

        # Correlation Matrix
        plt.figure(figsize=(10, 6))
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
        plt.title('Correlation Matrix')
        img = io.BytesIO()
        plt.savefig(img, format='png')
        img.seek(0)
        graphs.append(base64.b64encode(img.getvalue()).decode())

        return jsonify({'graphs': graphs})
    except Exception as e:
        return jsonify({'error': str(e)})

@regression_bp.route('/')
def regression_home():
    return render_template('regression.html')

@regression_bp.route('/predict', methods=['POST'])
def predict_regression():
    try:
        data = request.get_json(force=True)
        feature_names = fetch_california_housing().feature_names
        input_data = pd.DataFrame([data], columns=feature_names)
        logging.info(f'Received data for regression prediction: {input_data}')
        prediction = regression_model.predict(input_data)
        logging.info(f'Regression prediction: {prediction}')

        y_pred = regression_model.model.predict(X_test_reg)
        mae = mean_absolute_error(y_test_reg, y_pred)
        mse = mean_squared_error(y_test_reg, y_pred)
        r2 = r2_score(y_test_reg, y_pred)

        # Plotting actual vs predicted values
        plt.figure(figsize=(10, 6))
        sns.scatterplot(x=y_test_reg, y=y_pred)
        plt.scatter([y for y in y_test_reg], [prediction] * len(y_test_reg), color='red')
        plt.plot([min(y_test_reg), max(y_test_reg)], [min(y_test_reg), max(y_test_reg)], color='green', linestyle='--')
        plt.xlabel('Actual Values')
        plt.ylabel('Predicted Values')
        plt.title('Actual vs. Predicted Values')

        img = io.BytesIO()
        plt.savefig(img, format='png')
        img.seek(0)
        plot_url = base64.b64encode(img.getvalue()).decode()

        return jsonify({
            'prediction': prediction,
            'mae': mae,
            'mse': mse,
            'r2': r2,
            'plot_url': f'data:image/png;base64,{plot_url}'
        })
    except Exception as e:
        logging.error(f'Error during regression prediction: {e}')
        return jsonify({'error': str(e)})



@classification_bp.route('/')
def classification_home():
    return render_template('classification.html')

@classification_bp.route('/predict', methods=['POST'])
def predict_classification():
    try:
        logging.info('Entered the classification prediction route.')
        data = request.get_json(force=True)
        logging.info(f'Received data: {data}')

        # Replace underscores with spaces to match the feature names
        processed_data = {key.replace('_', ' '): value for key, value in data.items()}
        feature_names = load_iris().feature_names
        input_data = pd.DataFrame([processed_data], columns=feature_names)
        logging.info(f'Formatted input data: {input_data}')

        if classification_model.model is None:
            logging.error('Model is not loaded correctly.')
            return jsonify({'error': 'Model not loaded'})

        predicted_species = classification_model.predict(input_data)
        logging.info(f'Prediction result: {predicted_species}')

        y_pred = classification_model.model.predict(X_test_cls)
        accuracy = accuracy_score(y_test_cls, y_pred)
        precision = precision_score(y_test_cls, y_pred, average='weighted')
        recall = recall_score(y_test_cls, y_pred, average='weighted')
        f1 = f1_score(y_test_cls, y_pred, average='weighted')
        report = classification_report(y_test_cls, y_pred, output_dict=True)
        conf_matrix = confusion_matrix(y_test_cls, y_pred)

        # Plotting confusion matrix
        plt.figure(figsize=(8, 6))
        sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="YlGnBu")
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.title('Confusion Matrix')

        img = io.BytesIO()
        plt.savefig(img, format='png')
        img.seek(0)
        plot_url = base64.b64encode(img.getvalue()).decode()

        return jsonify({
            'prediction': predicted_species,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'report': report,
            'confusion_matrix': f'data:image/png;base64,{plot_url}'
        })
    except Exception as e:
        logging.error(f'Error during classification prediction: {e}')
        return jsonify({'error': str(e)})

# Register Blueprints
app.register_blueprint(regression_bp, url_prefix='/regression')
app.register_blueprint(classification_bp, url_prefix='/classification')
app.register_blueprint(random_forest_regression_bp, url_prefix='/random_forest_regression')



if __name__ == '__main__':
    serve(app, host='127.0.0.1', port=8080)
