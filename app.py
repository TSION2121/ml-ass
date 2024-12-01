from flask import Flask, render_template, request, jsonify, Blueprint
import pandas as pd
import joblib
import logging
import matplotlib.pyplot as plt
import seaborn as sns
import io
import base64
from model.classification_model import ClassificationModel
from model.regression_model import RegressionModel
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix, mean_absolute_error, mean_squared_error, r2_score
from sklearn.datasets import fetch_california_housing, load_iris

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

# Blueprint for regression routes
regression_bp = Blueprint('regression', __name__, template_folder='templates')

# Blueprint for classification routes
classification_bp = Blueprint('classification', __name__, template_folder='templates')

@regression_bp.route('/')
def regression_home():
    """
    Renders the home page for regression predictions.

    Returns
    -------
    HTML template
        The regression.html template.
    """
    return render_template('regression.html')

@regression_bp.route('/predict', methods=['POST'])
def predict_regression():
    """
    Predicts house prices based on input data and returns the prediction and evaluation metrics.

    Returns
    -------
    JSON
        Contains the prediction, mean absolute error, mean squared error, R², and a plot URL.
    """
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
    """
    Renders the home page for classification predictions.

    Returns
    -------
    HTML template
        The classification.html template.
    """
    return render_template('classification.html')

@classification_bp.route('/predict', methods=['POST'])
def predict_classification():
    """
    Predicts iris species based on input data and returns the prediction and evaluation metrics.

    Returns
    -------
    JSON
        Contains the prediction, accuracy, precision, recall, F1-score, classification report, and a confusion matrix plot URL.
    """
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

@app.route('/')
def home():
    """
    Renders the home page.

    Returns
    -------
    HTML template
        The home.html template.
    """
    return render_template('home.html')

if __name__ == '__main__':
    app.run(debug=True, port=5000)
