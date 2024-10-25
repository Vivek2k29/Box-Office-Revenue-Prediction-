from flask import Flask, request, jsonify, render_template
import joblib
import pandas as pd

app = Flask(__name__)

# Load the trained model
model = joblib.load('boxoffice_rf_model.pkl')


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    # Get form inputs
    domestic_revenue = float(request.form['domestic_revenue'])
    opening_revenue = float(request.form['opening_revenue'])
    opening_theaters = int(request.form['opening_theaters'])
    budget = float(request.form['budget'])
    release_days = int(request.form['release_days'])

    # Create DataFrame from input
    input_data = pd.DataFrame({
        'domestic_revenue': [domestic_revenue],
        'opening_revenue': [opening_revenue],
        'opening_theaters': [opening_theaters],
        'budget': [budget],
        'release_days': [release_days]
    })

    # Predict the global revenue using the model
    prediction = model.predict(input_data)

    # Return the prediction as JSON
    return jsonify({'predicted_revenue': prediction[0]})


if __name__ == '__main__':
    app.run(debug=True)


