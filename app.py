from flask import Flask, request, jsonify
import numpy as np
from tensorflow.keras.models import load_model
import joblib

app = Flask(__name__)

# Load model and scaler
model = load_model('models/lstm_model.h5', compile=False)
scaler = joblib.load('models/scaler.pkl')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        print("Request received:", data)

        sequence = np.array(data['sequence']).reshape(-1, 1)

        # Scale input
        sequence = scaler.transform(sequence)

        # Reshape for LSTM
        sequence = sequence.reshape((1, sequence.shape[0], 1))

        prediction = model.predict(sequence)

        # Convert back to original scale
        prediction = scaler.inverse_transform(prediction)

        return jsonify({
            'predicted_energy': float(prediction[0][0])
        })

    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)