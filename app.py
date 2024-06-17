from flask import Flask, request, render_template
from tensorflow.keras.models import load_model, Sequential
from tensorflow.keras.layers import Embedding, SimpleRNN, LSTM, Dense, SpatialDropout1D
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np

app = Flask(__name__)

# Load the tokenizer and fit on sample data
tokenizer = Tokenizer(num_words=5000, lower=True, oov_token='<OOV>')
sample_data = ["This is a sample email text."]  # Replace with your actual data
tokenizer.fit_on_texts(sample_data)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    email_text = request.form['email_text']
    model_type = request.form['model_type']
    
    sequences = tokenizer.texts_to_sequences([email_text])
    padded = pad_sequences(sequences, maxlen=100)
    
    if model_type == 'LSTM':
        model = load_model('phishing_email_detector.keras')
    elif model_type == 'RNN':
        model = load_model('phishing_email_detector_2.keras')
    else:
        return render_template('index.html', prediction_text='Invalid model type selected.')
    
    prediction = model.predict(padded)
    is_phishing = (prediction > 0.5).astype(int)
    
    return render_template('index.html', prediction_text='Phishing' if is_phishing[0][0] == 1 else 'Not Phishing')

if __name__ == "__main__":
    app.run(debug=True)
