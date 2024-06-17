
from flask import Flask, request, render_template
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np

app = Flask(__name__)

# Load the trained model
model = load_model('phishing_email_detector.keras')

# Initialize the tokenizer
tokenizer = Tokenizer(num_words=5000, lower=True, oov_token='<OOV>')

# Example function to fit tokenizer on sample data
# Replace `sample_data` with the actual data used during training
sample_data = ["This is a sample email text."]  # Replace with your actual data
tokenizer.fit_on_texts(sample_data)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    email_text = request.form['email_text']
    sequences = tokenizer.texts_to_sequences([email_text])
    padded = pad_sequences(sequences, maxlen=100)  # Ensure the maxlen is the same as used during training
    prediction = model.predict(padded)
    is_phishing = (prediction > 0.5).astype(int)

    return render_template('index.html', prediction_text='Phishing' if is_phishing[0][0] == 1 else 'Not Phishing')

if __name__ == "__main__":
    app.run(debug=True)
