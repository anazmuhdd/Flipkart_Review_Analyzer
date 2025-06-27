from flask import Flask, request, jsonify
from transformers import BertTokenizer, BertForSequenceClassification
import torch
from flask_cors import CORS

app = Flask(__name__)
CORS(app)
allow_cross_origin = True  # Set to True to allow cross-origin requests
model_path = 'model'
tokenizer = BertTokenizer.from_pretrained(model_path)
model = BertForSequenceClassification.from_pretrained(model_path)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)
model.eval()
label_map= {0: 'negative', 1: 'neutral', 2: 'positive'}

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    if 'review' not in data:
        return jsonify({'error': 'No review provided'}), 400
    review = data['review']
    encodings=tokenizer(review, truncation=True, padding=True, return_tensors='pt', max_length=128)
    input_ids = encodings['input_ids'].to(device)
    attention_mask = encodings['attention_mask'].to(device)
    
    with torch.no_grad():
        outputs = model(input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        predictions = torch.argmax(logits, dim=1)
        
    predicted_label = label_map[predictions.item()]
    return jsonify({'review': review, 'sentiment': predicted_label})

if __name__ == '__main__':
    app.run(debug=True)