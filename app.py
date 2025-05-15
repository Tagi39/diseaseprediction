from flask import Flask, render_template, request
import pickle

app = Flask(__name__)
model, mlb = pickle.load(open("model.pkl", "rb"))

@app.route('/')
def index():
    symptom_list = sorted(mlb.classes_)
    return render_template('index.html', symptoms=symptom_list)

@app.route('/predict', methods=['POST'])
def predict():
    selected = request.form.getlist('symptoms')
    input_data = mlb.transform([selected])
    prediction = model.predict(input_data)
    return render_template('index.html', prediction_text=f"Predicted Disease: {prediction[0]}", symptoms=sorted(mlb.classes_))

if __name__ == '__main__':
    app.run(debug=True)