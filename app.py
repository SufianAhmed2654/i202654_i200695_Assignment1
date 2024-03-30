from flask import Flask, render_template, request
import pickle

app = Flask(__name__)

# Load the model
with open('svm_model.pkl', 'rb') as file:
    model = pickle.load(file)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get the data from the POST request.
    sepal_length = float(request.form['sepal_length'])
    sepal_width = float(request.form['sepal_width'])
    petal_length = float(request.form['petal_length'])
    petal_width = float(request.form['petal_width'])

    # Make prediction using the model loaded
    prediction = model.predict([[sepal_length, sepal_width, petal_length, petal_width]])

    # Take the first value of prediction
    output = prediction[0]

    return render_template('index.html', prediction_text='The predicted Iris species is {}'.format(output))


if __name__ == "__main__":
    app.run(debug=True)
