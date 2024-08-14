from flask import Flask, render_template, request, jsonify
import numpy as np
import pickle


app = Flask(__name__)

# load the model from disk
model = pickle.load(open('model/model.pkl', 'rb'))


@app.route("/", methods=['GET'])
def welcome():
    return render_template('index.html')


@app.route("/predict", methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    int_features = [int(x) for x in request.form.values()]
    final_features = [np.array(int_features)]
    prediction = model.predict(final_features)

    output = round(prediction[0])

    return render_template('index.html', prediction_text='Price should be Rs. {}'.format(output))


@app.route('/predict_api',methods=['POST'])
def predict_api():
    '''
    For direct API calls through request
    '''
    data = request.get_json(force=True)
    prediction = model.predict([np.array(list(data.values()))])

    output = prediction[0]
    return jsonify(output)


if __name__ == '__main__':
    app.run(debug=True)