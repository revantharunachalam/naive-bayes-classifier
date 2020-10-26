import numpy as np
import pickle
from flask import Flask, request, jsonify, render_template

app = Flask(__name__)
app.static_folder = 'static'
NB_classifier = pickle.load(open('model.pkl', 'rb'))

@app.route('/', methods=['GET'])
def home():
    return render_template('index.html', class_title="Classifier", class_img="https://specials-images.forbesimg.com/imageserve/933666298/960x0.jpg?fit=scale")


@app.route('/predict', methods=['POST'])
def predict():
    '''
    For rendering results on html GUI 
    
    '''
    
    length = float(request.form.get("length"))
    width = float(request.form.get("width"))
    features = np.array([length, width]).reshape(1, 2)
    
    result = NB_classifier.predict(features)[0]
    
    # UI update
    if result == 0:
        return render_template('index.html', class_title="Species 1: Iris Versicolor", class_img="https://upload.wikimedia.org/wikipedia/commons/d/db/Iris_versicolor_4.jpg")
    elif result == 1:
        return render_template('index.html', class_title="Species 2: Iris Virginica", class_img="https://www.fs.fed.us/wildflowers/beauty/iris/Blue_Flag/images/iris_virginica/iris_virginica_virginica_lg.jpg")
    elif result == 2:
        return render_template('index.html', class_title="Species 3: Iris Setosa", class_img="https://alchetron.com/cdn/iris-setosa-0ab3145a-68f2-41ca-a529-c02fa2f5b02-resize-750.jpeg")
    else:
        return render_template('index.html', class_title="Error!!!", class_img="https://miro.medium.com/max/978/1*pUEZd8z__1p-7ICIO1NZFA.png")

if __name__ == "__main__": 
    app.run(debug=True)