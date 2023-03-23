from flask import Flask, render_template, request
import joblib
import pandas as pd

app = Flask(__name__)

model = joblib.load('automobile_model.bz2')
print(model)


@app.route('/')
def index():
    features = ['symboling', 'normalized-losses', 'make', 'fuel-type', 'aspiration',
                'num-of-doors', 'body-style', 'drive-wheels', 'engine-location',
                'wheel-base', 'length', 'width', 'height', 'curb-weight', 'engine-type',
                'num-of-cylinders', 'engine-size', 'fuel-system', 'bore', 'stroke',
                'compression-ratio', 'horsepower', 'peak-rpm', 'city-mpg']
    return render_template('index.html', names=features)


@app.route('/predict', methods=['POST'])
def prediction():
    if request.method == 'POST':
        result = request.form
        df = pd.DataFrame([result])
        df = df.replace('', 0)
        df.fillna(0, inplace=True)
        prediction = model.predict(df.astype(str))
        prediction = round(prediction[0], 2)
        # return {"prediction": str(prediction)}
        return render_template("result.html", result=prediction)
