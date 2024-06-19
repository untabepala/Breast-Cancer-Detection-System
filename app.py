from flask import Flask, request, render_template
import pandas
import numpy as np
import pickle

model = pickle.load(open("model.pkl",'rb'))

#flask app
app=Flask(__name__)

@app.route('/')
def index():
    return render_template("index.html")

@app.route("/predict",methods=['POST'])
def predict():
    features=request.form['feature']
    features_lst=features.split(',')
    np_features=np.asarray(features_lst,dtype=np.float32)
    pred=model.predict(np_features.reshape(1,-1))



    output=["Not Cancer" if pred[0] == 1 else "Cancer"]
    return render_template('index.html',message= output)

#python main
if __name__ == '__main__':
    app.run(debug=True)

