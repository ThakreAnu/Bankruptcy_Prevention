import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')
from flask import Flask, request,jsonify, render_template
import pickle
#create flask app
app= Flask(__name__)

#load pickle model
dt= pickle.load(open('model.pkl','rb'))

@app.route("/")
def Home():
    return render_template('index1.html')

@app.route("/predict", methods=["POST"])

def predict():
    float_features=[float(x) for x in request.form.values()]
    features=[np.array(float_features)]
    prediction=dt.predict(features)
    if prediction[0]==0:
        ans='Bankrupt'
    else:
        ans='Non-Bankrupt'

    return render_template('index1.html',prediction_text="The Company is "+ans )



if __name__=="__main__":
    app.run(debug= True)

