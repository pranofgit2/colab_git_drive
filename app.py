import numpy as np
from flask import Flask, request, jsonify, render_template
import tensorflow
from tensorflow import keras
import joblib

app = Flask(__name__)
model = keras.models.load_model('./md.h5')

@app.route('/')
def home():
    return render_template('index.html')
  
@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    #model = joblib.load("./md.pkl")
    features =  [float(x) for x in request.form.values()]########## change##########pipeline########
    final_features = np.reshape(np.array(features),(1,4))################# change##########pipeline#########
    prediction =np.argmax(model.predict(final_features))
    #prediction=1
    
    
    output='Error'

    if prediction==0:
        output='Iris-setosa'
    elif prediction==1:
        output='Iris-versicolor'
    else:
        output='Iris-virginica'
            

    return render_template('index.html', prediction_text='The Iris plant spcies is  {0}{1}{2}'.format(output,final_features+100,'a'))


if __name__ == "__main__":
    app.run(debug=True)
