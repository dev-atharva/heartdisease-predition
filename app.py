import numpy as np
import pickle
import os
from sklearn.preprocessing import StandardScaler
from flask import Flask, request, render_template
Scaler=StandardScaler()
numerical_var=[]
app = Flask(__name__)
model = pickle.load(open('finalized_model.pkl','rb'))
@app.route('/')
def home():
    return render_template('index1.html')

@app.route('/predict',methods=['POST'])
def predict():
    features=[float(i) for i in request.form.values()]
    numerical_var = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']
    array_features = [np.array(features)]
    Scaler.fit_transform(array_features)
    prediction = model.predict(array_features)
    output = prediction

    if output == 1:
        return render_template('index1.html',
        result = 'The patient is not likely to have heart disease!')
    else:
        return render_template('index1.html',
        result = 'The patient is likely to have heart disease!')

if __name__ == '__main__':
#Run the application
    app.run(debug=True)