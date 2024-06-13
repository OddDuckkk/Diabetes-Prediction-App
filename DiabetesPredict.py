from flask import Flask, render_template, request
import pickle
import numpy as np
import pandas as pd
import joblib
from pandas import DataFrame

# Load the machine learning model and transformers
# model1 = pickle.load(open('RFRFE.pkl', 'rb'))
with open('ComRFRFE.pkl', 'rb') as file:
    model1 = joblib.load(file)
model2 = pickle.load(open('RFKBEST.pkl', 'rb'))
quantile = pickle.load(open('quantile.pkl', 'rb'))
scaler = pickle.load(open('scaler.pkl', 'rb'))

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Debugging: Print the form data
        print(request.form)
        
        # Get form data
        age = int(request.form['Age'])
        Sex = int(request.form['Sex'])
        HighChol = int(request.form['HighChol'])
        CholCheck = int(request.form['CholCheck'])
        BMI = float(request.form['BMI'])
        Smoker = int(request.form['Smoker'])
        HeartDiseaseorAttack = int(request.form['HeartDiseaseorAttack'])
        PhysActivity = int(request.form['PhysActivity'])
        Fruits = int(request.form['Fruits'])
        Veggies = int(request.form['Veggies'])
        HvyAlcoholConsump = int(request.form['HvyAlcoholConsump'])
        GenHlth = int(request.form['GenHlth'])
        MentHlth = int(request.form['MentHlth'])
        PhysHlth = int(request.form['PhysHlth'])
        DiffWalk = int(request.form['DiffWalk'])
        Stroke = int(request.form['Stroke'])
        HighBP = int(request.form['HighBP'])
        Diabetes = int(0)

        # Convert age to category
        Age = min(max((age - 18) // 5 + 1, 1), 13)

        # Create a feature array
        features = np.array([[Age, Sex, HighChol, CholCheck, BMI, Smoker, HeartDiseaseorAttack,
                              PhysActivity, Fruits, Veggies, HvyAlcoholConsump, GenHlth, MentHlth,
                              PhysHlth, DiffWalk, Stroke, HighBP, Diabetes]])
        df_input = DataFrame(features)

        # Apply the transformations
        df_quantile = quantile.transform(df_input)
        df_quantile_hasil = DataFrame(df_quantile)
        df_quantile_hasil.columns =['Age','Sex', 'HighChol', 'CholCheck',	'BMI',	'Smoker',	'HeartDiseaseorAttack',	'PhysActivity',	'Fruits', 'Veggies', 'HvyAlcoholConsump',	'GenHlth',	'MentHlth',	'PhysHlth',	'DiffWalk',	'Stroke',	'HighBP', 'Diabetes']
        df_normalization = scaler.transform(df_quantile)
        df_normalization_hasil = DataFrame(df_normalization.round(2))
        df_normalization_hasil.columns =['Age','Sex', 'HighChol', 'CholCheck',	'BMI',	'Smoker',	'HeartDiseaseorAttack',	'PhysActivity',	'Fruits', 'Veggies', 'HvyAlcoholConsump',	'GenHlth',	'MentHlth',	'PhysHlth',	'DiffWalk',	'Stroke',	'HighBP', 'Diabetes']
        df_normalization_hasil = df_normalization_hasil.drop(['Diabetes'],axis=1)

        # Make predictions with both models
        prediction1 = model1.predict(df_normalization_hasil)
        prediction2 = model2.predict(df_normalization_hasil)
        result1 = 'Positive' if prediction1[0] == 1 else 'Negative'
        result2 = 'Positive' if prediction2[0] == 1 else 'Negative'

        # Render the result template with both predictions
        return render_template('result.html', result1=result1, result2=result2)
    except Exception as e:
        return str(e)

if __name__ == '__main__':
    app.run(debug=True)
