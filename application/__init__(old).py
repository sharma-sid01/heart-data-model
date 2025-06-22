# Import packages
from flask import Flask, request, Response, json
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# Load data
path = "./heartData/heart_data.csv"
df = pd.read_csv(path)

#drop index, id columns                 
df.drop(df.iloc[:,0:2], inplace =True, axis =1)

#Translate age in days to age in years - divide by 365.25 and convert to int
ageInYrs =  df['age'] = (df['age']/ 365.25).astype(int)  

#skip negative ap hi and lo values
df.drop(df[df['ap_hi'] < 20].index, inplace=True)
df.drop(df[df['ap_lo'] < 20].index, inplace=True)

df.drop(df[df['ap_hi'] > 900].index, inplace=True)
df.drop(df[df['ap_lo'] > 910].index, inplace=True)

#Standardize data
age_std_scale = StandardScaler()
gender_std_scale = StandardScaler()
height_std_scale = StandardScaler()
weight_std_scale = StandardScaler()
aphi_std_scale = StandardScaler()
aplo_std_scale = StandardScaler()
cholesterol_std_scale = StandardScaler()
gluc_std_scale = StandardScaler()
smoke_std_scale = StandardScaler()
alco_std_scale = StandardScaler()
active_std_scale = StandardScaler()

df['age'] = age_std_scale.fit_transform(df[['age']])
df['gender'] = gender_std_scale.fit_transform(df[['gender']])
df['height'] = height_std_scale.fit_transform(df[['height']])
df['weight'] = weight_std_scale.fit_transform(df[['weight']])
df['ap_hi'] = aphi_std_scale.fit_transform(df[['ap_hi']])
df['ap_lo'] = aplo_std_scale.fit_transform(df[['ap_lo']])
df['cholesterol'] = cholesterol_std_scale.fit_transform(df[['cholesterol']])
df['gluc'] = gluc_std_scale.fit_transform(df[['gluc']])
df['smoke'] = smoke_std_scale.fit_transform(df[['smoke']])
df['alco'] = alco_std_scale.fit_transform(df[['alco']])
df['active'] = active_std_scale.fit_transform(df[['active']])

# Modeling - Logistic Regression
X = df[['age','gender','height','weight','ap_hi','ap_lo','cholesterol','gluc','smoke','alco','active']]
y = df['cardio']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

#Create flask instance
app = Flask(__name__)

#if __name__ == "__main__":
#    app.run(debug=True)

#Create API
@app.route('/api', methods=['GET','POST'])
def predict():
    #get data from request
    data = request.get_json(force=True)
    
    data_age = np.array([data["age"]])
    data_age = np.reshape(data_age, (1, -1))
    data_age = np.array(age_std_scale.transform(data_age))

    data_gender = np.array([data["gender"]])
    data_gender= np.reshape(data_gender, (1, -1))
    data_gender = np.array(gender_std_scale.transform(data_gender))

    data_height = np.array([data["height"]])
    data_height= np.reshape(data_height, (1, -1))
    data_height = np.array(height_std_scale.transform(data_height))
    
    data_weight = np.array([data["weight"]])
    data_weight= np.reshape(data_weight, (1, -1))
    data_weight = np.array(weight_std_scale.transform(data_weight))

    data_ap_hi = np.array([data["ap_hi"]])
    data_ap_hi= np.reshape(data_ap_hi, (1, -1))
    data_ap_hi = np.array(aphi_std_scale.transform(data_ap_hi))

    data_ap_lo = np.array([data["ap_lo"]])
    data_ap_lo= np.reshape(data_ap_lo, (1, -1))
    data_ap_lo = np.array(aplo_std_scale.transform(data_ap_lo))

    data_cholesterol = np.array([data["cholesterol"]])
    data_cholesterol= np.reshape(data_cholesterol, (1, -1))
    data_cholesterol = np.array(cholesterol_std_scale.transform(data_cholesterol))

    data_gluc = np.array([data["gluc"]])
    data_gluc= np.reshape(data_gluc, (1, -1))
    data_gluc = np.array(gluc_std_scale.transform(data_gluc))

    data_smoke = np.array([data["smoke"]])
    data_smoke= np.reshape(data_smoke, (1, -1))
    data_smoke = np.array(aphi_std_scale.transform(data_smoke))

    data_alco = np.array([data["alco"]])
    data_alco= np.reshape(data_alco, (1, -1))
    data_alco = np.array(aplo_std_scale.transform(data_alco))

    data_active = np.array([data["active"]])
    data_active= np.reshape(data_active, (1, -1))
    data_active = np.array(active_std_scale.transform(data_active))

    data_final = np.column_stack((data_age, data_gender, data_height, data_weight, data_ap_hi, data_ap_lo, data_cholesterol, data_gluc, data_smoke, data_alco, data_active))
    data_final = pd.DataFrame(data_final, dtype=object)

    #make predicon using model
    prediction = model.predict(data_final)
    # return Response(json.dumps(prediction[0]))
    return Response(json.dumps(int(prediction[0])))
