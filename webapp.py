# Diabetes detection using Machine Learning. 

import pandas as pd 
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from PIL import Image
import streamlit as st

#Create a title and sub-title 

st.write("""
# Diabetes Detection 
Detect if someone has diabetes using ML and python.
""")

# Image 

image = Image.open('C:/Users/Lenovo/Desktop/Diabetes ML/diabetes.png') 
st.image(
    image,
    caption="Diabetes - Machine Learning",
    use_column_width=True)

#Get the data 

df = pd.read_csv('C:/Users/Lenovo/Desktop/Diabetes ML/diabetes.csv')

#Set sub-header
st.subheader('Data Information')

#Show data as a table
st.dataframe(df)

#Show statistics 
st.write(df.describe())

#Show data as chart 
chart = st.bar_chart(df)

#Split data 
X = df.iloc[:,0:8].values
Y = df.iloc[:,-1].values

X_train, X_test , Y_train, Y_test = train_test_split(X,Y,test_size=0.2,random_state=0)

#Get the feature input 
def get_user_input(): 
    preg = st.sidebar.slider('Pregnancies',0,17,3)
    glucose = st.sidebar.slider('Glucose',0,199,117)
    blood = st.sidebar.slider('Blood Pressure',0,122,72)
    skin = st.sidebar.slider('Skin Thickness',0,99,23)
    insulin = st.sidebar.slider('Insulin',0.0,846.0,30.0)
    bmi = st.sidebar.slider('BMI',0.0,67.1,32.0)
    dpf =  st.sidebar.slider('DPF',0.078,2.42,0.3725)
    age = st.sidebar.slider('Age',21,90,29)


    #Store a dictionary into a variable 

    user_data = {
        'pregnancies': preg, 
        'glucose': glucose,
        'blood_pressure' : blood,
        'skin_thickness': skin,
        'insulin' : insulin,
        'BMI': bmi,
        'DPF': dpf,
        'age': age
    }

    #Data-frame

    features = pd.DataFrame(user_data,index=[0])
    return features



#Store user input

user_input = get_user_input()

#Set sub-header to display user input

st.subheader('User Input:')
st.write(user_input)

#Create and train the model 

RandomForestClassifier = RandomForestClassifier()
RandomForestClassifier.fit(X_train,Y_train)

#Show accuracy

st.subheader('Model Test Accuracy Score: ')
st.write(str(accuracy_score(Y_test,RandomForestClassifier.predict(X_test))*100)+'%')


#Store prediction 

prediction = RandomForestClassifier.predict(user_input)

st.subheader('Classification')
st.write(prediction)