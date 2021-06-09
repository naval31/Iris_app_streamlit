import pandas as pd
import streamlit as st
from sklearn.neighbors import KNeighborsClassifier
from sklearn import datasets

def user_input():
  Sepal_Length=st.sidebar.slider('Sepal Length',4.3,7.9,5.4)  # min,max,initial
  Sepal_Width=st.sidebar.slider('Sepal Width',2.0,4.4,3.4)
  Petal_Length=st.sidebar.slider('Petal Length',1.0,6.9,1.4)
  Petal_Width=st.sidebar.slider('Petal Width',0.1,2.5,0.2)
  data={
      'Sepal_Length':Sepal_Length,
      'Sepal_Width':Sepal_Width,
      'Petal_Length':Petal_Length,
      'Petal_Width':Petal_Width
  }
  features=pd.DataFrame(data,index=[0])
  return features

st.write("#  Simple  ** Iris Flower** Prediction App")
st.write("# Iris Dataset")
st.write("number of classes: 3")
st.sidebar.header('user Input parameters')
st.subheader('User Input Paramteres')

df=user_input()
st.write(df)
iris=datasets.load_iris()
x=iris.data
y=iris.target

#applied KNN algorithm
model=KNeighborsClassifier(n_neighbors=5)
model.fit(x,y)
prediction=model.predict(df)
st.subheader('class Labels  and their corresponding index number')
st.write(iris.target_names)

st.subheader('prediction')
st.write(iris.target_names[prediction])