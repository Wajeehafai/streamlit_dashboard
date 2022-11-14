import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import accuracy_score, r2_score, mean_squared_error, mean_absolute_error


# make containers
header =st.container()
data_sets= st.container()
features = st.container()
model_training = st.container()

with header:
    st.title('Kashti App')
    st.text('This is a simple app to explore the titanic dataset')
    
    
    
with data_sets:
    st.header('Titanic Dataset')
    st.text('Just starting to explore the dataset')
    # import data
    df = sns.load_dataset('titanic')
    df.dropna(inplace=True)
    st.write(df.head(10))
    st.subheader('Class k plot')
    # plot bar plot
    st.bar_chart(df['class'].value_counts())
    # other plots
    st.subheader('Age k plots')
    st.line_chart(df['age'].value_counts())
        
with features:
    st.header('Classes in kashti')
    st.text('three classes')
    st.markdown(' _Feature 1_ . First Class')
    
    
with model_training:
    st.header('what happened to the passengers?')
    st.text('they died, some survived')

    # making columns
    input, display = st.columns(2)
    # In first column
    max_depth =input.slider('How many people died?', min_value=10,
                 max_value=100, value=50, step=10) 
n_estimators = input.selectbox('How many people survived?', options=[20,30, 40,'no values'])

# adding list of features

input.write(df.columns)


# input features from users 
input_features =input.text_input('Which features we want to use?')
    
# Machine learning model
    
model = RandomForestRegressor(max_depth=max_depth, n_estimators=n_estimators)

if n_estimators == 'no values':
    model = RandomForestRegressor(max_depth=max_depth)
else:
    model = RandomForestRegressor(max_depth=max_depth, n_estimators=n_estimators)
# define X and y
X= df[[input_features]]
y= df[['survived']]
    
model.fit(X,y)
# predict
pred = model.predict(y)
# display metrics

display.subheader('r2_score')
display.write(r2_score(y, pred))
display.subheader('mean_squared_error')
display.write(mean_absolute_error(y, pred))
display.subheader('mean_absolute_error')
display.write(mean_absolute_error(y, pred))
