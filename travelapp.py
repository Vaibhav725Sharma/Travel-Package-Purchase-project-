import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score,classification_report
import pandas as pd
df = pd.read_csv('modified_df.csv')
df = df.iloc[:,1:]
feature = df.drop('ProdTaken',axis =1)
target = df.ProdTaken
#Train-test split
x_train,x_test,y_train,y_test = train_test_split(feature,target,test_size = 0.2,shuffle = False)
clf = RandomForestClassifier(criterion = 'entropy', max_depth = 7, n_estimators = 250)
clf.fit(x_train,y_train)
header = st.container()
midsection = st.container()
foot = st.container()

with header:
    st.title('Travel Package Purchase Prediction')
    st.image('travel.jpg',width =700)
    st.header('Have a look at our dataset')
    st.write(df.head())
    st.markdown(' *** ')
    col1, col2 = st.columns(2)
    col1.title('List of Features you can study about')
    col1.dataframe(df.columns)
    col2.title('Distribution of Age feature')
    col2.line_chart(df.Age)
    st.markdown(' *** ')

with midsection:
    st.title('Predict the purchases ????')
    st.subheader('Feed all the values to predict the purchase')
    income = st.text_input('Please share your monthly income')
    passport = int(st.checkbox('Do you have passport'))
    pitch = st.text_input('Duration of Pitch')
    designation = st.selectbox('Are you an executive or manager',['Executive','Manager','regular employee'])
    deluxe = int(st.checkbox('Do you need deluxe package'))
    age = st.text_input('Please share your age')
    trips = st.text_input('No. of trip you want')
    gender = st.selectbox('Your Gender',['Male','Female'])
    if gender == 'Male':
        gender = 1
    gender = 0
    
    if designation == 'Executive':
        executive = 1
        manager = 0
    elif designation == 'Manager':
        executive = 0
        manager = 1
    else:
        executive = 0
        manager = 0

    predict = clf.predict(x_test)
    
    if st.button('Predict'):
        pred = clf.predict([[income,passport,pitch,executive,deluxe,manager,age,trips,3,gender]])
        st.write(f'Accuracy Score of prediction is {accuracy_score(y_test,predict)}')
        if pred[0] == 0:
            st.header('Traveller will not purchase')
        else:
            st.header('Traveller is likely to purchase')

    st.markdown(' *** ')

    
with foot:
    st.header('About the Project')
    st.markdown("Travel Package Purchase - It is used to predict the purchasing of product by travelling organization or anybody")
    st.markdown(' *** ')
    st.write('Developed By - Vaibhav Sharma')

