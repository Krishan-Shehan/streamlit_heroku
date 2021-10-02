import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
from sklearn.ensemble import VotingClassifier
from sklearn.linear_model import LogisticRegression

def app():
    # st.write("""
    # # Explore different classifier and datasets
    # Which one is the best?
    # """)

    data = pd.read_csv("Preprocessed_data.csv")

    classifier_name = st.sidebar.selectbox(
        'Select Classifier',
        ('Decision Tree Classifier', 'Random Forest Classifier', 'K-Neighbors Classifier', 'Gaussian Naives Bayes','Neural-Network Classifier' ,'Voting Classifier')
    )

    def add_parameter_ui(clf_name):
        params = dict()

        if clf_name == 'Decision Tree Classifier':
            params['criterion'] = "gini"
            params['random_state'] = 100
        elif clf_name == 'K-Neighbors Classifier':
            # n_neighbors = st.sidebar.slider('K', 1, 20)
            params['n_neighbors'] = 3
        return params

    params = add_parameter_ui(classifier_name)

    with st.sidebar.form(key='my_form'):
        budget = st.number_input('Enter your budget')
        if budget < 0:
            st.error("Enter a valid budget")
        elif budget > 3000000:
            st.error("Maximum budget is 3000000")
    
        sqfeet = st.number_input('Enter your sqfeet')
        if sqfeet < 0:
            st.error("Enter a valid property area")
        elif sqfeet > 20000:
            st.error("Maximum area allowed is 20000 square feet")

        beds = st.text_input('Enter preffered number of bedrooms')
        baths = st.text_input('Enter preffered number of bathdrooms')
        smoking = st.radio('Smoking allowed', ["Yes", "No"])
        wheelchair = st.radio('Wheelchair access', ["Yes", "No"])
        vehicle = st.radio('Electric vehicle charge access', ["Yes", "No"])
        funrnished = st.radio('Furnished', ["Yes", "No"])
        laundry = st.selectbox('Select laundry option',
                               ('Laundry on site', 'Laundry in building', 'W/D in unit', 'W/D hookups',
                                'No laundry on site'))
        parking = st.selectbox('Select parking options', (
            'Carport', 'Street parking', 'Attached garage', 'Off-street parking', 'Detached garage', 'No parking','Valet parking'))
        state = st.text_input('Enter your state code')
        submit = st.form_submit_button(label='Predict')

    if parking == 'Carport': parking = 4
    if parking == 'Street parking': parking = 1
    if parking == 'Attached garage': parking = 0
    if parking == 'Off-street parking': parking = 2
    if parking == 'Detached garage': parking = 5
    if parking == 'No parking': parking = 3
    if parking == 'Valet parking': parking = 6

    if laundry == 'Laundry on site': laundry = 3
    if laundry == 'Laundry in bldg': laundry = 4
    if laundry == 'W/D in unit': laundry = 0
    if laundry == 'W/D hookups': laundry = 2
    if laundry == 'No laundry on site': laundry = 1

    if smoking == 'Yes': smoking = 1
    if smoking == 'No': smoking = 0

    if wheelchair == 'Yes': wheelchair = 1
    if wheelchair == 'No': wheelchair = 0

    if vehicle == 'Yes': vehicle = 1
    if vehicle == 'No': vehicle = 0

    if funrnished == 'Yes': funrnished = 1
    if funrnished == 'No': funrnished = 0



    def get_classifier(clf_name, params):
        clf = None
        if clf_name == 'Decision Tree Classifier':
            clf = DecisionTreeClassifier(criterion=params['criterion'], random_state=params['random_state'])
        elif clf_name == 'Random Forest Classifier':
            clf = RandomForestClassifier()
        elif clf_name == 'K-Neighbors Classifier':
            clf = KNeighborsClassifier(n_neighbors=3)
        elif clf_name == 'Gaussian Naives Bayes':
            clf = GaussianNB()
        elif clf_name == 'Neural-Network Classifier':
            clf = MLPClassifier()
        elif clf_name == 'Voting Classifier':
            log_clf = LogisticRegression()
            rnd_clf = RandomForestClassifier()
            knn_clf = KNeighborsClassifier()
            svm_clf = SVC()
            clf = VotingClassifier(estimators=[('lr', log_clf), ('rnd', rnd_clf), ('knn', knn_clf)], voting='hard')
        return clf

    clf = get_classifier(classifier_name, params)

    def classify_model(data, model):
        if model == 1:
            X = data.drop(columns=["type", "pets_allowed"])
            Y = data.values[:, 1]
        elif model == 0:
            X = data.drop(columns=["pets_allowed", "type"])
            Y = data.values[:, 12]
        return X, Y

    #### CLASSIFICATION ####

    def multilableclasification():
        X, Y = classify_model(data, 1)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.25, random_state=1)
        clf.fit(X_train, Y_train)
        y_pred = clf.predict(X_test)
        acc = accuracy_score(Y_test, y_pred)
        return y_pred, acc, Y_test

    def bianryclasification():
        X, Y = classify_model(data, 0)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.25, random_state=1)
        clf.fit(X_train, Y_train)
        y_pred = clf.predict(X_test)
        acc = accuracy_score(Y_test, y_pred)
        return y_pred, acc

    def multilableclasification_specific(budget, sqfeet, beds, baths, smoking, wheelchair, vehicle, funrnished, laundry,
                                         parking, state):
        X, Y = classify_model(data, 1)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.25, random_state=1)
        clf.fit(X_train, Y_train)
        return clf.predict(
            [[budget, sqfeet, beds, baths, smoking, wheelchair, vehicle, funrnished, laundry, parking, state]])

    def bianryclasification_specific(budget, sqfeet, beds, baths, smoking, wheelchair, vehicle, funrnished, laundry,
                                     parking, state):
        X, Y = classify_model(data, 0)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.25, random_state=1)
        clf.fit(X_train, Y_train)
        return clf.predict(
            [[budget, sqfeet, beds, baths, smoking, wheelchair, vehicle, funrnished, laundry, parking, state]])

    if submit:
        y_pred2 = multilableclasification_specific(budget, sqfeet, beds, baths, smoking, wheelchair, vehicle,
                                                   funrnished, laundry,
                                                   parking, state)
        y_pred3 = bianryclasification_specific(budget, sqfeet, beds, baths, smoking, wheelchair, vehicle, funrnished,
                                               laundry, parking,
                                               state)
        if y_pred2 == 0: y_pred2 = 'TOWN HOUSE'
        if y_pred2 == 1: y_pred2 = 'CONDOMINIUM'
        if y_pred2 == 2: y_pred2 = 'APARTMENT'
        if y_pred2 == 3: y_pred2 = 'DUPLEX'
        if y_pred3 == 0: y_pred3 = 'PETS NOT ALLOWED'
        if y_pred3 == 1: y_pred3 = 'PETS ALLOWED'

        # st.write(f'Type  = {y_pred2}')
        # st.write(f'Pets allowed = {y_pred3}')

        # classifier = '<p style="font-family:sans-serif; color:Green; font-size: 42px; background-color: red ; height: 200px ; padding: 10px ; width: 100% ">classifier</p>' 
        # st.markdown(classifier, unsafe_allow_html=True)
    
        html_temp = """
        <div style="background-color:#0CB9A0;padding:1.5px">
        <h1 style="font-family:Courier; color:white;text-align:center; font-size:45px">Housing Type Prediction</h1>
        </div><br>"""
        
        st.markdown(html_temp,unsafe_allow_html=True)
        st.title(f'{classifier_name}')
        st.markdown('<style>h1{font-family:Courier; color: #1E8054;}</style>', unsafe_allow_html=True)


        # st.header("Classifier name")
        # st.write(f'{classifier_name}')

        st.header("Type")
        st.write(f'{y_pred2}')

        st.header("Pets allowed")
        st.write(f'{y_pred3}')

        
    

    


    binary_y_pred, binary_acc = bianryclasification()
    multilabel_y_pred, multilabel_acc, Y_test = multilableclasification()
    #st.write(f'Classifier = {classifier_name}')



