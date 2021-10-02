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
        'Select classifier',
        ('DecisionTreeClassifier', 'RandomForestClassifier', 'KNeighborsClassifier', 'GaussianNB','MLPClassifier' ,'VotingClassifier')
    )

    def add_parameter_ui(clf_name):
        params = dict()

        if clf_name == 'DecisionTreeClassifier':
            params['criterion'] = "gini"
            params['random_state'] = 100
        elif clf_name == 'KNeighborsClassifier':
            # n_neighbors = st.sidebar.slider('K', 1, 20)
            params['n_neighbors'] = 3
        return params

    params = add_parameter_ui(classifier_name)

    with st.sidebar.form(key='my_form'):
        budget = st.number_input('Enter your budget')
        sqfeet = st.text_input('Enter your sqfeet')
        beds = st.text_input('Enter preffered number of bedrooms')
        baths = st.text_input('Enter preffered number of bathdrooms')
        smoking = st.selectbox('Smoking allowed', [1, 0])
        wheelchair = st.selectbox('Wheelchair access', [1, 0])
        vehicle = st.selectbox('Electric vehicle charge access', [1, 0])
        funrnished = st.selectbox('Furnished', [1, 0])
        laundry = st.selectbox('Select laundry option',
                               ('laundry on site', 'laundry in bldg', 'w/d in unit', 'w/d hookups',
                                'no laundry on site'))
        parking = st.selectbox('Select parking options', (
            'carport', 'street parking', 'attached garage', 'off-street parking', 'detached garage', 'no parking',
            'valet parking'))
        state = st.text_input('Enter your state')
        # pets_allowed = st.selectbox('pets allowed', [1, 0])
        submit = st.form_submit_button(label='Submit')

    if parking == 'carport': parking = 4
    if parking == 'street parking': parking = 1
    if parking == 'attached garage': parking = 0
    if parking == 'off-street parking': parking = 2
    if parking == 'detached garage': parking = 5
    if parking == 'no parking': parking = 3
    if parking == 'valet parking': parking = 6

    if laundry == 'laundry on site': laundry = 3
    if laundry == 'laundry in bldg': laundry = 4
    if laundry == 'w/d in unit': laundry = 0
    if laundry == 'w/d hookups': laundry = 2
    if laundry == 'no laundry on site': laundry = 1

    def get_classifier(clf_name, params):
        clf = None
        if clf_name == 'DecisionTreeClassifier':
            clf = DecisionTreeClassifier(criterion=params['criterion'], random_state=params['random_state'])
        elif clf_name == 'RandomForestClassifier':
            clf = RandomForestClassifier()
        elif clf_name == 'KNeighborsClassifier':
            clf = KNeighborsClassifier(n_neighbors=3)
        elif clf_name == 'GaussianNB':
            clf = GaussianNB()
        elif clf_name == 'MLPClassifier':
            clf = MLPClassifier()
        elif clf_name == 'VotingClassifier':
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
        if y_pred2 == 0: y_pred2 = 'townhouse'
        if y_pred2 == 1: y_pred2 = 'condo'
        if y_pred2 == 2: y_pred2 = 'apartment'
        if y_pred2 == 3: y_pred2 = 'duplex'
        if y_pred3 == 0: y_pred3 = 'Pets not allowed'
        if y_pred3 == 1: y_pred3 = 'Pets allowed'

        st.write(f'Type  = {y_pred2}')
        st.write(f'Pets allowed = {y_pred3}')

    binary_y_pred, binary_acc = bianryclasification()
    multilabel_y_pred, multilabel_acc, Y_test = multilableclasification()
    st.write(f'Classifier = {classifier_name}')



