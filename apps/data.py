import streamlit as st
import numpy as np

import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
from scipy import stats
from mlxtend.preprocessing import TransactionEncoder
def app():


    item = st.sidebar.selectbox(
        'Select Item',
        ('eggs', 'chocolate', 'cookies' , 'eggs' ,'mineral water' , 'spaghetti')
    )

    if (item == "chocolate"):
        st.write("eggs", "mineral water", "spaghetti")
    elif (item == "cookies"):
        st.write("eggs")
    elif (item == "eggs"):
        st.write("chocolate", "cookies", "mineral water", "spaghetti")
    elif (item == "mineral water"):
        st.write("chocolate", "eggs", "spaghetti")
    elif (item == "spaghetti"):
        st.write("chocolate", "eggs", "mineral water")
    else:
        st.write("No related items")
