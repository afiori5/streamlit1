import streamlit as st 
import pandas as pd
import altair as alt
import numpy as np

import os
os.chdir("/Users/alexanderfiori/Desktop/streamlit_demos")

s = pd.read_csv("social_media_usage.csv")

st.markdown("#LinkedIn User Prediction App")

"### Select box"
answer = st.selectbox(label="Household Income",
options=("$10,000 to $20,000", "SQL", "Python", "Java", "Go", "C++"))
st.write("Here are some resources for ", answer)
