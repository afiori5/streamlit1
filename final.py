import streamlit as st 
import pandas as pd
import altair as alt
import numpy as np

import os
os.chdir("/Users/alexanderfiori/Desktop/streamlit1")

s = pd.read_csv("social_media_usage.csv")

st.markdown("#LinkedIn User Prediction App")

"### Select box"
Income = st.selectbox(label="Household Income",
options=("$10,000 to $20,000", 
"$20,000 to $30,000", 
"$40,000 to $50,000", 
"$50,000 to $75,000", 
"$100,000 to $150,000", 
"$150,000+", 
"Don't Know"))
