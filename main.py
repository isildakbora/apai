import numpy as np
import pandas as pd
import streamlit as st


import sys
from joblib import load

from PIL import Image
image = Image.open('logo.png')

st.image(image, caption='Sunrise by the mountains')

model_path = 'rf_pdw_crp_mcv.joblib'

model = load(model_path)
pdw = st.slider('Your PDW value:', min_value=0., max_value=20., step=0.1)
mcv = st.slider('Your MCW value:', min_value=0., max_value=200., step=0.1)
crp = st.slider('Your CRP value:', min_value=0., max_value=100., step=0.1)

# girdiler pdw mcv crp seklinde siralanmali
X_pred = [pdw, mcv, crp]
X_pred = [[float(arg) for arg in X_pred[0:]]]

pred = model.predict(X_pred)

# 1 --> non complicated , 2 --> complicated
if pred[0] == 1:
    st.header('\nnot complicated!')
else:
    st.header('\ncomplicated!')
