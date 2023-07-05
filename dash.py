import streamlit as st
import joblib
from streamlit_js_eval import streamlit_js_eval

with open('bad_model.pkl', 'rb') as f:
    model = joblib.load(f)

    x = st.slider('Select a value')
    st.write(x, 'squared is', x * x)


    if st.button("Reload page"):
        streamlit_js_eval(js_expressions="parent.window.location.reload()")
