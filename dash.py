import streamlit as st
import joblib
import numpy as np
# from streamlit_js_eval import streamlit_js_eval
from st_aggrid import AgGrid
from st_aggrid.grid_options_builder import GridOptionsBuilder


predicted_price = None

# with open('bad_model.pkl', 'rb') as f:
model = joblib.load('bad_model.pkl')

st.title("House Price Predictor")

predictTab, viewTab = st.tabs(['View Listings', 'Predict Your Own'])

viewTab.dataframe(model.coef_)




if st.button("Predict"):
    # a_lol[['LotArea', 'BedroomAbvGr', 'YearBuilt', 'FullBath', 'HalfBath']]
    predicted_price = model.predict(np.array([]))

if predicted_price != None:
    st.write("Predicted price: ", predicted_price)


list_price = st.number_input("List Price", value=100000)

x = st.slider('Select a value')
st.write(x, 'squared is', x * x)



# if st.button("Reload page"):
#     streamlit_js_eval(js_expressions="parent.window.location.reload()")

st.write("The model is", model)



