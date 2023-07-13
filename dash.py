import sklearn.metrics
import streamlit as st
import numpy as np
from attribOptions import ALL_ATTRIBS
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from scipy import stats

predicted_price = None


def get_attrib_type(attribute: str) -> str:
    """ Returns the type of the attribute."""
    return ALL_ATTRIBS[attribute]['_type']


def get_attrib_options(attribute: str) -> list[str]:
    """ Returns the options for the attribute."""
    return ALL_ATTRIBS[attribute]['options']


def get_attrib_value(attribute: str) -> list:
    """ Returns the value of the attribute."""
    return st.session_state[f"data--attrib-{attribute}"]


# ONE-HOT ENCODE
# https://stackabuse.com/one-hot-encoding-in-python-with-pandas-and-scikit-learn/
def one_hot(df, col, pre):
    """ One hot encodes the column and returns the encoded dataframe."""
    o_encoded = pd.get_dummies(df[col], prefix=pre)
    for column in o_encoded:
        o_encoded = o_encoded.rename(columns={column: col + "_" + column})
    o_encoded['Id'] = df['Id']
    return o_encoded


def debug():
    """ Place this anywhere in the app to debug the session state. """

    st.markdown("""
    ## Debugging
    """)
    "### Session State"
    st.session_state

    # Determine the coefficients for each attribute
    "### model_coef_"
    # model.coef_
    raw_coef_list = list(model.coef_)[0]
    coeffs_list = []
    prev_pos = 0
    for ind, attrib in enumerate(st.session_state.attribs):
        if ALL_ATTRIBS[attrib]["_type"] == 'numeric':
            coeffs_list.append([attrib, raw_coef_list[ind]])
            prev_pos += 1
        elif ALL_ATTRIBS[attrib]["_type"] == 'select':
            selected_state = st.session_state[f"data--attrib-{attrib}"]
            option_keys = list(ALL_ATTRIBS[attrib]["options"].keys())
            #  Find the index of the selected option
            coeffs_list.append([attrib, raw_coef_list[prev_pos + option_keys.index(selected_state)]])
            prev_pos += len(ALL_ATTRIBS[attrib]["options"])

    coeffs = pd.DataFrame(coeffs_list, columns=["Attribute", "Coefficient"])
    coeffs.sort_values(by="Coefficient", ascending=False, inplace=True)
    coeffs.reset_index(drop=True, inplace=True)
    st.dataframe(coeffs)

    st.divider()


@st.cache_data
def get_dataset():
    return pd.read_csv("data/train.csv")


@st.cache_resource
def train_model(input_cols: list = ['LotArea']) -> (LinearRegression, pd.DataFrame, pd.DataFrame, list, StandardScaler):
    # data = pd.read_csv("data/train.csv")
    data = get_dataset()

    print("Input cols: ", input_cols)
    # Always need to compare it against the SalePrice to train
    cols = input_cols.copy()
    cols = ['Id', *cols, 'SalePrice']
    data = data[cols]

    # get dataframe of numeric columns (integer or float)
    numeric_df = data.select_dtypes(include=['int64', 'float64'])
    # get columns which are categorical
    non_numeric_columns = data.columns[(data.dtypes != 'int64') & (data.dtypes != 'float64')].tolist()

    # get columns which have NULL values
    null_cols = numeric_df.columns[numeric_df.isna().any()].tolist()
    # impute NULL values with mean of that column
    for col in null_cols:
        mean = numeric_df[col].mean()
        numeric_df[col].fillna(value=mean, inplace=True)

    # Remove outliers more than 3 standard deviations away from the mean
    numeric_df = numeric_df[(np.abs(stats.zscore(numeric_df.drop(columns='Id'))) < 3).all(axis=1)]

    # Scale the data to have mean 0 and variance 1, so we can compare coefficients
    standard_transformer_columns = [col for col in numeric_df.columns if col not in ['Id', 'SalePrice']]
    scaler = StandardScaler()
    numeric_df[standard_transformer_columns] = scaler.fit_transform(numeric_df[standard_transformer_columns])

    # One hot encode the non-numeric columns
    for col in non_numeric_columns:
        encoded = one_hot(data, col, 'is')
        numeric_df = pd.merge(numeric_df, encoded, on='Id', how='left')

    numeric_df.drop(columns=['Id'], inplace=True)
    sale_price = numeric_df.pop('SalePrice')
    numeric_df['SalePrice'] = sale_price

    data_cols = list(numeric_df.columns)

    # Split the data into training/testing sets
    train, test = train_test_split(
        numeric_df, test_size=0.2,
        random_state=12123213)

    reg = LinearRegression().fit(train.loc[:, train.columns != 'SalePrice'], train[['SalePrice']])

    return reg, train, test, data_cols, scaler, standard_transformer_columns


if 'attribs' not in st.session_state or \
        st.session_state.attribs is None or \
        len(st.session_state.attribs) == 0:
    st.warning("No attributes selected. Using default.")
    st.session_state.attribs = ['LotArea']

st.title("House Price Predictor")
model, train, test, data_cols, scaler, standard_transformer_columns = train_model(st.session_state.attribs)

# Transform the test data back to the original scale, for graphing purposes
inverse_transformed_test = test.copy()
inverse_transformed_test[standard_transformer_columns] = scaler.inverse_transform(
    inverse_transformed_test[standard_transformer_columns])


def predict_with_model(model: LinearRegression) -> float:
    """ Predict the price of a house with the given model """

    predict_df = pd.DataFrame(columns=[x for x in data_cols if x != 'SalePrice'])

    for attrib in st.session_state.attribs:
        attrib_type = get_attrib_type(attrib)

        if attrib_type == 'numeric':
            predict_df[attrib] = [get_attrib_value(attrib)]
        elif attrib_type == 'select':
            if 'BldgType' in attrib:
                print("=== Building type options")
                print(get_attrib_options(attrib))
            # Re-one hot encode all the options
            for k, v in get_attrib_options(attrib).items():
                if get_attrib_value(attrib) == k:
                    predict_df[f"{attrib}_is_{k}"] = [1]
                else:
                    predict_df[f"{attrib}_is_{k}"] = [0]
        else:
            raise Exception("Unknown type")

    null_cols = predict_df.columns[predict_df.isna().any()].tolist()
    # impute NULL values of empty one-hot encoded columns with 0
    for col in null_cols:
        predict_df[col].fillna(value=0, inplace=True)

    # transform the columns that have been standard scaled
    predict_df[standard_transformer_columns] = scaler.transform(predict_df[standard_transformer_columns])

    return model.predict(predict_df)[0][0]


########################################################################################################################

"""
## Instructions:

1. Choose which attributes you want to use to predict the house price. We recommend to use the recommended set or choosing 7-9 attributes.
2. Modify the attribute values to match your house's features
3. View the predicted price of your house
4. View how your house's price compares to the other houses on the market.
"""

with st.container():
    "### Use a template, or go custom."
    c1, c2 = st.columns([1, 1])
    if c1.button("Select recommended attributes"):
        st.session_state.attribs = ['BldgType', 'FullBath', 'HalfBath', 'BedroomAbvGr']
        st.experimental_rerun()

    if c2.button("Select all attributes"):
        st.session_state.attribs = list(ALL_ATTRIBS.keys())
        st.experimental_rerun()


def on_multiselect_change():
    # Make sure that at least one attribute is selected
    if st.session_state.attribs is None or len(st.session_state) == 0:
        st.session_state.attribs = ['LotArea']
        st.experimental_rerun()


st.multiselect("Select attributes (we recommend 7-9)", ALL_ATTRIBS.keys(), on_change=on_multiselect_change,
               key="attribs")

"## Modify house parameters"
for attrib in st.session_state.attribs:
    attrib_session_key = f"data--attrib-{attrib}"
    description = f"{attrib} ({ALL_ATTRIBS[attrib]['description']})"
    if ALL_ATTRIBS[attrib]["_type"] == 'numeric':
        st.number_input(description, key=attrib_session_key, step=ALL_ATTRIBS[attrib]["step"])
    elif ALL_ATTRIBS[attrib]["_type"] == 'select':
        st.selectbox(description, key=f"data--attrib-{attrib}", options=ALL_ATTRIBS[attrib]["options"],
                     format_func=lambda x: ALL_ATTRIBS[attrib]["options"][x])  # Format it with human-readable values

########################################################################################################################

"## Predicted Price"

predicted_price = predict_with_model(model)

rmse = mean_squared_error(test[['SalePrice']], model.predict(test.loc[:, test.columns != 'SalePrice']), squared=False)

st.write(
    f"### \${(round(predicted_price) // 1000 * 1000):,d} ± {(round(rmse) // 1000 * 1000):,d}")  # Round to the nearest thousand, format with commas

# R squared
r2 = sklearn.metrics.r2_score(test[['SalePrice']], model.predict(test.loc[:, test.columns != 'SalePrice']))
st.write(f"R²: {r2:.3g} (positive is good, negative is bad)")
if r2 < 0.4:  # Ballpark estimate for a bad model
    st.warning("It looks like this model is not very good. Changing the attributes may help.")

########################################################################################################################

# Scatter plot of this house vs the training dataset
"## Where does your house fit in?"
'Find how your house compares with the market, broken down by house features.'

st.selectbox("Select an attribute to plot against SalePrice", st.session_state.attribs, key="scatterplot_attrib")
scatterplot_attrib = st.session_state.scatterplot_attrib

# Describe the attribute
st.write(f"{scatterplot_attrib} ({ALL_ATTRIBS[scatterplot_attrib]['description']})")

if ALL_ATTRIBS[scatterplot_attrib]["_type"] == 'numeric':
    # If we have a numeric attribute, we can draw a scatter plot
    # Draw a scatter plot of the test data, with a line of best fit
    scatter = px.scatter(inverse_transformed_test, x=scatterplot_attrib, y='SalePrice', trendline='ols')
    # Add where the house is
    scatter.add_scatter(x=[get_attrib_value(scatterplot_attrib)],
                        y=[predicted_price],
                        mode='markers',
                        marker=dict(color='red', size=15, symbol='x'),
                        name='Your house')
    st.plotly_chart(scatter, use_container_width=True)
elif ALL_ATTRIBS[scatterplot_attrib]["_type"] == 'select':
    # If we have a categorical attribute, we can draw multiple boxplots
    # e.g. if scatterplot_attrib is "LandContour", then colnames will be
    # [
    #     "LandContour_is_Bnk",
    #     "LandContour_is_HLS",
    #     "LandContour_is_Low",
    #     "LandContour_is_Lvl"
    # ]
    colnames = [x for x in test.columns if scatterplot_attrib in x]

    boxplot = go.Figure()

    for colname in colnames:
        boxplot.add_trace(go.Box(y=inverse_transformed_test.loc[test[colname] == 1, 'SalePrice'], name=colname))
        boxplot.update_xaxes(title_text=scatterplot_attrib)
        boxplot.update_yaxes(title_text="SalePrice ($)")

    # Add a marker to the right column
    scatterplot_attrib_value = get_attrib_value(scatterplot_attrib)
    scatterplot_attrib_session_key = f"{scatterplot_attrib}_is_{scatterplot_attrib_value}"
    boxplot.add_scatter(x=[scatterplot_attrib_session_key], y=[predicted_price],
                        mode='markers',
                        marker=dict(color='red', size=15, symbol='x'),
                        name='Your house')

    # Display the plot
    st.plotly_chart(boxplot, use_container_width=True)

########################################################################################################################

# Plot a histogram of the predicted prices
"## Predicted Price Distribution"
st.write(
    "Compare your house's value with the current listed houses on the market. Your house is marked with a red line.")
predicted_price_histogram = px.histogram(test, x='SalePrice')
# Set the column that "predicted_price" is in to red
predicted_price_histogram.add_vline(x=predicted_price, line_width=3, line_dash="dash", line_color="red")
st.plotly_chart(predicted_price_histogram, use_container_width=True)

########################################################################################################################

'## Model Coefficient Correlations'
'This shows how common a feature of a house is with another feature of a house.'
# Df of coefficients, correlation
df_train_corr = train.corr()
st.plotly_chart(px.imshow(df_train_corr), use_container_width=True)

########################################################################################################################

'## Most important features'
'These are the features that most influence the price of a house'
# Get the 3 most positive coefficients, and the 3 most negative coefficients

coefficients = pd.DataFrame(np.transpose(model.coef_), index=[x for x in train.columns if x != 'SalePrice'],
                            columns=['Coefficient'])

# Get the 3 most positive coefficients
most_positive = coefficients.sort_values(by='Coefficient', ascending=False).head(3)
# Get the 3 most negative coefficients
most_negative = coefficients.sort_values(by='Coefficient', ascending=False).tail(3)
# Combine them
most_important = pd.concat([most_positive, most_negative])
# Graph them

bar_graph = px.bar(most_important, x=most_important.index, y='Coefficient')
bar_graph.update_xaxes(title_text="Attribute")
bar_graph.update_yaxes(title_text="Price influence ($)")

# Set the first 3 columns to green, the last 3 to red
bar_graph.update_traces(marker_color=['green'] * 3 + ['red'] * 3)

st.plotly_chart(bar_graph, use_container_width=True)
