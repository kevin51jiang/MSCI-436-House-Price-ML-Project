import sklearn.metrics
import streamlit as st
from st_aggrid import AgGrid
from st_aggrid.grid_options_builder import GridOptionsBuilder
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

import math

predicted_price = None


def format_number(x):
    return '{:.2e}'.format(x)


millnames = ['', ' thousand', ' million', ' billion', ' trillion']


def millify(n):
    n = float(n)
    millidx = max(0, min(len(millnames) - 1,
                         int(math.floor(0 if n == 0 else math.log10(abs(n)) / 3))))

    return '{:.0f}{}'.format(n / 10 ** (3 * millidx), millnames[millidx])


@st.cache_resource
def make_dummy_model(input_cols: list = ['LotArea']) -> (LinearRegression, pd.DataFrame, pd.DataFrame):
    data = pd.read_csv("data/train.csv")

    print("Input cols: ", input_cols)
    # Always need to compare it against the SalePrice to train
    cols = input_cols.copy()
    cols = ['Id', *cols, 'SalePrice']

    data = data[cols]

    ## START ACTUAL MODEL
    numeric_df = data.select_dtypes(include=['int64', 'float64'])
    non_numeric_columns = data.columns[(data.dtypes != 'int64') & (data.dtypes != 'float64')].tolist()

    # ONE-HOT ENCODE
    # https://stackabuse.com/one-hot-encoding-in-python-with-pandas-and-scikit-learn/
    def one_hot(df, col, pre):
        o_encoded = pd.get_dummies(df[col], prefix=pre)
        for column in o_encoded:
            o_encoded = o_encoded.rename(columns={column: col + "_" + column})
        o_encoded['Id'] = df['Id']
        return o_encoded

    for col in non_numeric_columns:
        encoded = one_hot(data, col, 'is')
        numeric_df = pd.merge(numeric_df, encoded, on='Id', how='left')

    null_cols = numeric_df.columns[numeric_df.isna().any()].tolist()

    # impute NULL values with mean (LotFrontage, MasVnrArea, GarageYrBlt)
    for col in null_cols:
        mean = numeric_df[col].mean()
        numeric_df[col].fillna(value=mean, inplace=True)

    numeric_df.drop(columns=['Id'], inplace=True)
    sale_price = numeric_df.pop('SalePrice')
    numeric_df['SalePrice'] = sale_price

    # "Columns"
    # st.write(list(numeric_df.columns))
    data_cols = list(numeric_df.columns)

    ## END ACTUAL MODEL

    # Split the data into training/testing sets
    train, test = train_test_split(
        numeric_df, test_size=0.2,
        random_state=12123213)

    reg = LinearRegression().fit(train.loc[:, train.columns != 'SalePrice'], train[['SalePrice']])
    # reg = LinearRegression().fit(train.loc[:, train.columns != 'SalePrice'], train[['SalePrice']])
    # reg.coef_

    return reg, train, test, data_cols


if 'attribs' not in st.session_state or \
        st.session_state.attribs is None or \
        len(st.session_state.attribs) == 0:
    st.warning("No attributes selected. Using default.")
    st.session_state.attribs = ['LotArea']

st.title("House Price Predictor")

# model, train, test = None, None, None

model, train, test, data_cols = make_dummy_model(st.session_state.attribs)


def predict_with_model(model: LinearRegression) -> float:
    predict_vals = []
    session_attribs = st.session_state
    for attrib in st.session_state.attribs:
        if ALL_ATTRIBS[attrib]['_type'] == 'numeric':
            predict_vals.append(session_attribs[f"data--attrib-{attrib}"])
        elif ALL_ATTRIBS[attrib]['_type'] == 'select':
            # Re-one hot encode all the options
            for k, v in ALL_ATTRIBS[attrib]['options'].items():
                if session_attribs[f"data--attrib-{attrib}"] == k:
                    predict_vals.append(1)
                else:
                    predict_vals.append(0)
        else:
            raise Exception("Unknown type")

    return model.predict([predict_vals])[0][0]


ALL_ATTRIBS = {
    'MSSubClass': {"_type": 'numeric', 'description': 'Identifies the type of dwelling involved in the sale.',
                   "step": 1},
    'MSZoning': {"_type": 'select', 'description': 'Identifies the general zoning classification of the sale.',
                 "options": {
                     'C (all)': 'Commercial (all)',
                     'FV': 'Floating Village Residential',
                     'RH': 'Residential High Density',
                     'RL': 'Residential Low Density',
                     'RM': 'Residential Medium Density'
                 }},
    'OverallQual': {"_type": 'numeric',
                    "description": "Rates the overall material and finish of the house. 10 is best.", 'step': 1},
    'OverallCond': {"_type": 'numeric', "description": "Rates the overall condition of the house. 10 is best.",
                    'step': 1},
    'YearBuilt': {"_type": 'numeric', "description": "Original construction date", 'step': 1},
    'YearRemodAdd': {"_type": 'numeric',
                     "description": "Remodel date (same as construction date if no remodeling or additions)",
                     'step': 1},
    'MasVnrArea': {"_type": 'numeric', "description": "Masonry veneer area in square feet", 'step': 1},
    'BsmtFinSF1': {"_type": 'numeric', "description": "Type 1 finished square feet", 'step': 1},
    'BsmtFinSF2': {"_type": 'numeric', "description": "Type 2 finished square feet", 'step': 1},
    'LotFrontage': {"_type": 'numeric', "description": "Linear feet of street connected to property", "step": 1},
    'LotArea': {"_type": 'numeric', "description": "Lot size in square feet", "step": 100},
    'Street': {"_type": 'select', "description": "Type of road access to property",
               "options": {
                   'Grvl': 'Gravel',
                   'Pave': 'Paved'
               }},
    'Alley': {"_type": 'select', "description": "Type of alley access to property",
              "options": {
                  'Grvl': 'Gravel',
                  'Pave': 'Paved'
              }},
    'LotShape': {"_type": 'select', "description": "General shape of property",
                 "options": {
                     'Reg': 'Regular',
                     'IR1': 'Slightly irregular',
                     'IR2': 'Moderately Irregular',
                     'IR3': 'Irregular'
                 }},
    'LandContour': {"_type": 'select', "description": "Flatness of the property",
                    "options": {
                        'Lvl': 'Near Flat/Level',
                        'Bnk': 'Banked - Quick and significant rise from street grade to building',
                        'HLS': 'Hillside - Significant slope from side to side',
                        'Low': 'Depression'
                    }},
    'Utilities': {"_type": 'select', "description": "Type of utilities available",
                  "options": {
                      'AllPub': 'All public Utilities (E,G,W,& S)',
                      'NoSeWa': 'Electricity and Gas Only',
                  }},
    'LotConfig': {"_type": 'select', "description": "Lot configuration",
                  "options": {
                      'Inside': 'Inside lot',
                      'Corner': 'Corner lot',
                      'CulDSac': 'Cul-de-sac',
                      'FR2': 'Frontage on 2 sides of property',
                      'FR3': 'Frontage on 3 sides of property'
                  }},
    'LandSlope': {"_type": 'select', "description": "Slope of property",
                  "options": {
                      'Gtl': 'Gentle slope',
                      'Mod': 'Moderate Slope',
                      'Sev': 'Severe Slope'
                  }},
    'Neighborhood': {"_type": 'select', "description": "Physical locations within Ames city limits",
                     "options": {
                         'Blmngtn': 'Bloomington Heights',
                         'Blueste': 'Bluestem',
                         'BrDale': 'Briardale',
                         'BrkSide': 'Brookside',
                         'ClearCr': 'Clear Creek',
                         'CollgCr': 'College Creek',
                         'Crawfor': 'Crawford',
                         'Edwards': 'Edwards',
                         'Gilbert': 'Gilbert',
                         'IDOTRR': 'Iowa DOT and Rail Road',
                         'MeadowV': 'Meadow Village',
                         'Mitchel': 'Mitchell',
                         'Names': 'North Ames',
                         'NoRidge': 'Northridge',
                         'NPkVill': 'Northpark Villa',
                         'NridgHt': 'Northridge Heights',
                         'NWAmes': 'Northwest Ames',
                         'OldTown': 'Old Town',
                         'SWISU': 'South & West of Iowa State University',
                         'Sawyer': 'Sawyer',
                         'SawyerW': 'Sawyer West',
                         'Somerst': 'Somerset',
                         'StoneBr': 'Stone Brook',
                         'Timber': 'Timberland',
                         'Veenker': 'Veenker'
                     }},
    'Condition1': {"_type": 'select', "description": "Proximity to various conditions",
                   "options": {
                       'Artery': 'Adjacent to arterial street',
                       'Feedr': 'Adjacent to feeder street',
                       'Norm': 'Normal',
                       'RRNn': 'Within 200 of North-South Railroad',
                       'RRAn': 'Adjacent to North-South Railroad',
                       'PosN': 'Near positive off-site feature--park, greenbelt, etc.',
                       'PosA': 'Adjacent to postive off-site feature',
                       'RRNe': 'Within 200 of East-West Railroad',
                       'RRAe': 'Adjacent to East-West Railroad'
                   }},
    'Condition2': {"_type": 'select', "description": "Proximity to various conditions (if more than one is present)",
                   "options": {
                       'Artery': 'Adjacent to arterial street',
                       'Feedr': 'Adjacent to feeder street',
                       'Norm': 'Normal',
                       'RRNn': 'Within 200 of North-South Railroad',
                       'RRAn': 'Adjacent to North-South Railroad',
                       'PosN': 'Near positive off-site feature--park, greenbelt, etc.',
                       'PosA': 'Adjacent to postive off-site feature',
                       'RRAe': 'Adjacent to East-West Railroad'
                   }},
    'BldgType': {"_type": 'select', "description": "Type of dwelling",
                 "options": {
                     '1Fam': 'Single-family Detached',
                     '2FmCon': 'Two-family Conversion; originally built as one-family dwelling',
                     'Duplx': 'Duplex',
                     'TwnhsE': 'Townhouse End Unit',
                     'TwnhsI': 'Townhouse Inside Unit'
                 }},
    'HouseStyle': {"_type": 'select', "description": "Style of dwelling",
                   "options": {
                       '1Story': 'One story',
                       '1.5Fin': 'One and one-half story: 2nd level finished',
                       '1.5Unf': 'One and one-half story: 2nd level unfinished',
                       '2Story': 'Two story',
                       '2.5Fin': 'Two and one-half story: 2nd level finished',
                       '2.5Unf': 'Two and one-half story: 2nd level unfinished',
                       'SFoyer': 'Split Foyer',
                       'SLvl': 'Split Level'
                   }},

    'RoofStyle': {"_type": 'select', "description": "Type of roof",
                  "options": {
                      'Flat': 'Flat',
                      'Gable': 'Gable',
                      'Gambrel': 'Gabrel (Barn)',
                      'Hip': 'Hip',
                      'Mansard': 'Mansard',
                      'Shed': 'Shed'
                  }},
    'RoofMatl': {"_type": 'select', "description": "Roof material",
                 "options": {
                     'ClyTile': 'Clay or Tile',
                     'CompShg': 'Standard (Composite) Shingle',
                     'Membran': 'Membrane',
                     'Metal': 'Metal',
                     'Roll': 'Roll',
                     'Tar&Grv': 'Gravel & Tar',
                     'WdShake': 'Wood Shakes',
                     'WdShngl': 'Wood Shingles'
                 }},
    'Exterior1st': {"_type": 'select', "description": "Exterior covering on house",
                    "options": {
                        'AsbShng': 'Asbestos Shingles',
                        'AsphShn': 'Asphalt Shingles',
                        'BrkComm': 'Brick Common',
                        'BrkFace': 'Brick Face',
                        'CBlock': 'Cinder Block',
                        'CemntBd': 'Cement Board',
                        'HdBoard': 'Hard Board',
                        'ImStucc': 'Imitation Stucco',
                        'MetalSd': 'Metal Siding',
                        'Plywood': 'Plywood',
                        'Stone': 'Stone',
                        'Stucco': 'Stucco',
                        'VinylSd': 'Vinyl Siding',
                        'Wd Sdng': 'Wood Siding',
                        'WdShing': 'Wood Shingles'
                    }},
    'Exterior2nd': {"_type": 'select', "description": "Exterior covering on house (if more than one material)",
                    "options": {
                        'AsbShng': 'Asbestos Shingles',
                        'AsphShn': 'Asphalt Shingles',
                        'Brk Cmn': 'Brick Common',
                        'BrkFace': 'Brick Face',
                        'CBlock': 'Cinder Block',
                        'CmentBd': 'Cement Board',
                        'HdBoard': 'Hard Board',
                        'ImStucc': 'Imitation Stucco',
                        'MetalSd': 'Metal Siding',
                        'Other': 'Other',
                        'Plywood': 'Plywood',
                        'Stone': 'Stone',
                        'Stucco': 'Stucco',
                        'VinylSd': 'Vinyl Siding',
                        'Wd Sdng': 'Wood Siding',
                        'Wd Shng': 'Wood Shingles'
                    }},
    'MasVnrType': {"_type": 'select', "description": "Masonry veneer type",
                   "options": {
                       'BrkCmn': 'Brick Common',
                       'BrkFace': 'Brick Face',
                       'Stone': 'Stone'
                   }},

    'ExterQual': {"_type": 'select', "description": "Evaluates the quality of the material on the exterior",
                  "options": {
                      'Ex': 'Excellent',
                      'Gd': 'Good',
                      'TA': 'Average/Typical',
                      'Fa': 'Fair',
                  }},
    'ExterCond': {"_type": 'select', "description": "Evaluates the present condition of the material on the exterior",
                  "options": {
                      'Ex': 'Excellent',
                      'Gd': 'Good',
                      'TA': 'Average/Typical',
                      'Fa': 'Fair',
                      'Po': 'Poor'
                  }},
    'Foundation': {"_type": 'select', "description": "Type of foundation",
                   "options": {
                       'BrkTil': 'Brick & Tile',
                       'CBlock': 'Cinder Block',
                       'PConc': 'Poured Contrete',
                       'Slab': 'Slab',
                       'Stone': 'Stone',
                       'Wood': 'Wood'
                   }},
    'BsmtQual': {"_type": 'select', "description": "Evaluates the height of the basement",
                 "options": {
                     'Ex': 'Excellent (100+ inches)',
                     'Gd': 'Good (90-99 inches)',
                     'TA': 'Typical (80-89 inches)',
                     'Fa': 'Fair (70-79 inches)',
                 }},
    'BsmtCond': {"_type": 'select', "description": "Evaluates the general condition of the basement",
                 "options": {
                     'Ex': 'Excellent',
                     'Gd': 'Good',
                     'TA': 'Typical - slight dampness allowed',
                     'Fa': 'Fair - dampness or some cracking or settling'
                 }},
    'BsmtExposure': {"_type": 'select', "description": "Refers to walkout or garden level walls",
                     "options": {
                         'Gd': 'Good Exposure',
                         'Av': 'Average Exposure (split levels or foyers typically score average or above)',
                         'Mn': 'Mimimum Exposure',
                         'No': 'No Exposure'
                     }},
    'BsmtFinType1': {"_type": 'select', "description": "Rating of basement finished area",
                     "options": {
                         'GLQ': 'Good Living Quarters',
                         'ALQ': 'Average Living Quarters',
                         'BLQ': 'Below Average Living Quarters',
                         'Rec': 'Average Rec Room',
                         'LwQ': 'Low Quality',
                         'Unf': 'Unfinished'
                     }},
    'BsmtFinType2': {"_type": 'select', "description": "Rating of basement finished area (if multiple types)",
                     "options": {
                         'GLQ': 'Good Living Quarters',
                         'ALQ': 'Average Living Quarters',
                         'BLQ': 'Below Average Living Quarters',
                         'Rec': 'Average Rec Room',
                         'LwQ': 'Low Quality',
                         'Unf': 'Unfinished'
                     }},
    'BsmtUnfSF': {"_type": 'numeric', "description": "Unfinished square feet of basement area", 'step': 1},
    'TotalBsmtSF': {"_type": 'numeric', "description": "Total square feet of basement area", 'step': 1},
    'Heating': {"_type": 'select', "description": "Type of heating",
                "options": {
                    'Floor': 'Floor Furnace',
                    'GasA': 'Gas forced warm air furnace',
                    'GasW': 'Gas hot water or steam heat',
                    'Grav': 'Gravity furnace',
                    'Wall': 'Wall furnace',
                    'OthW': 'Hot water or steam heat other than gas',
                }},
    'HeatingQC': {"_type": 'select', "description": "Heating quality and condition",
                  "options": {
                      'Ex': 'Excellent',
                      'Gd': 'Good',
                      'TA': 'Average/Typical',
                      'Fa': 'Fair',
                      'Po': 'Poor'
                  }},
    'CentralAir': {"_type": 'select', "description": "Central air conditioning",
                   "options": {
                       'Y': 'Yes',
                       'N': 'No'
                   }},
    'Electrical': {"_type": 'select', "description": "Electrical system",
                   "options": {
                       'SBrkr': 'Standard Circuit Breakers & Romex',
                       'FuseA': 'Fuse Box over 60 AMP and all Romex wiring (Average)',
                       'FuseF': '60 AMP Fuse Box and mostly Romex wiring (Fair)',
                       'FuseP': '60 AMP Fuse Box and mostly knob & tube wiring (poor)',
                       'Mix': 'Mixed'
                   }},
    '1stFlrSF': {"_type": 'numeric', "description": "First Floor square feet", 'step': 1},
    '2ndFlrSF': {"_type": 'numeric', "description": "Second floor square feet", 'step': 1},
    'LowQualFinSF': {"_type": 'numeric', "description": "Low quality finished square feet (all floors)", 'step': 1},
    'GrLivArea': {"_type": 'numeric', "description": "Above grade (ground) living area square feet", 'step': 1},
    'BsmtFullBath': {"_type": 'numeric', "description": "Basement full bathrooms", 'step': 1},
    'BsmtHalfBath': {"_type": 'numeric', "description": "Basement half bathrooms", 'step': 1},
    'FullBath': {"_type": 'numeric', "description": "Full bathrooms above grade", 'step': 1},
    'HalfBath': {"_type": 'numeric', "description": "Half baths above grade", 'step': 1},
    'BedroomAbvGr': {"_type": 'numeric', "description": "Bedrooms above grade (does NOT include basement bedrooms)",
                     'step': 1},
    'KitchenAbvGr': {"_type": 'numeric', "description": "Kitchens above grade", 'step': 1},
    'KitchenQual': {"_type": 'select', "description": "Kitchen quality",
                    "options": {
                        'Ex': 'Excellent',
                        'Gd': 'Good',
                        'TA': 'Typical/Average',
                        'Fa': 'Fair'
                    }},
    'TotRmsAbvGrd': {"_type": 'numeric', "description": "Total rooms above grade (does not include bathrooms)",
                     'step': 1},
    'Functional': {"_type": 'select',
                   "description": "Home functionality (Assume typical unless deductions are warranted)",
                   "options": {
                       'Typ': 'Typical Functionality',
                       'Min1': 'Minor Deductions 1',
                       'Min2': 'Minor Deductions 2',
                       'Mod': 'Moderate Deductions',
                       'Maj1': 'Major Deductions 1',
                       'Maj2': 'Major Deductions 2',
                       'Sev': 'Severely Damaged',
                   }},
    'Fireplaces': {"_type": 'numeric', "description": "Number of fireplaces", 'step': 1},
    'FireplaceQu': {"_type": 'select', "description": "Fireplace quality",
                    "options": {
                        'Ex': 'Excellent - Exceptional Masonry Fireplace',
                        'Gd': 'Good - Masonry Fireplace in main level',
                        'TA': 'Average - Prefabricated Fireplace in main living area or Masonry Fireplace in basement',
                        'Fa': 'Fair - Prefabricated Fireplace in basement',
                        'Po': 'Poor - Ben Franklin Stove'
                    }},
    'GarageType': {"_type": 'select', "description": "Garage location",
                   "options": {
                       '2Types': 'More than one type of garage',
                       'Attchd': 'Attached to home',
                       'Basment': 'Basement Garage',
                       'BuiltIn': 'Built-In (Garage part of house - typically has room above garage)',
                       'CarPort': 'Car Port',
                       'Detchd': 'Detached from home'
                   }},
    'GarageYrBlt': {"_type": 'numeric', "description": "Year garage was built", 'step': 1},
    'GarageFinish': {"_type": 'select', "description": "Interior finish of the garage",
                     "options": {
                         'Fin': 'Finished',
                         'RFn': 'Rough Finished',
                         'Unf': 'Unfinished'
                     }},
    'GarageCars': {"_type": 'numeric', "description": "Size of garage in car capacity", 'step': 1},
    'GarageArea': {"_type": 'numeric', "description": "Size of garage in square feet", 'step': 1},
    'GarageQual': {"_type": 'select', "description": "Garage quality",
                   "options": {
                       'Ex': 'Excellent',
                       'Gd': 'Good',
                       'TA': 'Typical/Average',
                       'Fa': 'Fair',
                       'Po': 'Poor'
                   }},
    'GarageCond': {"_type": 'select', "description": "Garage condition",
                   "options": {
                       'Ex': 'Excellent',
                       'Gd': 'Good',
                       'TA': 'Typical/Average',
                       'Fa': 'Fair',
                       'Po': 'Poor'
                   }},
    'PavedDrive': {"_type": 'select', "description": "Paved driveway",
                   "options": {
                       'Y': 'Paved',
                       'P': 'Partial Pavement',
                       'N': 'Dirt/Gravel'
                   }},
    'WoodDeckSF': {"_type": 'numeric', "description": "Wood deck area in square feet", 'step': 1},
    'OpenPorchSF': {"_type": 'numeric', "description": "Open porch area in square feet", 'step': 1},
    'EnclosedPorch': {"_type": 'numeric', "description": "Enclosed porch area in square feet", 'step': 1},
    '3SsnPorch': {"_type": 'numeric', "description": "Three season porch area in square feet", 'step': 1},
    'ScreenPorch': {"_type": 'numeric', "description": "Screen porch area in square feet", 'step': 1},
    'PoolArea': {"_type": 'numeric', "description": "Pool area in square feet", 'step': 1},
    'PoolQC': {"_type": 'select', "description": "Pool quality",
               "options": {
                   'Ex': 'Excellent',
                   'Gd': 'Good',
                   'Fa': 'Fair'
               }},
    'Fence': {"_type": 'select', "description": "Fence quality",
              "options": {
                  'GdPrv': 'Good Privacy',
                  'MnPrv': 'Minimum Privacy',
                  'GdWo': 'Good Wood',
                  'MnWw': 'Minimum Wood/Wire'
              }},
    'MiscFeature': {"_type": 'select', "description": "Miscellaneous feature not covered in other categories",
                    "options": {
                        'Gar2': '2nd Garage (if not described in garage section)',
                        'Othr': 'Other',
                        'Shed': 'Shed (over 100 SF)',
                        'TenC': 'Tennis Court'
                    }},
    'MiscVal': {"_type": 'numeric', "description": "$Value of miscellaneous feature", 'step': 1},
    'MoSold': {"_type": 'numeric', "description": "Month Sold (MM)", 'step': 1},
    'YrSold': {"_type": 'numeric', "description": "Year Sold (YYYY)", 'step': 1},
    'SaleType': {"_type": 'select', "description": "Type of sale",
                 "options": {
                     'WD': 'Warranty Deed - Conventional',
                     'CWD': 'Warranty Deed - Cash',
                     'New': 'Home just constructed and sold',
                     'COD': 'Court Officer Deed/Estate',
                     'Con': 'Contract 15% Down payment regular terms',
                     'ConLw': 'Contract Low Down payment and low interest',
                     'ConLI': 'Contract Low Interest',
                     'ConLD': 'Contract Low Down',
                     'Oth': 'Other'
                 }},
    'SaleCondition': {"_type": 'select', "description": "Condition of sale",
                      "options": {
                          'Normal': 'Normal Sale',
                          'Abnorml': 'Abnormal Sale -  trade, foreclosure, short sale',
                          'AdjLand': 'Adjoining Land Purchase',
                          'Alloca': 'Allocation - two linked properties with separate deeds, typically condo with a garage unit',
                          'Family': 'Sale between family members',
                          'Partial': 'Home was not completed when last assessed (associated with New Homes)'
                      }}
}

# st.session_state.attribs = []


with st.container():
    "### Use a template, or go custom."
    c1, c2 = st.columns([1, 1])
    if c1.button("Select recommended attributes"):
        # TODO: Add recommended attributes
        st.session_state.attribs = ['BldgType', 'FullBath', 'HalfBath', 'BedroomAbvGr']
        st.experimental_rerun()

    if c2.button("Select all attributes"):
        st.session_state.attribs = list(ALL_ATTRIBS.keys())
        st.experimental_rerun()


def on_multiselect_change():
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

# "Allowed Attributes:"
# ALL_ATTRIBS


"## Predicted Price"
# input = np.array([st.session_state[f"data--attrib-{attrib}"] for attrib in st.session_state.attribs])
# input.reshape(-1, 1)
# input
predicted_price = predict_with_model(model)

rmse = mean_squared_error(test[['SalePrice']], model.predict(test.loc[:, test.columns != 'SalePrice']), squared=False)
# readable_price = "{0:.3g}".format(predicted_price)
st.write(
    f"\${(round(predicted_price) // 1000 * 1000):,d} ± {(round(rmse) // 1000 * 1000):,d}")  # Round to the nearest thousand, format with commas

f"RMSE: {format_number(rmse)}"
# R squared
"R squared:"
st.write(sklearn.metrics.r2_score(test[['SalePrice']], model.predict(test.loc[:, test.columns != 'SalePrice'])))

# Scatter plot of this house vs the training dataset


"Influence of each attribute on price"
st.selectbox("Select an attribute to plot against SalePrice", st.session_state.attribs, key="scatterplot_attrib")
scatterplot_attrib = st.session_state.scatterplot_attrib

if ALL_ATTRIBS[scatterplot_attrib]["_type"] == 'numeric':
    st.plotly_chart(px.scatter(test, x=scatterplot_attrib, y='SalePrice', trendline='ols'), use_container_width=True)
elif ALL_ATTRIBS[scatterplot_attrib]["_type"] == 'select':
    colnames = [x for x in test.columns if scatterplot_attrib in x]
    # Plot test with colnames
    st.plotly_chart(px.box(train, x=colnames, y='SalePrice'), use_container_width=True)

# Plot a histogram of the predicted prices
"## Predicted Price Distribution"

st.write("Histogram of current market prices with the predicted price of this house in red")
predicted_price_histogram = px.histogram(test, x='SalePrice')
# Set the column that "predicted_price" is in to red
predicted_price_histogram.add_vline(x=predicted_price, line_width=3, line_dash="dash", line_color="red")
st.plotly_chart(predicted_price_histogram, use_container_width=True)



'## Model Coefficient Correlations'
# Df of coefficients, correlation
df_train_corr = train.corr()
st.plotly_chart(px.imshow(df_train_corr), use_container_width=True)

"""
## Debugging
"""

"Session State"
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
