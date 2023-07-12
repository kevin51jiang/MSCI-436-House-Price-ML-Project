# JSON-like dictionary showing all allowed attributes, and their descriptions

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
                         # 'Names': 'North Ames',
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
                     # '2FmCon': 'Two-family Conversion; originally built as one-family dwelling',
                     # 'Duplx': 'Duplex',
                     'TwnhsE': 'Townhouse End Unit',
                     # 'TwnhsI': 'Townhouse Inside Unit'
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
                     # 'Ex': 'Excellent',
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
