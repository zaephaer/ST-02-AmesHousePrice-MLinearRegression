import streamlit as st
import pandas as pd

st.set_page_config(layout="wide")
st.set_option('deprecation.showPyplotGlobalUse', False)

# Apllication Title
st.write("""
# House Price Prediction of Ames Dataset using PAS
This application will try to do prediction of housing price based on its selected feature variables
""")
st.write('---')

# Loads Dataset
data = pd.read_csv("./data/ames_dataset.csv")
st.header("Dataset Overview")  # add a title
st.write(data.head())  # visualize my dataframe in the Streamlit app
#-------------------------------------------------------------------------------------------------------
st.header("Dataset Description")
st.write(data.shape)
st.write(data.describe())

data.rename(columns = {"1stFlrSF": "FirstFlrSF"}, inplace = True)
#-------------------------------------------------------------------------------------------------------
st.header("Focus at Target (Dependent) variable")
st.write(data['SalePrice'].describe().to_frame())
#-------------------------------------------------------------------------------------------------------
st.header("Using correlation to find the best 5 Feature (Independent) variables")
corr = data.corr()
st.write(corr["SalePrice"].sort_values(ascending=False).head(7))
st.write('---')

# Creating input sliders for feature Variables
st.sidebar.header('Specify Input (Feature Variables): ')
def user_input_features():
    OverallQual = st.sidebar.slider('Overall House Quality', int(data['OverallQual'].min()), int(data['OverallQual'].max()))
    GrLivArea   = st.sidebar.slider('Ground Living Area', int(data['GrLivArea'].min()), int(data['GrLivArea'].max()))
    GarageCars  = st.sidebar.slider('Garage cars', int(data['GarageCars'].min()), int(data['GarageCars'].max()))
    GarageArea  = st.sidebar.slider('Garage Area', int(data['GarageArea'].min()), int(data['GarageArea'].max()))
    TotalBsmtSF = st.sidebar.slider('Total Basement sqft', int(data['TotalBsmtSF'].min()), int(data['TotalBsmtSF'].max()))
    FirstFlrSF    = st.sidebar.slider('1st Floor sqft', int(data['FirstFlrSF'].min()), int(data['FirstFlrSF'].max()))
    datax = {'OverallQual': OverallQual,'GrLivArea': GrLivArea,'GarageCars': GarageCars,'GarageArea': GarageArea,
            'TotalBsmtSF': TotalBsmtSF, 'FirstFlrSF':FirstFlrSF}
    features = pd.DataFrame(datax, index=[0])
    return features
dy = user_input_features()

# Print selected input parameters
st.header('Selected Input parameters')
st.write(dy)

# split data and modelling -----------------------------------------------------------------------------------------
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split

X = data[["OverallQual","GrLivArea","GarageCars","GarageArea","TotalBsmtSF","FirstFlrSF"]]
y = data['SalePrice']

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3)
model = DecisionTreeRegressor()
model.fit(X_train, y_train)

y_predict = model.predict(X_test)

#param = [[55,500,25,200,200]]
st.header('Prediction of House Price')
st.write("Predicted house price with given parameter will be", int(model.predict(dy)))

# calculate model good from test dataset compare actual saleprice and predicted salesprice
from sklearn.metrics import r2_score
st.write("Model r2 score is", round(r2_score(y_test, y_predict),4))
