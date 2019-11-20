import os
import pandas as pd
from sklearn.tree import DecisionTreeRegressor

# get script directory
run_path = os.path.dirname(os.path.abspath(__file__))
melb_path = os.path.join(run_path, "..", "melb_data.csv")

# read melb_data
melb_data = pd.read_csv(melb_path)
# drop missing values (Drop N/A)
melb_data = melb_data.dropna(axis=0)

###########################################################
# We use dot notation to select the prediction target 'y' #
###########################################################
y = melb_data.Price

#############################################################
# We use column list to select the columns that are inputed #
# into our model to predit the target 'y', features 'x'     # 
#############################################################
melb_features = ['Rooms', 'Bathroom', 'Landsize', 'Lattitude', 'Longtitude']
X = melb_data[melb_features]

# Define model
# Specify a number for random_state to ensure same results each run
melb_model = DecisionTreeRegressor(random_state=1)
# fit model
melb_model.fit(X, y)

melb_model.predict(X)
