import os
import pandas as pd
from sklearn.tree import DecisionTreeRegressor

# get script directory
run_path = os.path.dirname(os.path.abspath(__file__))
train_path = os.path.join(run_path, "..", "train.csv")

# read train_data
home_data = pd.read_csv(train_path)

# set prediction target 'y'
y = home_data.SalePrice
# set features 'X'
features = ["LotArea", "YearBuilt", "1stFlrSF", "2ndFlrSF",
            "FullBath", "BedroomAbvGr", "TotRmsAbvGrd"]
X = home_data[features]

iowa_model = DecisionTreeRegressor(random_state=0)
iowa_model.fit(X, y)

# make a prediction
prediction = iowa_model.predict(X)
print(prediction)

