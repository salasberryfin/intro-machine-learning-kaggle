import os
import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split

# get script directory
run_path = os.path.dirname(os.path.abspath(__file__))
melb_path = os.path.join(run_path, "..", "melb_data.csv")

# read melb_data
melb_data = pd.read_csv(melb_path)
# drop missing values (Drop N/A)
melb_data = melb_data.dropna(axis=0)

y = melb_data.Price
melb_features = ['Rooms', 'Bathroom', 'Landsize', 'BuildingArea', 'YearBuilt', 'Lattitude', 'Longtitude']
X = melb_data[melb_features]

# We use different data for training and validation
# so that we can properly evaluate the accuracy of the 
# prediction
train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=0)

# Create the new model based on the split
melb_model = DecisionTreeRegressor(random_state=0)
melb_model.fit(train_X, train_y)

# Predict based on the other part of the split
home_price_prediction = melb_model.predict(val_X)
print(home_price_prediction)

# Obtaine MAE
mae = mean_absolute_error(val_y, home_price_prediction)
print(mae)
