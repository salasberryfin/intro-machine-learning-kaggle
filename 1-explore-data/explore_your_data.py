import os
import pandas as pd
from datetime import datetime

# get script directory
run_path = os.path.dirname(os.path.abspath(__file__))
train_path = os.path.join(run_path, "..", "train.csv")

# read train_data
train_data = pd.read_csv(train_path)
# get summary statistics
train_stats = train_data.describe()

# average lot size
lot_size = round(train_stats.get('LotArea')).get('mean')
# how old is the newest home
current_year = datetime.now().year
newest_home_age = current_year - int(train_stats.get("YearBuilt").get("max"))
