import pandas as pd

from sklearn.metrics import  mean_absolute_error
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import  train_test_split

def get_mae(max_leaf_nodes, train_X, val_X, train_y, val_y):
    model = DecisionTreeRegressor(max_leaf_nodes=max_leaf_nodes,random_state=0)
    model.fit(train_X,train_y)
    preds_val = model.predict(val_X)
    mae = mean_absolute_error(val_y,preds_val)
    return(mae)


#The data is loaded into train_X, val_X, train_y and val_y using the code you've already seen (and which you've already written).
# Load data
melbourne_file_path = '~/test/melb_data.csv'
melbourne_data = pd.read_csv(melbourne_file_path)
# Filter rows with missing values
filtered_melbourne_data = melbourne_data.dropna(axis=0)

# Choose target and features
y = filtered_melbourne_data.Price
melbourne_features = ['Rooms', 'Bathroom', 'Landsize', 'BuildingArea','YearBuilt', 'Lattitude', 'Longtitude']

X = filtered_melbourne_data[melbourne_features]
train_X,val_X,train_y,val_y = train_test_split(X,y,random_state=0)

#We can use a for-loop to compare the accuracy of models built with different values for max_leaf_nodes.
# compare MAE with differing values of max_leaf_nodes
for max_leaf_nodes in [5,50,500,5000]:
    my_mae = get_mae(max_leaf_nodes, train_X, val_X, train_y, val_y)
    print("Max leaf nodes: %d \t\t Mean Absolute Error: %d"  %(max_leaf_nodes,my_mae) )

# Max leaf nodes: 5 		 Mean Absolute Error: 347380
# Max leaf nodes: 50 		 Mean Absolute Error: 258171
# Max leaf nodes: 500 		 Mean Absolute Error: 243495
# Max leaf nodes: 5000 		 Mean Absolute Error: 254983

#Of the options listed, 500 is the optimal number of leaves.

# Conclusion
# Here's the takeaway: Models can suffer from either:
# 
# Overfitting: capturing spurious patterns that won't recur in the future, leading to less accurate predictions, or
# Underfitting: failing to capture relevant patterns, again leading to less accurate predictions.
# We use validation data, which isn't used in model training, to measure a candidate model's accuracy. This lets us try many candidate models and keep the best one.
# 

