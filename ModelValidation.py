import pandas as pd
from  sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split



melbourne_file_path = '~/test/melb_data.csv'
melbourne_data = pd.read_csv(melbourne_file_path)
#print(melbourne_data.columns)
melbourne_data = melbourne_data.dropna(axis=0)
#print(melbourne_data)
y= melbourne_data.Price
#print(y)

melbourne_features = ['Rooms', 'Bathroom', 'Landsize', 'Lattitude', 'Longtitude']
X = melbourne_data[melbourne_features]
print(X.describe())
print(X.head())

melbourne_model = DecisionTreeRegressor(random_state=1)

print(melbourne_model.fit(X,y))

print("Making predictions for the following 5 houses:")
print(X.head())
print("The predictions are")
print(melbourne_model.predict(X.head()))

print("实际价格：")
print(y.head())

predicted_home_prices = melbourne_model.predict(X)
print(mean_absolute_error(y,predicted_home_prices))


train_X,val_X,train_y,val_y = train_test_split(X,y,random_state=0)
melbourne_model = DecisionTreeRegressor()
melbourne_model.fit(train_X,train_y)

val_preditions = melbourne_model.predict(val_X)
print(mean_absolute_error(val_y,val_preditions))

##https://www.kaggle.com/dansbecker/model-validation
