import pandas as pd
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns
print("Setup Complete")

# Path of the file to read
fifa_filepath = "../input/fifa.csv"

# Fill in the line below to read the file into a variable fifa_data
fifa_data = pd.read_csv(fifa_filepath,index_col="Date",parse_dates=True)

fifa_data.head()
# Set the width and height of the figure
plt.figure(figsize=(16,6))

# Fill in the line below: Line chart showing how FIFA rankings evolved over time
sns.lineplot(data=fifa_data)
# Check your answer
