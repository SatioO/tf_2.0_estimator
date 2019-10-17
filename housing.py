import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# read data from csv
train = pd.read_csv("./train.csv")
test = pd.read_csv("./test.csv")

# Let's take a reference of the train and test ids's for future reference
train_ids = train.index
test_ids = test.index

# Let's plot the dependent variable to see it's distribution
sns.distplot(train["SalePrice"])
plt.show()

# As we can clearly see, it is positively skewed.
# Let's fix it by applying some non-linear transformation
# In our case, it is log transformation
train_y = train.pop("SalePrice")
train_y = np.log(train_y)
# We can clearly see applying log transformation worked really well.
# Our dependent variable now look normally distributed
sns.distplot(train_y)
plt.show()

# let's do some missing values analysis
missing_df = pd.DataFrame(data=None)
missing_df["Percentage"] = (train.isna().sum() / train.isna().count()) * 100
missing_df.sort_values(by="Percentage", ascending=False, inplace=True)
missing_df = missing_df[missing_df["Percentage"] > 0]
print(missing_df)
print()