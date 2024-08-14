import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import pickle

df = pd.read_csv("D:\Chrome Downloads\VE3 Takehome\dataset\House_Rent_Dataset.csv")


# Removing unnecessary columns from the dataset
df.pop("Posted On")
df.pop("Floor")
df.pop("Area Locality")
df.pop("Point of Contact")
df.pop("Area Type")
df.pop("Tenant Preferred")


# Encoding categorical features
df['City'] = df['City'].replace(["Mumbai","Bangalore","Hyderabad","Delhi","Chennai","Kolkata"],[5,4,3,2,1,0])
df['Furnishing Status'] = df['Furnishing Status'].replace(["Furnished","Semi-Furnished","Unfurnished"],[2,1,0])

target = df.pop("Rent")


# Spliting the data into training and testing sets (80% train, 20% test)
from sklearn.model_selection import train_test_split
x_train, x_test, y_train , y_test = train_test_split(df, target, test_size=0.2)


# Linear Regression Model Training
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(x_train,y_train)


pickle.dump(lin_reg, open('model.pkl','wb')) # Saved the model to a file using pickle
