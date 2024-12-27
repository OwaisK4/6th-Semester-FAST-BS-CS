# Code1:
import pandas as pd
df = pd.read_csv("data.csv")
df.head()
print(df.shape)
print(df.columns)
print(df.info())
df.describe()
df["Pulse"].mean()

# Code2:
import pandas as pd
mydataset = {
'cars': ["BMW", "Volvo", "Ford"],
'passings': [3, 7, 2]
}
myvar = pd.DataFrame(mydataset)
print(myvar)

# Code3:
import pandas as pd
data = {
"calories": [420, 380, 390],
"duration": [50, 40, 45]
}
#load data into a DataFrame object:
df = pd.DataFrame(data)
print(df)
#refer to the row index:
print(df.loc[0])
#use a list of indexes:
print(df.loc[[0, 1]])

# Code4:
#read CSV:
df = pd.read_csv('data.csv')
print(df)
#Analyzing dataframe:
#The head() method returns the headers and a specified number of rows, starting from the top.
df = pd.read_csv('data.csv')
#printing the first 10 rows of the DataFrame:
print(df.head(10))
#There is also a tail() method for viewing the last rows of the DataFrame.
#The tail() method returns the headers and a specified number of rows, starting from the bottom.
#Print the last 5 rows of the DataFrame:
print(df.tail())
#The DataFrames object has a method called info(), that gives you more information about the data set.
#Print information about the data:
print(df.info())
#Cleaning Empty cell:
new_df = df.dropna()
#If you want to change the original DataFrame, use the inplace = True argument:
#Remove all rows with NULL values:
df.dropna(inplace = True)
#The fillna() method allows us to replace empty cells with a value:
#Replace NULL values with the number 130:
df.fillna(130, inplace = True)
#Replace NULL values in the "Calories" columns with the number 130:
df["Calories"].fillna(130, inplace = True)

#
import pandas as pd

values = {
    'Duration': [60, 60, 60, 45, 45],
    'Pulse': [110, 117, 103, 109, 117],
    'Max Pulse': [130, 145, 135, 175, 148],
    'Calories': [409.1, 479, 340, 282.4, 406]
}

values_table = pd.DataFrame(values)

values_table.to_csv("data.csv", index=False)

print(values_table)