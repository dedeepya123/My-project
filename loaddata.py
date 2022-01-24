import pandas as pd
from matplotlib.ticker import ScalarFormatter, FormatStrFormatter 
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import numpy as np
#data loading
csv_file = 'population.csv'
df =pd.read_csv(csv_file, skiprows=4)
#printing all column names
print(df.columns)
#printing entire data from csv file
print(pd.DataFrame(df))
#datacleaning by removing unnecessary columns
columns_to_drop = ['Country Code', 'Indicator Name', 'Indicator Code', 'Unnamed: 64']
df.drop(columns=columns_to_drop, inplace=True)
#REnaming columns
df.rename(columns={'Country Name': 'Country'}, inplace=True)
#replacing nan values with zeros
df.fillna(0,inplace=True)
#Visulaising Australias population
print('Enter a Country to view Population')
cntryname=input()
record = df[df['Country'] == cntryname] # get the tuple with the Australia's population data
years = record.columns.tolist()[1:] # get the years
population = record.values.tolist()[0][1:] # get the population with respect to the year
plt.scatter(years, population)  # plot scatter plot
plt.plot(years, population) # line to connect the points)
plt.tick_params(axis='x',color='red')#for changing ticks properties, rotate x axis labels text to vertical inorder for it to show up and not cluster together
plt.xticks(rotation='vertical',color='blue')#for changing lables properties
plt.title(cntryname+' population from 1960 to 2017') # set graph title
plt.xlabel('Year') # set y axis label
plt.ylabel('Total Population') # set y axis label
plt.gca().yaxis.set_major_formatter(FormatStrFormatter('%.0f')) # turn of scientific notations i.e.,to format or change labels
plt.show() # display graph
countries = df['Country'].tolist()
temp_df = pd.DataFrame()
for country in countries:
    # prepare data for the model
    #get one by one coutry drops country name and stores year and population
    record = df[df['Country'] == country].drop(['Country'], axis=1)

    #transpose horizontal years into vertical 
    record = record.T
    record.reset_index(inplace=True)
    record.columns = ['Year', 'Population']
    X = record['Year']
    Y = record['Population']


    # train the model
    regressor = LinearRegression()#splitting the data into training and testing data set
    regressor.fit(np.array(X).reshape(-1,1), Y)#-1 indicates any no. of rows 1 indicates one column


    # predict future population with respective year and add back to current record
    for year in range(2018,2031):
        future_population = round(regressor.predict(np.array([year]).reshape(-1,1))[0])#0 indicates 2019,2020,2021,......
        row = pd.DataFrame([[year,future_population]], columns=['Year','Population'])
        record = record.append(row, ignore_index=True)

    # change narrow dataframe back to a wide one
    record = record.T#tranpose
    new_header = record.iloc[0]#gets all years 2019,2010,.....
    record = record[1:]#gets population of all years.....
    record.columns = new_header#setting years as column names
    record.columns.name = None
    record.index = [country]#indexing country names
    temp_df = pd.concat([temp_df, record])#finnally concatinating every country 2019,2020, population to temporary dataframe
df = temp_df
df.to_csv('future_world_population.csv')
#plotting future population of australia
record = df[df.index == cntryname]
record.columns = record.columns.astype(str)
years = record.columns.tolist()
population = record.values.tolist()[0]
plt.scatter(years, population)
plt.plot(years, population)
plt.xticks(rotation='vertical') 
plt.title(cntryname+' Predicted population')
plt.xlabel('Year')
plt.ylabel('Total Population')
plt.gca().yaxis.set_major_formatter(FormatStrFormatter('%.0f'))
plt.show()




