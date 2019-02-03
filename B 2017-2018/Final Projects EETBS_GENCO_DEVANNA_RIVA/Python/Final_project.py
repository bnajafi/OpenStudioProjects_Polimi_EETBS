# -*- coding: utf-8 -*-

# FINAL_PROJECT

# De Vanna Guglielmo, Genco Federico, Riva Federico 

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

DataFolderPath = "/Users/Fede/Desktop/Final_project/DataSet"
FileName= "NEW-DATA-1.T15.txt"
FilePath = DataFolderPath+"/"+FileName

DataFrame = pd.read_csv(FilePath, sep= " ", index_col=11)
DataFrame["period"] = DataFrame["1:Date"].map(str) + " " + DataFrame["2:Time"].map(str)
DataFrame.index = DataFrame["period"]

previousIndex= DataFrame.index 
ParsedIndex= pd.to_datetime(previousIndex)
DataFrame.index= ParsedIndex

DF_myChosenDates = DataFrame["2012-03-25 12:00:00 ":"2012-03-31 12:00:00"]
DF_cleaned = DF_myChosenDates.dropna()
DF_cleaned.corr()

fig = plt.figure()

ax1= fig.add_subplot(3,1,1) 
ax2= fig.add_subplot(3,1,2)
ax3= fig.add_subplot(3,1,3)

DF_cleaned["4:Temperature_Habitacion_Sensor"].plot(ax=ax1,color = "r", legend = True)
DF_cleaned["11:Lighting_Habitacion_Sensor"].plot(ax=ax2,color = "y", legend = True)
DF_cleaned["18:Meteo_Exterior_Piranometro"].plot(ax=ax3,color = "g", legend = True)

###

DF_FinalDataSet = DF_cleaned.copy() 

def lag_column(df,column_name,lag_period=1):
    for i in range(1,lag_period+1,1):
        new_column_name = column_name+"-"+str(i)+"hr"
        df[new_column_name] = df[column_name].shift(i)

    return df

DF_FinalDataSet= lag_column(DF_FinalDataSet,"4:Temperature_Habitacion_Sensor",24)
DF_FinalDataSet.dropna(inplace=True)
DF_FinalDataSet.index                
DF_FinalDataSet.index.hour

DF_FinalDataSet["hour"] = DF_FinalDataSet.index.hour
DF_FinalDataSet["month"] = DF_FinalDataSet.index.month
DF_FinalDataSet["week_of_year"] = DF_FinalDataSet.index.week   

def weekendDetector(day):
    weekendLabel=0
    if (day== 5or day==6):
        weekendLabel=1
    else:
        weekendLabel=0
    return weekendLabel   
    
def dayDetector(hour):
    dayLabel=1
    if(hour<20 and hour>9):
        dayLabel=1 
    else:
        dayLabel=0
    return dayLabel      

DF_FinalDataSet["weekend"]= [weekendDetector(thisDay) for thisDay in DF_FinalDataSet.index.dayofweek]   
DF_FinalDataSet["day_night"]= [dayDetector(thisHour) for thisHour in DF_FinalDataSet.index.hour]     
DF_FinalDataSet.drop(DF_FinalDataSet.columns[[0,1,6,8,9,10,11,12,13,14,15,17,18,19,23]],axis=1,inplace=True);
DF_FinalDataSet.dropna(inplace=True)    

DF_FinalDataSet.corr()

DF_target = DF_FinalDataSet["4:Temperature_Habitacion_Sensor"]
DF_features = DF_FinalDataSet.drop("4:Temperature_Habitacion_Sensor", axis = 1) #axis 1 means it's a column

#so now I have inputs and outputs, let's build the model

from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(DF_features,DF_target,test_size=0.2,random_state=41234)

#now we test the model

from sklearn import linear_model
linear_reg = linear_model.LinearRegression()

#linear module with training data (like fitting a line)

linear_reg.fit(X_train,Y_train)
predict_linearReg_split = linear_reg.predict(X_test) 

Y_test.index

predict_DF_linearReg_split = pd.DataFrame(predict_linearReg_split,index = Y_test.index,columns = ["Temperature prediction"])
predict_DF_linearReg_split = predict_DF_linearReg_split.join(Y_test)
predict_DF_linearReg_split_ChosenDates = predict_DF_linearReg_split["2012-03-25":"2012-03-31"]
predict_DF_linearReg_split_ChosenDates.plot()

plt.xlabel("time")
plt.ylabel("temperature")
plt.title("Linear Regression Prediction")

#metrics

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

mean_absolute_error_linearReg_split = mean_absolute_error(Y_test,predict_linearReg_split)
mean_squared_error_linearReg_split = mean_squared_error(Y_test,predict_linearReg_split)
R2_score_linearReg_split = r2_score(Y_test,predict_linearReg_split)

#cross validation (implemented in sklearn)

from sklearn.model_selection import cross_val_predict

predict_linearReg_CV = cross_val_predict(linear_reg,DF_features,DF_target,cv=10)
predict_DF_linearReg_CV = pd.DataFrame(predict_linearReg_CV,index = DF_target.index,columns = ["TemperaturePredict_linearReg_split"])
predict_DF_linearReg_CV = predict_DF_linearReg_CV.join(DF_target)
predict_DF_linearReg_CV_ChosenDates = predict_DF_linearReg_CV["2012-03-25 12:00:00 ":"2012-04-10 12:00:00"]
predict_DF_linearReg_CV_ChosenDates.plot()

plt.xlabel("time")
plt.ylabel("temperature")
plt.title("Cross Validation Prediction")

#metrics

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

mean_absolute_error_linearReg_CV = mean_absolute_error(DF_target,predict_linearReg_CV)
mean_squared_error_linearReg_CV = mean_squared_error(DF_target,predict_linearReg_CV)
R2_score_linearReg_CV = r2_score(DF_target,predict_linearReg_CV)

#cross validation, randomForest

from sklearn.ensemble import RandomForestRegressor

reg_RF = RandomForestRegressor()
predict_RF_CV = cross_val_predict(reg_RF,DF_features,DF_target,cv=10)
predict_DF_RF_CV = pd.DataFrame(predict_RF_CV,index = DF_target.index,columns = ["TemperaturePredict_linearReg_split"])
predict_DF_RF_CV = predict_DF_RF_CV.join(DF_target)
predict_DF_RF_CV_ChosenDates = predict_DF_RF_CV["2012-03-25":"2012-03-31"]
predict_DF_RF_CV_ChosenDates.plot()

plt.xlabel("time")
plt.ylabel("temperature")
plt.title("Random Forest Regressor Prediction")

#metrics

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

mean_absolute_error_linearReg_RF = mean_absolute_error(DF_target,predict_RF_CV)
mean_squared_error_linearReg_RF = mean_squared_error(DF_target,predict_RF_CV)
R2_score_linearReg_RF = r2_score(DF_target,predict_RF_CV)
