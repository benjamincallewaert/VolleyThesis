import csv
import os
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.model_selection import train_test_split

testData_df =pd.DataFrame(columns=['PosService','MT'])

def processData(data) :
    i=0
    matchEvents_df = pd.DataFrame(
        columns=['Team', 'Player', 'Skill', 'Type', 'Quality', 'SetHomeTeam', 'SetAwayTeam','ScoreHomeTeam','ScoreAwayTeam'])
    SetHomeTeam=0
    SetAwayTeam=0
    HomeScore=0
    AwayScore=0
    for event in data:
        result = processLine(event)
        if result!=0:
            if result=="Home":
                HomeScore+=1
            else :
                if result=="Away":
                    AwayScore+=1
                else :
                    if result=="set":
                        if HomeScore > AwayScore:
                            SetHomeTeam += 1
                        else :
                            SetAwayTeam += 1
                        HomeScore=0
                        AwayScore=0
                    else:
                        result.extend([SetHomeTeam,SetAwayTeam,HomeScore,AwayScore])
                        matchEvents_df.loc[i]=result
                        i+=1
    return matchEvents_df

def processLine(str):
    event=str
    result=list()
    if event[0] == "*" :
        result.append("Home")
    else :
        if event[0] == "a":
            result.append("Away")
        else :
            return 0
    if event[1:2].isdigit():
        result.append(event[1:3])
        result.append(event[3])
        result.append(event[4])
        result.append(event[5])
        return result
    else :
        if event[1] == "p":
            return result[0]
        else :
            if event[0:2] == "**" :
                return "set"
            else:
                return 0

def updateTestDate(df,belgium):
    for row in df.iterrows() :
        if row[1]['Skill']=="S" and row[1]['Team']==belgium:
            set=row[1]['SetHomeTeam']+row[1]['SetAwayTeam']
            if (row[1]['ScoreHomeTeam'] >= 20 and row[1]['ScoreAwayTeam'] >= 20) or set==5 :
                if row[1]['Quality'] in ("#","/"):
                    testData_df.loc[len(testData_df)] =[1, 1]
                if row[1]['Quality'] in ("-","="):
                    testData_df.loc[len(testData_df)] = [0, 1]
            else:
                if row[1]['Quality'] in ("#","/"):
                    testData_df.loc[len(testData_df)]=[1,0]
                if row[1]['Quality'] in ("-","="):
                    testData_df.loc[len(testData_df)]=[0,0]

def handleBelgiumFile(filename):
    pathname = str("data\\") + filename
    f=open(pathname, "r" , encoding="utf8")
    filedata=f.read().split("[3SCOUT]")[1]
    file=open("matchdata.txt","w")
    file.writelines(filedata)
    file.close()
    match_rawdata = pd.read_csv("matchdata.txt",sep = ';', header = None)
    match_data= match_rawdata[0].tolist()
    df=processData(match_data)
    if filename[18:21]=="BEL" or filename[19:22]=="BEL":
        updateTestDate(df,"Home")
    else:
        updateTestDate(df,"Away")

def logisticRegression(testData):
    X_train, X_test, y_train, y_test = train_test_split(testData['MT'].values.reshape(-1,1).astype('int'), testData['PosService'].values.reshape(-1,1).astype('int').ravel(), test_size=0.25, random_state=0)
    logreg = LogisticRegression(random_state=0, class_weight='balanced',solver='lbfgs')
    logreg.fit(X_train,y_train)
    y_pred=logreg.predict(X_test)
    print("\n---SKLEARN LOG REGRESSION---")
    print("Intercept: " + str(logreg.intercept_[0]))
    print("Coefficiënt Xmt: "+ str(logreg.coef_[0][0]))
    test=metrics.accuracy_score(y_test, y_pred)
    print("Accuracy score: " + str(test))
    cnf_matrix = metrics.confusion_matrix(y_test, y_pred)
    print("Confusion Matrix: \n"+ str(cnf_matrix))
    print("\n---STATSMODEL LOG REGRESSION---")
    X_train=sm.add_constant(X_train)
    model = sm.Logit(endog=y_train, exog=X_train)
    result = model.fit()
    print(result.summary2())



def calculateMean():
    stats_df=pd.DataFrame(columns=['NPosService','NService'])
    stats_df.loc['MT']=[0,0]
    stats_df.loc['NoMT'] = [0, 0]
    for line in testData_df.iterrows():
        if line[1]['MT']==0:
            stats_df.loc['NoMT']['NPosService']=stats_df.loc['NoMT']['NPosService']+line[1]['PosService']
            stats_df.loc['NoMT']['NService'] = stats_df.loc['NoMT']['NService']+1
        if line[1]['MT']==1:
            stats_df.loc['MT']['NPosService'] = stats_df.loc['MT']['NPosService'] + line[1]['PosService']
            stats_df.loc['MT']['NService'] = stats_df.loc['MT']['NService'] + 1
    meanMT=stats_df.loc['MT']['NPosService']/stats_df.loc['MT']['NService']
    meanNoMT=stats_df.loc['NoMT']['NPosService']/stats_df.loc['NoMT']['NService']
    stats_df.insert(2,"Mean",[meanMT,meanNoMT])
    return stats_df

if __name__ == '__main__':
    for filename in os.listdir(r"C:\Users\calle\OneDrive\Documenten\DataThesis"):
        handleBelgiumFile(filename)
        print(filename + " done")
    logisticRegression(testData_df)
    print(calculateMean())



