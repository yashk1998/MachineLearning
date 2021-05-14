#Import libraries

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import seaborn as sns
import re

#Dataset
df = pd.read_csv("C:/Users/Dell/Desktop/Yash/Data Science/diabetic_data.csv")
# df.drop_duplicates(subset = ['patient_nbr'],inplace = True)

for cols in list(df.columns):
    list_unique_val = df[cols].unique()
    
    # if number of unique values is less than 25, print the values. Otherwise print the number of unique values
    if len(list_unique_val)<25:
        print(cols)
        print(list_unique_val)
    else:
        print(cols + ': ' +str(len(list_unique_val)) + ' unique values')
#This helps us figure out if column will provide any information to the model
df.drop(columns = ['examide','citoglipton'], inplace = True)

#Cleaning Data
for col in df.columns:
    if df[col].dtype == object and df[col][df[col] == '?'].count()>0:
         print(col,':',df[col][df[col] == '?'].count(),':',round((df[col][df[col] == '?'].count()/len(df))*100),'%')
         
#Owing to less interpretibility 
df.drop(columns = ['payer_code','weight','medical_specialty'], inplace = True)

#Dropping all the non-indicative diagonosis
#Adding filters to it 

empty_ID = set(df[(df['diag_1']=="?") & (df['diag_2']=='?') & (df['diag_3']=='?')].index)
empty_ID = empty_ID.union(set(df['gender'][df['gender'] == 'Unknown/Invalid'].index))

new_ID = list(set(df.index) - set(empty_ID))
df = df.iloc[new_ID]
df = df[df['race'] != '?']

#Replace readmitted with 1s and 0s
df['readmitted'] = df['readmitted'].apply(lambda x: 0 if x =='NO' else 1)

#Sorting according to age
sortage = df.sort_values(by = 'age')
x = sns.stripplot(x = "age", y = "num_medications", data = sortage, color = 'red')
sns.despine() #remove top and right axes
x.figure.set_size_inches(10, 6)
x.set_xlabel('Age')
x.set_ylabel('Number of Medications')
x.axes.set_title('Number of Medications vs. Age')
plt.show()

#Gender and Readmissions,
plot1 = sns.countplot(x = 'gender', hue = 'readmitted', data = df) 
sns.despine()
plot1.figure.set_size_inches(8, 6)
plot1.legend(title = 'Readmitted patients', labels = ('No', 'Yes'))
plot1.axes.set_title('Readmissions Balance by Gender')
plt.show()

#Relation between age and readmission,
b = df.age.unique()
b.sort()
b_sort = np.array(b).tolist()
ageplt = sns.countplot(x = 'age', hue = 'readmitted', data = df, order = b_sort) 
sns.despine()
ageplt.figure.set_size_inches(7, 6)
ageplt.legend(title = 'Readmitted within 30 days', labels = ('No', 'Yes'))
ageplt.axes.set_title('Readmissions Balance by Age')
plt.show()

#Discharge disposition ID, some nuumbers indicated: Dead
df = df.loc[~df['discharge_disposition_id'].isin([11,13,14,19,20,21])]

#To properly give numbers to age pruos since 0-10 will not provide any solid information 
age_dict = {"[0-10)":5, "[10-20)":15, "[20-30)":25, "[30-40)":35, "[40-50)":45, "[50-60)":55, "[60-70)":65, "[70-80)":75, "[80-90)":85, "[90-100)":95}
df['age'] = df['age'].map(age_dict).astype('int64')

# merge ids with same meaning
def merge(df, col, same_ids):
    for ids in same_ids:
        for k in ids[1:]:
            df[col] = df[col].replace(k, ids[0])
    return df

df = merge(df, 'admission_type_id', [
    [1, 2, 7],  # emergence
    [5, 6, 8],  # not avaliable
])
df = merge(df, 'discharge_disposition_id', [
    [18, 25, 26],  # not avaliable
    [1, 6, 8],  # to home
    [2, 3, 4, 5],  # discharge to another hospital
    [10, 12, 15, 16, 17],  # discharge to outpatient
])
df = merge(df, 'admission_source_id', [
    [1, 2, 3], # Referral
    [4, 5, 6, 10, 22, 25], # from another hospital
    [9, 15, 17, 20, 21]  # not avaliable
])

#Comorbidity
diagnosis = df[['diag_1','diag_2','diag_3']]
diagnosis = diagnosis.replace('?',0)

#Calculating comorbidity
def calculate_Comorbidity(row):
    diabetes_code = "^[2][5][0]"
    circulatory_code = "^[3][9][0-9]|^[4][0-5][0-9]"
    value =0
    
    if  (   not(bool(re.match(diabetes_code,str(np.array(row['diag_1']))))) and
            not(bool(re.match(diabetes_code,str(np.array(row['diag_2'])))))and 
            not(bool(re.match(diabetes_code,str(np.array(row['diag_3'])))))
        ) and (not(bool(re.match(circulatory_code,str(np.array(row['diag_1']))))) and 
              not(bool(re.match(circulatory_code,str(np.array(row['diag_2'])))))and 
              not(bool(re.match(circulatory_code,str(np.array(row['diag_3'])))))
        ):
        value= 0
    if (  bool(re.match(diabetes_code,str(np.array(row['diag_1'])))) or 
          bool(re.match(diabetes_code,str(np.array(row['diag_2'])))) or 
          bool(re.match(diabetes_code,str(np.array(row['diag_3']))))
         )and (not(bool(re.match(circulatory_code,str(np.array(row['diag_1']))))) and 
              not(bool(re.match(circulatory_code,str(np.array(row['diag_2']))))) and 
              not(bool(re.match(circulatory_code,str(np.array(row['diag_3'])))))
         ): value= 1
        
    if (   not(bool(re.match(diabetes_code,str(np.array(row['diag_1']))))) and
            not(bool(re.match(diabetes_code,str(np.array(row['diag_2'])))))and 
            not(bool(re.match(diabetes_code,str(np.array(row['diag_3'])))))
        ) and (bool(re.match(circulatory_code,str(np.array(row['diag_1'])))) or 
               bool(re.match(circulatory_code,str(np.array(row['diag_2'])))) or 
               bool(re.match(circulatory_code,str(np.array(row['diag_3']))))
         ):
          value= 2
    if (  bool(re.match(diabetes_code,str(np.array(row['diag_1'])))) or 
          bool(re.match(diabetes_code,str(np.array(row['diag_2'])))) or 
          bool(re.match(diabetes_code,str(np.array(row['diag_3']))))
         )and (bool(re.match(circulatory_code,str(np.array(row['diag_1'])))) or 
               bool(re.match(circulatory_code,str(np.array(row['diag_2'])))) or 
               bool(re.match(circulatory_code,str(np.array(row['diag_3']))))
         ):
            value= 3     
    return value

df['comorbidity'] = diagnosis.apply(calculate_Comorbidity, axis=1)

#Making a table for all data in comorbidity
pd.crosstab(df['readmitted'], df['comorbidity'], margins = True)
df.groupby(['comorbidity'])['number_inpatient'].mean().T

# After having comorbidity column we can drop columns diag_1 , diag_2 and diag_3
df = df.drop(columns=["diag_1", "diag_2" , "diag_3"])

keys = ['miglitol', 'repaglinide',  'chlorpropamide', 'acetohexamide','glimepiride', 'glipizide', 'glyburide', 'pioglitazone', 'rosiglitazone', 'acarbose', 'metformin', 'tolbutamide','insulin', 'glyburide-metformin', 'tolazamide', 'metformin-pioglitazone','metformin-rosiglitazone', 'nateglinide','glimepiride-pioglitazone', 'glipizide-metformin', 'troglitazone']

#To check the number of changes in dosage
df['num_of_change'] = 0
for col in keys:
    colname = str(col) + 'temp'
    df[colname] = df[col].apply(lambda x: 0 if (x == 'No' or x == 'Steady') else 1)
    df['num_of_change'] = df['num_of_change'] + df[colname]
    del df[colname]

#To check whether a medicine is given to a patient or not
for col in keys:
    df[col] = df[col].apply(lambda x: 0 if x == 'No' else 1)

#Calculating the number of medicines given to a patient
df['num_of_med'] = 0
for col in keys:
    df['num_of_med'] = df['num_of_med'] + df[col]
df.drop(columns = keys, inplace = True)

#Convert A1C results and glucose serum results into numbers for model to work better
# 1: test result is  abnormal
# 2: test result is  very abnormal
df['A1Cresult'] = df['A1Cresult'].apply(lambda x: 0 if x == 'Norm' else (1 if x =='>7' else (2 if x =='>8' else -1)))
df['max_glu_serum'] = df['max_glu_serum'].apply(lambda x: 0 if x == 'Norm' else (1 if x =='>200' else (2 if x =='>300' else -1)))

#If the medication was changed
df['change'] = df['change'].apply(lambda x: 0 if x == 'No' else 1)
df['gender'] = df['gender'].apply(lambda x: 0 if x == 'Female' else 1)
df['diabetesMed'] = df['diabetesMed'].apply(lambda x: 0 if x == 'No' else 1)

# Create service utilization as a new feature
df['service_utilization'] = df['number_outpatient'] + df['number_emergency'] + df['number_inpatient']

# get list of only numeric features
numerics = list(set(list(df._get_numeric_data().columns))- {'readmitted'})

#Encounter ID and patient number do not help for the model
df.drop(['encounter_id', 'patient_nbr'], axis=1, inplace=True)

#Some data is in days and some is a number (dimensionless). Therefore, we must normalize the data
listnormal = ['time_in_hospital', 'num_lab_procedures', 'num_procedures', 'num_medications', 'number_outpatient', 'number_emergency', 'number_inpatient', 'number_diagnoses']
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
df[listnormal] = sc.fit_transform(df[listnormal])

#Let's store readmitted as target variable and rest of the columns in Independent variables
y = df['readmitted']
X = df.drop(['readmitted'], axis =1)
X = pd.get_dummies(X)

#Defining the test and train set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

#Using the Models

# 1: Logistic Regression 

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
lr = LogisticRegression(tol=1e-7, penalty='l2', C=0.0005,solver='liblinear')
lr.fit(X_train, y_train)
y_pred_lr = lr.predict(X_test)
score_lr =accuracy_score(y_test,y_pred_lr)
# checking the confusion matrix
from sklearn.metrics import confusion_matrix
cm_lr = confusion_matrix(y_test, y_pred_lr)

# 2: Random Forest Classifier

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
random_forest = RandomForestClassifier(random_state=42, n_estimators=100, min_samples_split =10)
gridParam = {'n_estimators': [100,200,400,500], 'min_samples_split': [5,10,15,20]}
rf = GridSearchCV(random_forest, param_grid = gridParam, cv = 3)
rf.fit(X_train, y_train)
y_pred_rf = rd.predict(X_test)
score_rf =accuracy_score(y_test,y_pred_rf)
cm_rf = confusion_matrix(y_test, y_pred_rf)

# 3: Decision Tree Classifier

from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
from sklearn.model_selection import cross_val_score
dtc = DecisionTreeClassifier(max_depth = 5, criterion = 'gini', min_samples_split = 5)
gridParam = {'criterion': ['gini', 'entropy'], 'max_depth': ['5','10','15'], 'min_samples_split': [5,10,15]}
dt = GridSearchCV(dtc, cv=3, param_grid = gridParam)
dt.fit(X_train, y_train)
y_pred_dt = dt.predict(X_test)
score_dt = accuracy_score(y_test, y_pred_dt)
cm_dt = confusion_matrix(y_test, y_pred_dt)

# 4: Adaboost classifier

from sklearn.ensemble import AdaBoostClassifier
ada_boost = AdaBoostClassifier(n_estimators = 20, learning_rate = 0.15, random_state = 42)
gridParam ={'n_estimators': [100,200,400,500],'learning_rate': [0.2,0.5,1.0,0.15]}
adb = GridSearchCV(ada_boost, cv = 3, n_jobs = 3, param_grid = gridParam)
adb.fit(X_train, y_train)
y_pred_adb = adb.predict(X_test)
score_adb = accuracy_score(y_test, y_pred_adb)
cm_adb = confusion_matrix(y_test, y_pred_adb)