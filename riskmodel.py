
# coding: utf-8


# importing relevant libraries
import matplotlib
# Force matplotlib to not use any Xwindows backend.
matplotlib.use('Agg')
import pandas as pd
import numpy as np
import sqlalchemy as sa
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import scale
import pandas as pd
from sklearn import datasets, linear_model, cross_validation, grid_search
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.cross_validation import KFold, StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.grid_search import GridSearchCV
from sklearn_pandas import DataFrameMapper
from sklearn.preprocessing import OneHotEncoder
from sklearn import metrics
from sklearn.metrics import confusion_matrix
from xgboost import XGBClassifier
from sklearn.ensemble import ExtraTreesClassifier
import datetime
from dateutil.relativedelta import relativedelta
import os
import functools


# Turn off pandas chained assignment warning
pd.options.mode.chained_assignment = None  # default='warn'
pd.options.display.max_columns = 999


# ### 1. CLEAN PLI, PITT & TAX DATA


# # create directory paths for opening files
curr_path = os.path.dirname(os.path.realpath(__file__))
dataset_path = os.path.join(curr_path, "datasets/")
inter_path = os.path.join(curr_path,"interResults/")

# read in data
# Reading plidata
plidata = pd.read_csv(os.path.join(dataset_path, "pli.csv"), encoding='utf-8', dtype={'STREET_NUM': 'str', 'STREET_NAME': 'str'}, low_memory=False)
# Reading city of Pittsburgh dataset
pittdata = pd.read_csv(os.path.join(dataset_path, "pittdata.csv"), encoding="ISO-8859-1", dtype={'PROPERTYADDRESS': 'str', 'PROPERTYHOUSENUM': 'str', 'CLASSDESC': 'str'}, low_memory=False)
# Reading tax data
taxdata = pd.read_csv("./datasets/tax.csv", encoding='utf-8')
#read parcel data (matches parcels to census tract and block group
parcel = pd.read_csv(os.path.join(dataset_path, "parcels.csv"), encoding='utf-8')
#read ACS data
acs_data = ['acs_income.csv','acs_occupancy.csv','acs_year_built.csv','acs_year_moved.csv']
def clean_acs(df):
    #Use descriptive names in first row
    df.columns = df.loc[0]
    df = df.drop(0)
    df = df.drop(['Id', 'Id2'], axis=1)
    #Extract census block and tract
    df[['BLOCKCE10', 'TRACTCE10']] = df['Geography'].str.extract(
        'Block Group (\d), Census Tract (\d+\.?\d*)')
    df = df.drop(['Geography'], axis=1)
    #Drop first two columns since they only contain totals
    df = df.drop(df.columns[[0,1]], axis=1)
    #Drop margin of errors
    df = df.drop(df.columns[df.columns.str.startswith('Margin')], axis=1)
    #Convert to numbers
    df['BLOCKCE10'] = df['BLOCKCE10'].astype('float')
    df['TRACTCE10'] = df['TRACTCE10'].astype('float')
    #Multiply tract by 100 to be consistent with other data
    df['TRACTCE10'] = df['TRACTCE10'] * 100
    return df
acs_data = map(lambda x: os.path.join(dataset_path, x), acs_data)
acs_data = map(pd.read_csv, acs_data)
acs_data = map(clean_acs, acs_data)
#Merge datasets together
acs_data_combined = functools.reduce(lambda x,y:x.merge(y, how='outer', on=['BLOCKCE10','TRACTCE10']), acs_data)

# cleaning pitt dataset
# removing all properties outside Pittsburgh, Wilkinsburg, and Ingram
pittdata = pittdata[(pittdata.PROPERTYCITY == 'PITTSBURGH')]  # & (pittdata.PROPERTYCITY == 'WILKINSBURG') & (pittdata.PROPERTYCITY == 'INGRAM')]
# include only residential data
pittdata = pittdata[pittdata['CLASSDESC'] == 'RESIDENTIAL']
address_parcels = pittdata[['PARID','PROPERTYADDRESS','PROPERTYHOUSENUM']].drop_duplicates()
pittdata = pittdata[pittdata['PROPERTYHOUSENUM'] != '0']
pittdata = pittdata[pittdata['PROPERTYADDRESS'] != '']
# dropping columns with less than 15% data
pittdata = pittdata.dropna(thresh=4000, axis=1)
pittdata = pittdata.rename(columns={pittdata.columns[0]: 'PARID'})
# pick out necessary columns
pittdata = pittdata[['PARID','PROPERTYHOUSENUM','PROPERTYADDRESS','MUNIDESC','SCHOOLDESC','NEIGHCODE',
                     'TAXDESC','OWNERDESC','USEDESC','LOTAREA','SALEPRICE','FAIRMARKETBUILDING','FAIRMARKETLAND']]
pittdata = pittdata.drop_duplicates()

# cleaning pli dataset
# removing extra whitespaces
plidata['STREET_NAME'] = plidata['STREET_NAME'].str.strip()
plidata['STREET_NUM'] = plidata['STREET_NUM'].str.strip()
# include only residential data
plidata = pd.merge(plidata, address_parcels[['PARID']], how='inner',left_on=['PARCEL'], right_on=['PARID'])
# pick out necessary columns
plidata=plidata[['PARCEL', 'INSPECTION_DATE', 'INSPECTION_RESULT', 'VIOLATION']]
# converting to datetime
plidata.INSPECTION_DATE = pd.to_datetime(plidata.INSPECTION_DATE)
plidata['violation_year'] = plidata['INSPECTION_DATE'].map(lambda x: x.year)
plidata = plidata.drop_duplicates()

# cleaning tax dataset
# removing all properties outside Pittsburgh, Wilkinsburg, and Ingram
taxdata = taxdata[(taxdata.municipality == 'Pittsburgh')]  # & (tax.municipality == 'Wilkinsburg Boro') & (tax.municipality == 'Ingram Boro')]
taxdata = taxdata[['pin', 'filing_date', 'tax_year', 'lien_description', 'amount', 'satisfied']]
taxdata = taxdata.dropna(subset=['pin', 'tax_year', 'lien_description', 'amount', 'satisfied'])
# include only residential data
taxdata = pd.merge(taxdata, address_parcels[['PARID']], how='inner', left_on=['pin'], right_on=['PARID'])
# pick out necessary columns
taxdata = taxdata[['pin', 'filing_date', 'tax_year', 'lien_description', 'amount','satisfied']]
taxdata.filing_data = pd.to_datetime(taxdata.filing_date)
taxdata.tax_year = pd.to_datetime(taxdata.tax_year)
taxdata = taxdata.drop_duplicates()

# cleaning parcel dataset
# keep only parcel, tract, and block group
parcel = parcel[(parcel.geo_name_cousub == 'Pittsburgh city')]
parcel_blocks = parcel[['PIN', 'TRACTCE10', 'BLOCKCE10']]
#get first digit of block, convert to int
parcel_blocks['BLOCKCE10'] = parcel_blocks['BLOCKCE10'].astype(str).str[0].astype(float)
#ignore bad parcels
parcel_blocks = parcel_blocks[parcel_blocks['PIN'] != ' ']
parcel_blocks = parcel_blocks[parcel_blocks['PIN'] != 'COMMON GROUND']
parcel_blocks = parcel_blocks[~parcel_blocks['PIN'].str.match('.*County')]
parcel_blocks=parcel_blocks.drop_duplicates()


# #### 1.1 Aggregate pittdata to census block, then merge with acs data


pittdata_blocks=pd.merge(pittdata, parcel_blocks, how='left', left_on=['PARID'], right_on=['PIN'])
#drop extra columns
pittdata_blocks = pittdata_blocks.drop(['PARID','PIN','PROPERTYHOUSENUM','PROPERTYADDRESS'], axis=1)


#group by blocks
grouped = pittdata_blocks.groupby(['TRACTCE10','BLOCKCE10'])
#change the '-DESC' columns to the most common in each group (block)
#change the other columns to the mean
max_count = lambda x:x.value_counts().index[0]
pittdata_blocks = grouped.agg({
    'MUNIDESC':max_count,'SCHOOLDESC':max_count,'NEIGHCODE':max_count,
    'TAXDESC':max_count,'OWNERDESC':max_count,'USEDESC':max_count,'LOTAREA':np.mean,
    'SALEPRICE':np.mean,'FAIRMARKETBUILDING':np.mean,'FAIRMARKETLAND':np.mean
})
#reset index to columns
pittdata_blocks = pittdata_blocks.reset_index(level=[0,1])
#merge pittdata with acs
pittacs = pd.merge(pittdata_blocks, acs_data_combined, how='inner', on=['BLOCKCE10','TRACTCE10'])

# keep a copy of blocks and tracts
blocks = pittacs[['TRACTCE10','BLOCKCE10']].drop_duplicates()


# #### 1.2 merge plidata with census block¶


#group by blocks
plidata_blocks = pd.merge(plidata, parcel_blocks, how='left', left_on=['PARCEL'], right_on=['PIN'])
#drop extra columns
plidata_blocks = plidata_blocks.drop(['PARCEL','PIN'], axis=1)
plidata_blocks=plidata_blocks.dropna(subset=['TRACTCE10','BLOCKCE10'])


# #### 1.3 Aggregate taxdata to census block¶


# group by blocks
taxdata_blocks = pd.merge(taxdata,parcel_blocks, how='left', left_on=['pin'], right_on=['PIN'])
taxdata_blocks = taxdata_blocks.drop(['pin','PIN','filing_date','satisfied'],axis=1)
taxdata_blocks = taxdata_blocks.dropna(subset=['TRACTCE10','BLOCKCE10'])


# ### 2. Clean fire incident data


# loading fire incidents csvs
fire_pre14 = pd.read_csv(os.path.join(dataset_path, "Fire_Incidents_Pre14.csv"), encoding='latin-1', dtype={'street': 'str', 'number': 'str'}, low_memory=False)
fire_new = pd.read_csv(os.path.join(dataset_path, "Fire_Incidents_New.csv"), encoding='utf-8', dtype={'street': 'str', 'number': 'str'}, low_memory=False)

# cleaning columns of fire_pre14
fire_pre14['full.code'] = fire_pre14['full.code'].str.replace('  -', ' -')
fire_pre14['st_type'] = fire_pre14['st_type'].str.strip()
fire_pre14['street'] = fire_pre14['street'].str.strip()
fire_pre14['number'] = fire_pre14['number'].str.strip()
fire_pre14['st_type'] = fire_pre14['st_type'].str.replace('AV', 'AVE')
fire_pre14['street'] = fire_pre14['street'].str.strip() + ' ' + fire_pre14['st_type'].str.strip()

# drop irrelevant columns
pre14_drop = ['Unnamed: 0','PRIMARY_UNIT', 'MAP_PAGE', 'alm_dttm', 'arv_dttm', 'XCOORD', 
              'YCOORD','inci_id', 'inci_type', 'alarms', 'st_prefix',
              'st_suffix', 'st_type', 'CALL_NO','descript','ï..AGENCY']
for col in pre14_drop:
  del fire_pre14[col]


post14_drop = ['alm_dttm', 'arv_dttm', 'XCOORD', 'YCOORD', 'alarms', 
               'inci_type', 'CALL_NO','descript']
for col in post14_drop:
  del fire_new[col]

# joining both the fire incidents file together
fire_new = fire_new.append(fire_pre14, ignore_index=True)
fire_new = fire_new[fire_new['full.code'].str.strip() != '540 - Animal problem, Other']
fire_new = fire_new[fire_new['full.code'].str.strip() != '5532 - Public Education (Station Visit)']
fire_new = fire_new[fire_new['full.code'].str.strip() != '353 - Removal of victim(s) from stalled elevator']

# correcting problems with the street column
fire_new['street'] = fire_new['street'].replace(to_replace=', PGH', value='', regex=True)
fire_new['street'] = fire_new['street'].replace(to_replace=', P', value='', regex=True)
fire_new['street'] = fire_new['street'].replace(to_replace=',', value='', regex=True)
fire_new['street'] = fire_new['street'].replace(to_replace='#.*', value='', regex=True)
fire_new['street'] = fire_new['street'].str.strip()
fire_new['number'] = fire_new['number'].str.strip()

# converting to date time and extracting year
fireDate, fireTime = fire_new['CALL_CREATED_DATE'].str.split(' ', 1).str
fire_new['CALL_CREATED_DATE'] = fireDate
fire_new['CALL_CREATED_DATE'] = pd.to_datetime(fire_new['CALL_CREATED_DATE'])
fire_new['fire_year'] = fire_new['CALL_CREATED_DATE'].map(lambda x: x.year)

# removing all codes with less than 20 occurences
for col, val in fire_new['full.code'].value_counts().iteritems():
    if val < 20 and col[0] != '1':
        fire_new = fire_new[fire_new['full.code'] != col]

#Split street column when there are 2 streets
street_split = fire_new['street'].str.split('/')
fire_new['street'] = street_split.map(lambda x:x[0])
fire_new = fire_new.dropna(subset=['CALL_CREATED_DATE'])
fire_new = fire_new.drop_duplicates()


# #### 2.1 merge fire incident to census block

# convert from addresses to parcels
fire_parcel = pd.merge(fire_new, address_parcels, how='inner',
                        left_on=['street','number'], right_on=['PROPERTYADDRESS','PROPERTYHOUSENUM'])
# convert from parcels to census blocks
fire_blocks = pd.merge(fire_parcel, parcel_blocks, how='left',left_on=['PARID'], right_on=['PIN'])
#drop extra columns
fire_blocks=fire_blocks.drop(['number','street','PARID','PROPERTYADDRESS',
                              'PROPERTYHOUSENUM','PIN', 'Unnamed: 0',
                              'st_prefix', 'st_suffix', 'st_type',
                              'prop_use_code','response_time',
                              'CALL_TYPE_FINAL', 'COUNCIL', 'NEIGHBORHOOD',
                              'PRIMARY_UNIT','fire_year','prop_use_descript'],axis=1)
#drop data without block or tract (this drops non-residential data)
fire_blocks = fire_blocks.dropna(subset=['TRACTCE10','BLOCKCE10'])
# dropping columns with less than 15% data
fire_blocks = fire_blocks.dropna(thresh=len(fire_blocks)*0.15, axis=1)
fire_blocks = fire_blocks.drop_duplicates()


# ### 3 Join four datasets together

# #### 3.1 joining dynamic data with fire incidents


# making the fire column with all type 100s as fires and map it to 0 or 1
fire_blocks['fire'] = fire_blocks['full.code'].astype(str).                    map(lambda x: 1 if x[0]=='1' else 0)
# keep non-fire incidents as features
nonfire_incidents = fire_blocks[fire_blocks['fire'] != 1]
nonfire_incidents = nonfire_incidents[['CALL_CREATED_DATE','full.code','TRACTCE10', 'BLOCKCE10']]
fire_blocks.drop('full.code',axis=1,inplace=True)



# group by every certain period of time
# reason for setting period to year: tax data is based on year
period = 'Y' 
fire_groups = fire_blocks.groupby(pd.Grouper(key='CALL_CREATED_DATE', freq=period))
nonfire_groups = nonfire_incidents.groupby(pd.Grouper(key='CALL_CREATED_DATE', freq=period))
plidata_groups = plidata_blocks.groupby(pd.Grouper(key='INSPECTION_DATE', freq=period))
taxdata_groups = taxdata_blocks.groupby(pd.Grouper(key='tax_year', freq=period))


# then group fire by census blocks
def groupByBlock(df,categoricals, method):
    dummies=[pd.get_dummies(df[feature]) for feature in categoricals]
    df = pd.concat([df]+dummies,axis=1)
    df.drop(categoricals,axis=1,inplace=True)
    df = pd.merge(df, blocks, how='right',on=['TRACTCE10','BLOCKCE10'])
    df_grouped=df.groupby(['TRACTCE10','BLOCKCE10'])
    if method == 'max':
        df_grouped=df_grouped.max()
    if method == 'sum':
        df_grouped=df_grouped.sum()
    return df_grouped
fire_divided = fire_groups.apply(groupByBlock,categoricals=[],method='max')
fire_divided.drop('CALL_CREATED_DATE',axis=1,inplace=True)
fire_divided=fire_divided.reset_index()
fire_divided=fire_divided.fillna(0)


# group nonfire incidents by census blocks
nonfire_divided = nonfire_groups.apply(groupByBlock,categoricals=['full.code'],method='sum')
nonfire_divided=nonfire_divided.reset_index()
nonfire_divided=nonfire_divided.fillna(0)



# group pli incidents by census blocks
def groupByBlock_pli(df):
    INSPECTION_RESULT_dummies=pd.get_dummies(df['INSPECTION_RESULT'])
    VIOLATION_dummies=df['VIOLATION'].str.get_dummies(sep=' :: ')
    df = pd.concat([df,INSPECTION_RESULT_dummies,VIOLATION_dummies],axis=1)
    df.drop(['INSPECTION_RESULT','VIOLATION','violation_year'],axis=1,inplace=True)
    df = pd.merge(df, blocks, how='right',on=['TRACTCE10','BLOCKCE10'])
    df_grouped=df.groupby(['TRACTCE10','BLOCKCE10']).sum()
    return df_grouped
pli_divided=plidata_groups.apply(groupByBlock_pli)
pli_divided=pli_divided.reset_index()
pli_divided=pli_divided.fillna(0)



# group tax data by census blocks
tax_divided=taxdata_groups.apply(groupByBlock,categoricals=['lien_description'],method='sum')
tax_divided=tax_divided.reset_index()
tax_divided=tax_divided.fillna(0)



# join fire, nonfire, pli, tax data together
fire_nonfire = pd.merge(fire_divided,nonfire_divided,how='outer',
                        on=['CALL_CREATED_DATE','TRACTCE10','BLOCKCE10'])
fire_nonfire_pli = pd.merge(fire_nonfire,pli_divided,how='outer',
                           left_on=['CALL_CREATED_DATE','TRACTCE10','BLOCKCE10'],
                           right_on=['INSPECTION_DATE','TRACTCE10','BLOCKCE10'])
fire_nonfire_pli_tax = pd.merge(fire_nonfire_pli,tax_divided,how='outer',
                               left_on=['CALL_CREATED_DATE','TRACTCE10','BLOCKCE10'],
                               right_on=['tax_year','TRACTCE10','BLOCKCE10'])
fire_nonfire_pli_tax['CALL_CREATED_DATE']=fire_nonfire_pli_tax['CALL_CREATED_DATE'].                                          fillna(fire_nonfire_pli_tax['CALL_CREATED_DATE'])
fire_nonfire_pli_tax.drop(['INSPECTION_DATE','tax_year'],axis=1,inplace=True)



# drop columns with less than thresold% data
threshold=0.0001
s=fire_nonfire_pli_tax.sum()
drop_columns=s[s<len(fire_nonfire_pli_tax)*threshold].index
fire_nonfire_pli_tax.drop(drop_columns,axis=1,inplace=True)



# join with pitt_blocks
combined = pd.merge(fire_nonfire_pli_tax,pittacs,
                    how='left',on=['TRACTCE10','BLOCKCE10'])
features = ['SCHOOLDESC', 'OWNERDESC', 'MUNIDESC', 'NEIGHCODE','TAXDESC', 'USEDESC']
dummies= [pd.get_dummies(combined[feature]) for feature in features]
encoded_combined=pd.concat([combined]+dummies,axis=1)
encoded_combined.drop(features,axis=1,inplace=True)
encoded_combined=encoded_combined.dropna(subset=['CALL_CREATED_DATE'])
encoded_combined = encoded_combined.drop_duplicates()
encoded_combined=encoded_combined.fillna(0)


# ### 4 Split data into training set and test set

# PREPARING THE TESTING DATA (final 1 year of data)
cutoff = datetime.datetime.now() - relativedelta(years=1)
cutoffdate = cutoff.strftime("%m/%d/%Y")
# preparing training set
traindata = encoded_combined[encoded_combined.CALL_CREATED_DATE <= cutoffdate]
traindata.drop(['CALL_CREATED_DATE','TRACTCE10','BLOCKCE10'],axis=1,inplace=True)
traindata.fillna(0)
X_train=np.array(traindata.drop(['fire'],axis=1))
y_train=np.array(traindata['fire'])


# preparing test set
testdata = encoded_combined[encoded_combined.CALL_CREATED_DATE > cutoffdate]
testdata.drop(['CALL_CREATED_DATE','TRACTCE10','BLOCKCE10'],axis=1,inplace=True)
testdata.fillna(0)
X_test=np.array(testdata.drop(['fire'],axis=1))
y_test=np.array(testdata['fire'])



# The XG Boost model
model = XGBClassifier(learning_rate=0.13,
                      n_estimators=1500,
                      max_depth=5, min_child_weight=1,
                      gamma=0,
                      subsample=0.8,
                      colsample_bytree=0.8,
                      objective='binary:logistic',
                      nthread=4,
                      seed=27)
model.fit(X_train, y_train)
pred = model.predict(X_test)
real = y_test
cm_xg = confusion_matrix(real, pred)
print(cm_xg)

from sklearn.metrics import cohen_kappa_score
kappa_xg = cohen_kappa_score(real, pred)

fpr, tpr, thresholds = metrics.roc_curve(y_test, pred, pos_label=1)
roc_auc = metrics.auc(fpr, tpr)

acc_xg = 'Accuracy = {0} \n \n'.format(float(cm_xg[0][0] + cm_xg[1][1]) / len(real))
kapp_xg = 'kappa score = {0} \n \n'.format(kappa_xg)
auc_xg = 'AUC Score = {0} \n \n'.format(metrics.auc(fpr, tpr))
recall_xg = 'recall = {0} \n \n'.format(tpr[1])
precis_xg = 'precision = {0} \n \n'.format(float(cm_xg[1][1]) / (cm_xg[1][1] + cm_xg[0][1]))

print(acc_xg)
print(kapp_xg)
print(auc_xg)
print(recall_xg)
print(precis_xg)


# Write model performance to log file:
log_path = os.path.join(curr_path, "log/")

with open('{0}ModelPerformance_XGBoost_{1}.txt'.format(log_path, datetime.datetime.now().strftime('%m%d-%H%M%S')), 'a') as log_file:
  log_file.write("Confusion Matrix: \n \n")
  log_file.write(np.array2string(cm_xg)+"\n \n")
  log_file.write("Model performance metrics: \n \n")
  log_file.write(acc_xg)
  log_file.write(kapp_xg)
  log_file.write(auc_xg)
  log_file.write(recall_xg)
  log_file.write(precis_xg)


# Adaboost model
from sklearn.ensemble import AdaBoostClassifier

model_adaboost = AdaBoostClassifier(n_estimators = 65, random_state=27)
model_adaboost.fit(X_train, y_train)
pred_adaboost = model_adaboost.predict(X_test)
real = y_test
cm_ada = confusion_matrix(real, pred_adaboost)
print(cm_ada)

kappa_ada = cohen_kappa_score(real, pred_adaboost)

fpr, tpr, thresholds = metrics.roc_curve(y_test, pred_adaboost, pos_label=1)
roc_auc = metrics.auc(fpr, tpr)

acc_ada = 'Accuracy = {0} \n \n'.format(float(cm_ada[0][0] + cm_ada[1][1]) / len(real))
kapp_ada = 'kappa score = {0} \n \n'.format(kappa_ada)
auc_ada = 'AUC Score = {0} \n \n'.format(metrics.auc(fpr, tpr))
recall_ada = 'recall = {0} \n \n'.format(tpr[1])
precis_ada = 'precision = {0} \n \n'.format(float(cm_ada[1][1]) / (cm_ada[1][1] + cm_ada[0][1]))

print(acc_ada)
print(kapp_ada)
print(auc_ada)
print(recall_ada)
print(precis_ada)

# Write model performance to log file:
with open('{0}ModelPerformance_AdaBoost_{1}.txt'.format(log_path, datetime.datetime.now().strftime('%m%d-%H%M%S')), 'a') as log_file:
  log_file.write("Confusion Matrix: \n \n")
  log_file.write(np.array2string(cm_ada)+"\n \n")
  log_file.write("Model performance metrics: \n \n")
  log_file.write(acc_ada)
  log_file.write(kapp_ada)
  log_file.write(auc_ada)
  log_file.write(recall_ada)
  log_file.write(precis_ada)


# Adaboost model
from sklearn.ensemble import RandomForestClassifier

model_rf = RandomForestClassifier(n_estimators = 65)
model_rf.fit(X_train, y_train)
pred_rf = model_rf.predict(X_test)
real = y_test
cm_rf = confusion_matrix(real, pred_rf)
print(cm_rf)

kappa_rf = cohen_kappa_score(real, pred_rf)

fpr, tpr, thresholds = metrics.roc_curve(y_test, pred_rf, pos_label=1)
roc_auc = metrics.auc(fpr, tpr)

acc_rf = 'Accuracy = {0} \n \n'.format(float(cm_rf[0][0] + cm_rf[1][1]) / len(real))
kapp_rf = 'kappa score = {0} \n \n'.format(kappa_rf)
auc_rf = 'AUC Score = {0} \n \n'.format(metrics.auc(fpr, tpr))
recall_rf = 'recall = {0} \n \n'.format(tpr[1])
precis_rf = 'precision = {0} \n \n'.format(float(cm_rf[1][1]) / (cm_rf[1][1] + cm_rf[0][1]))

print(acc_rf)
print(kapp_rf)
print(auc_rf)
print(recall_rf)
print(precis_rf)

# Write model performance to log file:
with open('{0}ModelPerformance_RF_{1}.txt'.format(log_path, datetime.datetime.now().strftime('%m%d-%H%M%S')), 'a') as log_file:
  log_file.write("Confusion Matrix: \n \n")
  log_file.write(np.array2string(cm_rf)+"\n \n")
  log_file.write("Model performance metrics: \n \n")
  log_file.write(acc_rf)
  log_file.write(kapp_rf)
  log_file.write(auc_rf)
  log_file.write(recall_rf)
  log_file.write(precis_rf)


from sklearn import linear_model
samples = np.array([0.1 if i == 0 else 1.2 for i in y_train])
model_lg = linear_model.LogisticRegression(C=1e5)
model_lg.fit(X_train, y_train,sample_weight = samples)
pred_lg = model.predict(X_test)
real = y_test
cm_lg = confusion_matrix(real, pred_lg)
print (cm_lg)

kappa_lg = cohen_kappa_score(real, pred)

fpr, tpr, thresholds = metrics.roc_curve(y_test, pred_rf, pos_label=1)
roc_auc = metrics.auc(fpr, tpr)

acc_lg = 'Accuracy = {0} \n \n'.format(float(cm_lg[0][0] + cm_lg[1][1]) / len(real))
kapp_lg = 'kappa score = {0} \n \n'.format(kappa_lg)
auc_lg = 'AUC Score = {0} \n \n'.format(metrics.auc(fpr, tpr))
recall_lg = 'recall = {0} \n \n'.format(tpr[1])
precis_lg = 'precision = {0} \n \n'.format(float(cm_lg[1][1]) / (cm_lg[1][1] + cm_lg[0][1]))

print(acc_lg)
print(kapp_lg)
print(auc_lg)
print(recall_lg)
print(precis_lg)

# Write model performance to log file:
with open('{0}ModelPerformance_LG_{1}.txt'.format(log_path, datetime.datetime.now().strftime('%m%d-%H%M%S')), 'a') as log_file:
  log_file.write("Confusion Matrix: \n \n")
  log_file.write(np.array2string(cm_lg)+"\n \n")
  log_file.write("Model performance metrics: \n \n")
  log_file.write(acc_lg)
  log_file.write(kapp_lg)
  log_file.write(auc_lg)
  log_file.write(recall_lg)
  log_file.write(precis_lg)
