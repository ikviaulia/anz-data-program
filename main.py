import pandas as pd 
import numpy as np 


data = pd.read_excel('~/Downloads/ANZ/ANZ synthesised transaction dataset.xlsx', delimiter='/t')
#print (data.head())
#print (data.isnull().sum())

#learn about 1st column card_present_flag
#card = data.replace(to_replace=np.nan, value=0)
#print(data['card_present_flag'].isnull().sum())
#print(firstdf)


#seconddf=pd.isnull(data['card_present_flag']) 
#print(data[seconddf])

data.fillna(data.mean(), inplace=True)

data['bpay_biller_code'].fillna(0, inplace=True)
data['merchant_id'].fillna(0, inplace=True)
data['merchant_suburb'].fillna(0, inplace=True)
data['merchant_state'].fillna(0, inplace=True)
data['merchant_long_lat'].fillna(0, inplace=True)
print (data.isnull().sum())

writer = pd.ExcelWriter('cleaned.xlsx')
data.to_excel(writer)
writer.save()
print('Dataframe is written succesfully to excel file yo')