

# -*- coding: utf-8 -*-
"""
Created on Sat May 30 07:19:59 2020

@author: shenawy

"""
import pandas as pd
import seaborn as sns
#Reading dataset
alarm= pd.read_csv("k.csv")
alarm = alarm.rename(columns={"Alarm Text": "AlarmText"})
df= pd.read_csv("1st_cell.csv")
df =df.rename(columns={"Period start time": "Periodstarttime"})
df_length=len(df)
alarm_length=len(alarm)
df['label']='Normal'
alarm['label']='Normal'

time_outage=[]
k=0
#First_Condition,we determine the outage candidate cells (they mayn't be in outage)
for i in range(alarm_length):
    if alarm.AlarmText.str.contains("outage alarm")[i]:
        time_outage.append(alarm['Alarm Time'][i])
        alarm["label"][i]='outage'
    
time_outage=pd.DataFrame(time_outage)     
length=len(time_outage)  


time_outage=pd.to_datetime(time_outage[0])
df['Periodstarttime'] = pd.to_datetime(df['Periodstarttime'])  

i=0
h=df["OFR_Inter X2 based HO Att E-UTRAN HO Attempts, inter eNB X2 based"]
h=h.apply(pd.to_numeric, args=('coerce',))
mean_HO=h.mean()

for k in range(length):
 for i in range (df_length):
   
   if  time_outage[k].hour+1 == df['Periodstarttime'][i].hour and time_outage[k].date() == df['Periodstarttime'][i].date():
     rr_calc=(float(df["OFR_Inter X2 based HO Att E-UTRAN HO Attempts, inter eNB X2 based"][i])-mean_HO)*100/mean_HO
     if (df['incoming HO succ rate'][i] == 0 and df['incoming HO succ rate'][i-1]>0 and df['Cell Availability'][i]==0 and rr_calc < 50) :
         df['label'][i]='outage'         
     else :    
          df['label'][i]='Normal'

              
#sns.lmplot(data=df, x= "incoming HO succ rate", y= "Cell Availability", hue="label", fit_reg=False , size=4, aspect=1.5)

# print cell outage list
'''
writer = pd.ExcelWriter('A:\\Outage_cells.xlsx', engine='xlsxwriter')

outage_list.to_excel(writer,'Sheet1',index=False)

writer.save()

'''
#Third_condition,check the powersaving mode or switching off cell by operator(but there is no data available)








































