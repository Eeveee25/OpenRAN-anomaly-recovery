
import pandas as pd

from pandas import ExcelWriter
import numpy as np
import datetime as DT
#from sklearn.decomposition import PCA
#pca = PCA(n_components=2)
#X=pca.fit(related_kpis).transform(related_kpis)

#data =pd.read_csv("L_D_INT-RD1_0426AL.csv")
dataKPIS =pd.read_csv("Orange_LTE18_main_KPIs-RSLTE-LNCEL-2-hour.csv")
#df = data.iloc[:,[4,7,8,11]]
#df = df.rename(columns={"Alarm Text": "AlarmText"})
#df = df.rename(columns={"Alarm Time": "AlarmTime"})
#df = df.rename(columns={"Distinguished Name": "DistinguishedName"})
#dataKPIS = dataKPIS.rename(columns={"Period start time": "Periodstarttime"})

#n_rows,_= df.shape
#df=df.assign(label='normal')
#for i in range(n_rows):
    
   #if df.AlarmText.str.contains("CELL FAULTY")[i] or df.AlarmText.str.contains("BASE STATION CONNECTIVITY LOST")[i] :
  
       
      # df['Label'][i] ='outage'
       
   #else :  
       
       #df['Label'][i] ='Normal'
       
       #dd=df.iloc[:,2]
      #unique_classes = np.unique(dd)
  
#d1=df[df.DistinguishedName.str.contains("PLMN-PLMN/MRBTS-21636/LNBTS-21636/LNCEL-1")]
#d2=df[df.DistinguishedName.str.contains("PLMN-PLMN/MRBTS-21636/LNBTS-21636/LNCEL-2")]
#d3=df[df.DistinguishedName.str.contains("PLMN-PLMN/MRBTS-21636/LNBTS-21636/LNCEL-3")]
#d1['AlarmTime'] = pd.to_datetime(d1['AlarmTime'])
#d2['AlarmTime'] = pd.to_datetime(d2['AlarmTime'])
#d3['AlarmTime'] = pd.to_datetime(d3['AlarmTime'])

finalkpi=dataKPIS[dataKPIS.DN.str.contains("PLMN-PLMN/MRBTS-21636")]
KPIs=finalkpi.iloc[:,[0,3,29,8,9,34,51,10]]
#KPIs = KPIs.rename(columns={"Cell Availability": "CellAvailability"})




d4=finalkpi[finalkpi.DN.str.contains("PLMN-PLMN/MRBTS-21636/LNBTS-21636/LNCEL-1")]
d5=finalkpi[finalkpi.DN.str.contains("PLMN-PLMN/MRBTS-21636/LNBTS-21636/LNCEL-2")]
d6=finalkpi[finalkpi.DN.str.contains("PLMN-PLMN/MRBTS-21636/LNBTS-21636/LNCEL-3")]
d7=finalkpi[finalkpi.DN.str.contains("PLMN-PLMN/MRBTS-21636/LNBTS-21636/LNCEL-4")]

#d4['Periodstarttime'] = pd.to_datetime(d4['Periodstarttime'])
#d5['Periodstarttime'] = pd.to_datetime(d5['Periodstarttime'])
#d6['Periodstarttime'] = pd.to_datetime(d6['Periodstarttime'])
#kpisite2= pd.concat([d4, d5 , d6 ])   

########################## Traffic ###################################
#n,_=d4.shape

#indexval=[]
#for k in range (n):  
       #today = d4.Periodstarttime[d4.index[k]]
       #week_ago = today - DT.timedelta(days=7)       
      # indexval= d4[d4.Periodstarttime == week_ago]
      # if len(indexval) == 0 :
     #       d4['ULTrafficVolumeweek_ago'][d4.index[k]] = '0'
    #   else :     
   #        c=int(indexval.index.values)
 #          d4['ULTrafficVolumeweek_ago'][d4.index[k]]=d4.ULTrafficVolume[c]
    
#################################################################################    

d44=d4.apply(pd.to_numeric, args=('coerce',))
k=d44[["Call Drop Rate","LTE_call_setup_success_rate","incoming HO succ rate"]]
n,_=k.shape

k=k.assign(label='normal')
n,_=k.shape

for i in range(n):
    
     if k["LTE_call_setup_success_rate"][k.index[i]] < 90 or  k["Call Drop Rate"][k.index[i]] > 2 or (k["incoming HO succ rate"][k.index[i]] <90 ):
        k['label'][k.index[i]]='critical degradation'
     elif 90 < k["LTE_call_setup_success_rate"][k.index[i]] < 98 or 90 < k["incoming HO succ rate"][k.index[i]] < 98:
        k['label'][k.index[i]]='medium degradation'   
     else :
         k["label"][k.index[i]]='Normal' 

#################################################################################
         
d55=d5.apply(pd.to_numeric, args=('coerce',))
k1=d55[["Call Drop Rate","LTE_call_setup_success_rate","incoming HO succ rate"]]

n,_=k1.shape
k1=k1.assign(label='normal')
i=0
for i in range(n):
    
     if k1["LTE_call_setup_success_rate"][k1.index[i]] < 90 or  k1["Call Drop Rate"][k1.index[i]] > 2 or (k1["incoming HO succ rate"][k1.index[i]] <90 ):
        k1['label'][k.index[i]]='critical degradation'
     elif 90 < k1["LTE_call_setup_success_rate"][k1.index[i]] < 98 or 90 < k1["incoming HO succ rate"][k1.index[i]] < 98:
        k1['label'][k1.index[i]]='medium degradation'   
     else :
         k1["label"][k1.index[i]]='Normal' 
#################################################################################
         
d66=d6.apply(pd.to_numeric, args=('coerce',))
k2=d66[["Call Drop Rate","LTE_call_setup_success_rate","incoming HO succ rate"]]

n,_=k2.shape
k2=k2.assign(label='normal')
i=0
for i in range(n):
    
     if k2["LTE_call_setup_success_rate"][k2.index[i]] < 90 or  k2["Call Drop Rate"][k2.index[i]] > 2 or (k2["incoming HO succ rate"][k2.index[i]] <90 ):
        k2['label'][k2.index[i]]='critical degradation'
     elif 90 < k2["LTE_call_setup_success_rate"][k2.index[i]] < 98 or 90 < k2["incoming HO succ rate"][k2.index[i]] < 98:
        k2['label'][k2.index[i]]='medium degradation'   
     else :
         k2["label"][k2.index[i]]='Normal' 
###################################################################################
         
d77=d7.apply(pd.to_numeric, args=('coerce',))
k3=d77[["Call Drop Rate","LTE_call_setup_success_rate","incoming HO succ rate"]]

n,_=k3.shape
k3=k3.assign(label='normal')
i=0
for i in range(n):
    
     if k3["LTE_call_setup_success_rate"][k3.index[i]] < 90 or  k3["Call Drop Rate"][k3.index[i]] > 2 or (k3["incoming HO succ rate"][k3.index[i]] <90 ):
        k3['label'][k3.index[i]]='critical degradation'
     elif 90 < k3["LTE_call_setup_success_rate"][k3.index[i]] < 98 or 90 < k3["incoming HO succ rate"][k3.index[i]] < 98:
        k3['label'][k3.index[i]]='medium degradation'   
     else :
         k3["label"][k3.index[i]]='Normal'         

kpisite2= pd.concat([k, k1 , k2 ,k3])   




#writer = ExcelWriter('site 2.xlsx')
#kpisite2.to_excel(writer,'Sheet1',index=False)
#writer.save()














































