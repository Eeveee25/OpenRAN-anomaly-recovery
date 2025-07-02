# -*- coding: utf-8 -*-
"""
Created on Thu Aug 20 00:14:07 2020

@author: Hp
"""


import numpy as np
import pandas as pd
from sklearn.preprocessing import Imputer


imputer=Imputer(missing_values='NaN',strategy='mean',axis=0)
intinal_days=[]
q=[]
cell_cor=[]
l=[]
ll=[]
H=[]
cell_correlation=[]
#from pandas import ExcelWriter
kpi = pd.read_excel("Orange_LTE18_main_KPIs-RSLTE-LNCEL-2-hour.xlsx")
kpi= kpi[["DN","Period start time","LTE_call_setup_success_rate","Call Drop Rate","RRC Setup Success Rate","ERAB AbnormRel(times)","OFR_Inter X2 based HO Att E-UTRAN HO Attempts, inter eNB X2 based","UL Traffic Volume(GB)","DL Traffic Volume (GB)"]].copy()
#cell 1
kpi1 = kpi[(kpi["DN"]=="PLMN-PLMN/MRBTS-20830/LNBTS-20830/LNCEL-2")]
kpi1=kpi1.reset_index(drop=True)
#cell 2
kpi2 = kpi[(kpi["DN"]=="PLMN-PLMN/MRBTS-20830/LNBTS-20830/LNCEL-3")]
kpi2=kpi2.reset_index(drop=True)
#cell 3
kpi3 = kpi[(kpi["DN"]=="PLMN-PLMN/MRBTS-20830/LNBTS-20830/LNCEL-4")]
kpi3=kpi3.reset_index(drop=True)
#
kpi1['Period start time'] =pd.to_datetime(kpi1['Period start time'])
kpi1 = kpi1.sort_values(by='Period start time')
kpi1=kpi1.reset_index(drop=True)

kpi2['Period start time'] =pd.to_datetime(kpi2['Period start time'])
kpi2 = kpi2.sort_values(by='Period start time')
kpi2=kpi2.reset_index(drop=True)

kpi3['Period start time'] =pd.to_datetime(kpi3['Period start time'])
kpi3 = kpi3.sort_values(by='Period start time')
kpi3=kpi3.reset_index(drop=True)



site=kpi1.append([kpi2,kpi3], ignore_index = True)
df =pd.DataFrame(site)
df1=pd.DataFrame(kpi1
    ,columns= ["DN","Period start time","LTE_call_setup_success_rate","Call Drop Rate","ERAB AbnormRel(times)","RRC Setup Success Rate","OFR_Inter X2 based HO Att E-UTRAN HO Attempts, inter eNB X2 based"])
df2=pd.DataFrame(kpi2)
df3=pd.DataFrame(kpi3)
k=df1[["ERAB AbnormRel(times)","OFR_Inter X2 based HO Att E-UTRAN HO Attempts, inter eNB X2 based"]]
e=df1[["Call Drop Rate","ERAB AbnormRel(times)"]]
imputer=imputer.fit(k)
k=imputer.transform(k)
imputer=imputer.fit(e)
e=imputer.transform(e)

coveriance_2=np.cov(e[:, 0], e[:, 1])
r = np.corrcoef(e[:, 0], e[:, 1])
print("covariance=",coveriance_2,"correlation coff.=",r[0,1])