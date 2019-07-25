@author: Asim
'''
from graph2 import Gs,cbdduckids,dates,wdkduckids,cbdhpids
from rest_fea import duckfeature, hpfeature
from ReadY import duckY,HPY
import numpy as np

graph_feas_train=[]

rest_feas_train=[]

Ys_train=[]

graph_feas_test=[]

rest_feas_test=[]

Ys_test=[]

for rest in cbdduckids:
    for date in dates[0:-14]:
        graph_feas_train.append(Gs[(rest,date,'1')])
        rest_feas_train.append(list(map(float,duckfeature[(rest,'1')])))
        Ys_train.append([float(duckY[(rest,date,'1')])])
    for date in dates[-14:]:
        graph_feas_test.append(Gs[(rest,date,'1')])
        rest_feas_test.append(list(map(float,duckfeature[(rest,'1')])))
        Ys_test.append([float(duckY[(rest,date,'1')])])


for rest in cbdhpids:
    for date in dates:
        graph_feas_train.append(Gs[(rest,date,'2')])
        rest_feas_train.append(list(map(float,hpfeature[(rest,'1')])))
        Ys_train.append([float(HPY[(rest,date,'1')])])
    for date in dates[-14:]:
        graph_feas_test.append(Gs[(rest,date,'2')])
        rest_feas_test.append(list(map(float,hpfeature[(rest,'1')])))
        Ys_test.append([float(HPY[(rest,date,'1')])])        
