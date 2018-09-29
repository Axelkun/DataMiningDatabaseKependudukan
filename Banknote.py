# -*- coding: utf-8 -*-
"""
Created on Mon Sep 24 23:55:14 2018

@author: Antraxiana
"""
import pandas as pd
import matplotlib.pyplot as plt
import math

h1 = [[0.3,0.2,-0.1,-0.4],
      [0.3,-0.1,0.4,0.2]
      ]

b1 = [0.2,0.2]

h2= [[0.3,0.5],
     [0.2,0.1],
     [0.3,0.1]
     ]

b2= [-0.2,-0.2,-0.2]

ol= [[0.2,-0.2,0.3]
     ]

ob=[0.1]

a=0.1


#Dataset
data = pd.read_csv("banknote.csv", header = None)
dataset=data.values.tolist()

#fungsi SGD
def fH(X, w, b):
    ans = []
    for j in range(0, len(w)):
        H=0
        for i in range(0,len(w[j])):
            H+=X[i]*w[j][i]
        H+= b[j]
        ans.append(H)
    return ans

def sigmoid(h): #fungsi sigmoid
    ans = []
    for i in range(0, len(h)):
        ans.append(1/(1+math.e**-h[i]))
    return ans

def error(dataset, sigo): #perhitungan nilai ERROR
    ans = []
    for i in range(0, len(sigo)):
        ans.append((dataset[4]-sigo[i])**2)
    return ans

def deltaO(fact, s): #fungsi perhitungan bobot dari output layer
    ans = []
    for i in range(0, len(s)):
        ans.append(2*(s[i]-fact[4])*(1-s[i])*s[i])
    return ans

def deltaH(weight,deltah, sigH): #fungsi perhitungan bobot dari hidden layer
    ans= []
    for j in range(0,len(sigH)):
        delta = 0
        for i in range(0,len(deltah)):
            delta+=(weight[i][j]*deltah[i])
        delta*=(1-sigH[j])*sigH[j]
        ans.append(delta)
    return ans


def NewWeight(delta,x,w): #fungsi pembaharuan bobot
    ans = []
    for j in range(0, len(delta)):
        listw = []
        for i in range(0, len(w[j])):
            dw=x[i]*delta[j]
            weight= w[j][i]-(a*dw)
            listw.append(weight)
        ans.append(listw)
    return ans

def NewBias(delta,bias): #fungsi mengbaharuan bias
    ans = []
    for i in range(0, len(bias)):
        ans.append(bias[i]-(a*delta[i]))
    return ans

def BP(i,dataset,h1,b1,h2,b2,ol,ob): #fungsi backpropagation menggunakan Sigmoid
    hH1 = fH(dataset[i],h1,b1)
    hs1 = sigmoid(hH1)
    hH2 = fH(hs1,h2,b2)
    hs2 = sigmoid(hH2)
    oH = fH(hs2,ol,ob)
    os = sigmoid(oH)
    
    deltao = deltaO(dataset[i], os);
    deltah2 = deltaH(ol, deltao , hs2);
    deltah1 = deltaH(h2, deltah2, hs1);
    
    err = error(dataset[i], os)
    if (len(err)==1):
        [err]=err
    newWO  = NewWeight(deltao , hs2, ol)
    newWH2 = NewWeight(deltah2, hs1, h2)
    newWH1 = NewWeight(deltah1, dataset[i],h1)
    newob = NewBias(deltao , ob)
    newb2 = NewBias(deltah2, b2)
    newb1 = NewBias(deltah1, b1)

    output = [newWH1,newb1,newWH2,newb2,newWO,newob,err]
    return output
    

def MLBP(ddataset,h1,b1,h2,b2,ol,ob): #SUBPROGRAM ML backpropagation dengan fungsi aktifasi sigmoid
    AvgErrTrain = []
    #dilakukan pembagian data agar tidak terjadi overfitting
    for j in range(0, 50):
        AvgErr = []
        for i in range(0, len(dataset)):
            output = BP(i,dataset,h1,b1,h2,b2,ol,ob)
            h1 = output[0]
            b1 = output[1]
            h2 = output[2]
            b2 = output[3]
            ol = output[4]
            ob = output[5]
            AvgErr.append(output[6])
        Terr = 0
        MLBP.a=AvgErr
        #Diukur rata-rata errornya
        for i in range(0, len(AvgErr)):
            Terr += AvgErr[i]
        AvgErrTrain.append(Terr / len(AvgErr))
        
    
    # memunculkan grafik
    plt.plot(AvgErrTrain, label = "train")
    plt.legend()
    plt.ylabel('Error')
    plt.xlabel('Epoch')
    plt.title('Sigmoid Error Graph')
    plt.show()

#=============================================================================
    
#Menjalankan fungsi program ML BP menggunakan fungsi aktifasi Sigmoid
MLBP(dataset,h1,b1,h2,b2,ol,ob) 
