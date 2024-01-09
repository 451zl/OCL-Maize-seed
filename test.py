# -*- coding: utf-8 -*-
import torch
import numpy as np
from torch.utils.data import Dataset,DataLoader
from net import seed_svdd

##
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

### Take A variety as an example
R_A = np.load('./Radius and Center/R_A.npy')
C_A = np.load('./Radius and Center/C_A.npy')

R_A= torch.tensor(R_A)
R_A = R_A.to(device)               
R_A = R_A.type(torch.cuda.FloatTensor)

C_A= torch.tensor(C_A)
C_A = C_A.to(device)               
C_A = C_A.type(torch.cuda.FloatTensor)

## load spatial data
## A 
var_spa1 = np.load('./data/Test_spa/A_test_spa.npy')
# transpose image channel position
var_spa1 =var_spa1.transpose(0,3,1,2)

## Non A (i.e., B,C, D,..., T) (other 19 varieties)
var_spa2 = np.load('./data/Test_spa/Non_A_test_spa.npy')
# transpose image channel position
var_spa2 =var_spa2.transpose(0,3,1,2)

## load spectral data
## A 
var_spe1 = np.load('./data/Test_spe/A_test.npy')
## Non A (i.e., B,C, D,..., T) (other 19 varieties)
var_spe2 = np.load('./data/Test_spe/Non_A_test.npy')
## load model                   
model_A=torch.load('./trained_model/model_A.pkl')
model_A = model_A.to(device)

## Test Receive Acc
inputs_spa1 = torch.tensor(var_spa1)
inputs_spa1 = inputs_spa1.to(device)               
inputs_spa1 = inputs_spa1.type(torch.cuda.FloatTensor)

inputs_spe1 = torch.tensor(var_spe2) 
inputs_spe1 = inputs_spe1.to(device)               
inputs_spe1 = inputs_spe1.type(torch.cuda.FloatTensor)

feature_test1  = model_A(inputs_spa1, inputs_spe1)                    
dist_test1 = (feature_test1 - C_A)**2
dist_test1 = torch.mean(dist_test1, dim = 1)

dist_test1[(abs(dist_test1) < R_A) ] = 0 # receive
dist_test1[(abs(dist_test1) > R_A) ] = 1 # reject

Score1 = dist_test1
print("Score1:", Score1)
zeros_1  = sum(Score1 == 0).item()
ones_1 = sum(Score1 == 1).item()

acc_receive = zeros_1/(ones_1+zeros_1)
print('acc_receive:', acc_receive)

## Test Reject Acc
inputs_spa2 = torch.tensor(var_spa2)
inputs_spa2 = inputs_spa2.to(device)               
inputs_spa2 = inputs_spa2.type(torch.cuda.FloatTensor)

inputs_spe2 = torch.tensor(var_spe2) 
inputs_spe2 = inputs_spe2.to(device)               
inputs_spe2 = inputs_spe2.type(torch.cuda.FloatTensor)

feature_test2  = model_A(inputs_spa2, inputs_spe2)                    
dist_test2 = (feature_test2 - C_A)**2
dist_test2 = torch.mean(dist_test2, dim = 1)


dist_test2[(abs(dist_test2) < R_A) ] = 0 # receive
dist_test2[(abs(dist_test2) > R_A) ] = 1 # reject

Score2 = dist_test2
print("Score2:", Score2)
zeros_2  = sum(Score2 == 0).item()
ones_2 = sum(Score2 == 1).item()

acc_reject = ones_2/(ones_2 + zeros_2)
print('acc_reject:', acc_reject)