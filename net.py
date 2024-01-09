# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import numpy as np

class seed_svdd(nn.Module):
    def __init__(self):
        super(seed_svdd, self).__init__()
        
        ##spatial feature extractor
        self.layer11 = nn.Sequential(
            nn.Conv2d(1, 16, 8, stride=4, padding=2),
            nn.BatchNorm2d(16),
            nn.ELU(inplace=False))
        
        self.layer22 = nn.Sequential(
            nn.Conv2d(16, 8, 4, stride=2, padding=1),
            nn.BatchNorm2d(8),
            nn.ELU(inplace=False))
        
        self.layer33 = nn.Sequential(
            nn.Conv2d(8, 4, 4, stride=2, padding=1),
            nn.BatchNorm2d(4),
            nn.ELU(inplace=False))
                       
        ## band attention module (BAM)
        self.BAM = nn.Sequential(
                nn.Linear(678, 512, bias=False),
                nn.ELU(),
                nn.Linear(512, 678, bias=False),
                nn.Sigmoid()
                )             
        
        self.fc = nn.Linear(4*4*4 + 4*84, 128)            
                                                                         
       ##spectral feature extractor
        self.layer1 = nn.Sequential(
            nn.Conv1d(1, 16, 4, stride=2, padding=1),
            nn.BatchNorm1d(16),
            nn.ELU())
        
        self.layer2 = nn.Sequential(
            nn.Conv1d(16, 8, 4, stride=2, padding=1),
            nn.BatchNorm1d(8),
            nn.ELU())
        
        self.layer3 = nn.Sequential(
            nn.Conv1d(8, 4, 4, stride=2, padding=1),
            nn.BatchNorm1d(4),
            nn.ELU())
                   
    def forward(self, x1, x2):
        
        ##spatial
        input_spa = x1   
        L11 = self.layer11(input_spa)
        L22 = self.layer22(L11)
        L33 = self.layer33(L22)
        
        f_spa = torch.flatten(L33, 1)
        
        ##spectral band attention                     
        input_spe = x2        
        input_spe = input_spe.view(input_spe.size(0), -1)          
        weight =self.BAM(input_spe)               
        weight = weight.view(x2.size(0), x2.size(1), x2.size(2))                        
        z = x2 * weight                
        
        L1 = self.layer1(z)
        L2 = self.layer2(L1)
        L3 = self.layer3(L2)
        
        f_spe = torch.flatten(L3, 1)                                 
        
        ##feature fusing
        f_new = torch.cat((f_spa, f_spe),1)
        output = self.fc(f_new)
                               
        return output
    
# if __name__=='__main__':
#     model = seed_svdd()
#     x1 = torch.randn(25,1,64,64)
#     x2 = torch.randn(25,1,678) 
#     a = model(x1, x2)
      
# print('size',a.shape)