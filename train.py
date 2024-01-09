import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from net import seed_svdd
from dataset import BatchData
from tqdm import tqdm
import scipy.io as io
import torch.optim as optim


# Take variety A as an example
## load spatial data  
var1_spa = np.load('./data/Train_spa/A_train_spa.npy')
var2_spa = np.load('./data/Test_spa/A_test_spa.npy')
print (var1_spa.shape)

# transpose image channel position
var1_spa=var1_spa.transpose(0,3,1,2)
var2_spa=var2_spa.transpose(0,3,1,2)

var1_spe = np.load('./data/Train_spe/A_train.npy')
var2_spe = np.load('./data/Test_spe/A_test.npy')

train_loader = DataLoader(BatchData(var1_spa,var1_spe),
                          batch_size = 50,
                          shuffle = True)
print(var1_spe.shape)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = seed_svdd().to(device)
optimizer=torch.optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.999), weight_decay=1e-6)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.1)
epoch_n = 800
eps = 0.1

class DeepSVDDTrainer():
        
    def train(self):
        
        model.train()      
        for epoch in range(epoch_n):
        
            losses = []
            print("Epoch {}/{}".format(epoch+1,epoch_n))
            print("-"*10)
            # start training                  
            for i, (spa, spe) in enumerate(tqdm(train_loader)):
                                
                # spatial inputs
                inputs_spa = spa            
                inputs_spa = inputs_spa.to(device)               
                inputs_spa = inputs_spa.type(torch.cuda.FloatTensor)
                
                # spectral inputs         
                inputs_spe = spe                                                  
                inputs_spe=inputs_spe.to(device)                    
                inputs_spe = inputs_spe.type(torch.cuda.FloatTensor)
               
                # output
                output = model(inputs_spa, inputs_spe)                                                                                                          
                c, _ = self.init_center_c(spa, spe, model)                                                
                dist=torch.sum((output-c)**2, dim=1)                          
                loss_train = torch.mean(dist)
                                                                                              
                # BP
                optimizer.zero_grad()
                loss_train.backward()
                optimizer.step()                
                losses.append(loss_train.item())
                               
            print("train loss :", np.mean(losses))

        return model
                    
    def init_center_c(self, train_data1, train_data2 = train_loader, net = model, eps=eps):

            n_samples = 0            
            c = torch.zeros([128]).cuda()         
            model.eval()
                        
            with torch.no_grad():
                                    
                    data_spa = train_data1
                    data_spe = train_data2                    
                                        
                    ## get the inputs of the batch   
                    # spatial data
                    data_spa = data_spa.to(device)               
                    data_spa = data_spa.type(torch.cuda.FloatTensor)
                   
                    # spectra data        
                    data_spe = data_spe            
                    data_spe = data_spe.to(device)                    
                    data_spe = data_spe.type(torch.cuda.FloatTensor)
                    
                    # feature
                    feature = model(data_spa, data_spe)
                    
                    n_samples += feature.shape[0]                                             
                    c += torch.sum(feature, dim=0)                    
            
            c /= n_samples
        
            # If c_i is too close to 0, set to +-eps. Reason: a zero unit can be trivially matched with zero weights.
            c[(abs(c) < eps) & (c < 0)] = -eps
            c[(abs(c) < eps) & (c > 0)] = eps            
                                
            return c, feature
    
main = DeepSVDDTrainer()    
model = main.train()
torch.save(model, './trained_model/model_A.pkl')

## The Center of all samples
# spatial data
inputs_spa1 = torch.tensor(var1_spa)
# spectral data         
inputs_spe1 = torch.tensor(var1_spe) 
c, feature = main.init_center_c(inputs_spa1, inputs_spe1)

print('C.shape & C:', c.shape, c)
print('feature.shape & feature:', feature.shape, feature)

## Define R (95th percentile)
Rc= (feature - c)**2 
Rc=torch.mean(Rc, dim = 1)                                     
Rc=Rc.cpu().detach().numpy()
Rm=np.percentile(Rc, 95)
print(Rm)

## Save R and C
save_path1 = './Radius and Center/R_A.npy' 
save_path2 = './Radius and Center/C_A.npy' 
np.save(save_path1, Rm)
np.save(save_path2, c.cpu().detach().numpy())

## Test: receive acc
inputs_spa2 = torch.tensor(var2_spa)
inputs_spa2 = inputs_spa2.to(device)               
inputs_spa2 = inputs_spa2.type(torch.cuda.FloatTensor)

inputs_spe2 = torch.tensor(var2_spe) 
inputs_spe2 = inputs_spe2.to(device)               
inputs_spe2 = inputs_spe2.type(torch.cuda.FloatTensor)

feature_test  = model(inputs_spa2, inputs_spe2)                    
dist_test = (feature_test - c)**2
dist_test=torch.mean(dist_test, dim = 1)
 
print('R_test:', dist_test)
print('R_test_shape:', dist_test.shape)
print('R_m:', Rm)                                     

dist_test[(abs(dist_test) < Rm) ] = 0 # receive
dist_test[(abs(dist_test) > Rm) ] = 1 # reject
Score = dist_test
print("Score:", Score)
zeros  = sum(Score == 0).item()
ones = sum(Score == 1).item()

acc_receive = zeros/(ones+zeros)
print('acc_receive:',acc_receive)