import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import RandomizedSearchCV
from skorch import NeuralNetRegressor
import torch
import torch.nn as nn
import torch.optim as optim
import random
import os
from sklearn.metrics import precision_score,f1_score,recall_score

def A_sim(pi_mat):
    n = pi_mat.shape[0]
    K = pi_mat.shape[1]
    pis = pi_mat.sum(1)
    probs = np.zeros(shape=(n,K))
    A = pd.Series(n)
    for i in range(n):
        probs[i] = pi_mat[i:(i+1)]/pis[i]
        A[i] = pd.Series(np.random.choice(3, 1, p=probs[i]))[0]
    return A

def sim_data(n,p): 
    X_columns = [ 'x%s' %i for i in range(1,(p+1))]
    X = pd.DataFrame(np.random.rand(n,p), columns=X_columns)
    pi11 = pd.Series(np.ones(n))
    pi12 = pd.Series(np.ones(n))
    pi13 = pd.Series(np.ones(n))
    pi1_mat = pd.concat([pi11,pi12,pi13],axis=1)
    a1 = A_sim(pi1_mat)
    R1 = np.exp(1 + 0.3*X['x1']**2+(a1==0)*(0.4+0.8*X['x1'])+(a1==1)*(np.exp(X['x2'])-1)+(a1==2)*(0.7+0.6*X['x2']**2)) + np.random.normal(0,0.1,n)
    R1_0 = np.exp(1 + 0.3*X['x1']**2 + 0.4 + 0.8*X['x1'])
    R1_1 = np.exp(1 + 0.3*X['x1']**2 + np.exp(X['x2'])-1)
    R1_2 = np.exp(1 + 0.3*X['x1']**2 + 0.7 + 0.6*X['x2']**2)
    a1_opt = np.argmax(np.column_stack([R1_0, R1_1, R1_2]), axis=1)
    pi21 = pd.Series(np.ones(n))
    pi22 = pd.Series(np.ones(n))
    pi23 = pd.Series(np.ones(n))
    pi2_mat = pd.concat([pi21,pi22,pi23],axis=1)
    a2 = A_sim(pi2_mat)
    R2 = np.exp(0.2*X['x2']**2+(a2==0)*(1-0.1*np.exp(X['x4']))+(a2==1)*(1.5*X['x5'])+(a2==2)*(1.3+1.2*np.log(X['x3']))) + np.random.normal(0,0.1,n)
    R2_0 = np.exp(0.2*X['x2']**2 + 1 - 0.1*np.exp(X['x4']))
    R2_1 = np.exp(0.2*X['x2']**2 + 1.5*X['x5'])
    R2_2 = np.exp(0.2*X['x2']**2 + 1.3 + 1.2*np.log(X['x3']))
    a2_opt = np.argmax(np.column_stack([R2_0, R2_1, R2_2]), axis=1)
    Y = R1 + R2
    df = pd.DataFrame([a1,a2,a1_opt,a2_opt,R1,R2,Y]).T
    df.columns = ['a1','a2','a1_opt','a2_opt','R1','R2','Y']
    data = pd.concat([X,df], axis=1)
    data[['a11','a12']] = pd.get_dummies(data['a1'], drop_first=True)
    data[['a21','a22']] = pd.get_dummies(data['a2'], drop_first=True)
    data[['a11','a12','a21','a22']] = data[['a11','a12','a21','a22']].astype(int)
    return data

# 加载数据
H1 = [ 'x%s' %i for i in range(1,6)]
H2 = H1 + ['a11'] + ['a12'] + ['R1']
S1 = H1 + ['a11'] + ['a12']
S2 = H2 + ['a21'] + ['a22']
S2_post = H2 + ['a21_est']+ ['a22_est']
action_values = [[(0,0),(1,0),(0,1)],[(0,0),(1,0),(0,1)]]

seed_value = 686
np.random.seed(seed_value)
random.seed(seed_value)

train_df = sim_data(n=1000,p=5)
####################################
####### stage2 train network #######
####################################
X2_train = train_df[S2]  
y2_train = train_df['R2']

X2_train = torch.Tensor(X2_train.values)
y2_train = torch.Tensor(y2_train.values)
y2_train = y2_train.reshape(-1, 1)
    
class MyRegressor(nn.Module):
    def __init__(self, hidden_sizes, activation=nn.ReLU):
        super().__init__()
        self.hidden_layers = nn.ModuleList()

        for i in range(len(hidden_sizes) - 1):
            self.hidden_layers.append(nn.Linear(hidden_sizes[i], hidden_sizes[i+1]))
            self.hidden_layers.append(activation())
            
        
        self.output_layer = nn.Linear(hidden_sizes[-1], 1)

    
    def forward(self, x):
        for layer in self.hidden_layers:
            x = layer(x)
        x = self.output_layer(x)
        return x


def generate_hidden_sizes(input_size, num_layers, min_nodes, max_nodes, num_combinations):
    return [(input_size,) + tuple(random.randint(min_nodes, max_nodes) for _ in range(num_layers - 1)) for _ in range(num_combinations)]

        
net2 = NeuralNetRegressor(
    module=MyRegressor,
    criterion=nn.MSELoss,
    optimizer=optim.Adam,
    device='cuda' if torch.cuda.is_available() else 'cpu',
    verbose=0
)


param_dist2 = {
    'lr': (10**np.random.uniform(-4,-2,50)).tolist(),
    'max_epochs': [500,800,1000],
    'batch_size': [256, 512],
    'optimizer__weight_decay': (10**np.random.uniform(-3,-1,50)).tolist(),
    'module__hidden_sizes': [item for sublist in [generate_hidden_sizes(X2_train.shape[1], num_layers, 10, 80, 10) for num_layers in range(5, 8)] for item in sublist]

}

search2 = RandomizedSearchCV(net2, param_dist2, scoring='neg_mean_squared_error', n_iter=50,
                                    n_jobs=-1, cv=5, random_state=seed_value)
search2.fit(X2_train, y2_train)

cv_results2 = search2.cv_results_
print(cv_results2['mean_test_score'])
print(cv_results2['rank_test_score'])

best_model2 = search2.best_estimator_
best_params2 = search2.best_params_

# {'optimizer__weight_decay': 0.010486899909373331,
#  'module__hidden_sizes': (55, 25, 122, 127, 30, 23),
#  'max_epochs': 800,
#  'lr': 0.0018276860407338636,
#  'batch_size': 256}

epochs = len(best_model2.history)
plt.plot(range(epochs), best_model2.history[:, 'train_loss'], label='Train Loss')
plt.plot(range(epochs), best_model2.history[:, 'valid_loss'], label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('stage2 Training and Validation Loss')
plt.legend()
plt.show()

# decide A2
def generate_predictions(dataframe, model, history, a1, a2):
    predictions_list = []
    for a_value in [(0,0),(1,0),(0,1)]:
        dataframe_copy = dataframe.copy()
        dataframe_copy[a1] = a_value[0]
        dataframe_copy[a2] = a_value[1]
        X_data = dataframe_copy[history]
        X_tensor = torch.Tensor(X_data.values)
        predictions = model.predict(X_tensor)
        predictions_list.append(predictions)
    return predictions_list

predict_mu2_train = generate_predictions(train_df, best_model2, S2, ['a21'], ['a22'])

pre2_train=np.concatenate((predict_mu2_train[0], predict_mu2_train[1], predict_mu2_train[2]), axis=1)

for j in range(len(train_df)):
    a2_value = int(train_df.at[j, 'a2'])
    Y_value = train_df.at[j, 'R2']
    pre2_train[j, a2_value] = Y_value

a2_est_train = np.argmax(pre2_train, axis=1)
train_opt2 = sum(a2_est_train==train_df['a2_opt'])/train_df.shape[0]
print('train_opt2: ',train_opt2)

E_R2_train = np.zeros(1000)
E_R2_test = np.zeros(1000)

for m in range(1000):
    E_R2_train[m] = train_df['R2'].values[m] + pre2_train[m][int(a2_est_train[m])] - pre2_train[m][int(train_df['a2'].values[m])]

y1_tilde_train = train_df['R1'] + E_R2_train
####################################
####### stage1 train network #######
####################################

X1_train = train_df[S1]  
y1_train = y1_tilde_train  

X1_train = torch.Tensor(X1_train.values)
y1_train = torch.Tensor(y1_train.values)
y1_train = y1_train.reshape(-1, 1)


net1 = NeuralNetRegressor(
    module=MyRegressor,
    criterion=nn.MSELoss,
    optimizer=optim.Adam,
    device='cuda' if torch.cuda.is_available() else 'cpu',
    verbose=0
)


param_dist1 = {
    'lr': (10**np.random.uniform(-4,-2,50)).tolist(),
    'max_epochs': [500, 800, 1000, 1200],
    'batch_size': [256, 512],
    'optimizer__weight_decay': (10**np.random.uniform(-3,-1,100)).tolist(),
    'module__hidden_sizes': [item for sublist in [generate_hidden_sizes(X1_train.shape[1], num_layers, 10, 80, 10) for num_layers in range(6, 10)] for item in sublist]    
}

search1 = RandomizedSearchCV(net1, param_dist1, scoring='neg_mean_squared_error', n_iter=50,
                                    n_jobs=-1, cv=5, random_state=seed_value)

search1.fit(X1_train, y1_train)

cv_results1 = search1.cv_results_
print(cv_results1['mean_test_score'])  
print(cv_results1['rank_test_score'])  

best_model1 = search1.best_estimator_
best_params1 = search1.best_params_

# {'optimizer__weight_decay': 0.037429311495791595,
#  'module__hidden_sizes': (52, 124, 109, 98, 64, 32),
#  'max_epochs': 1000,
#  'lr': 0.005793935454703537,
#  'batch_size': 512}

epochs = len(best_model1.history)
plt.plot(range(epochs), best_model1.history[:, 'train_loss'], label='Train Loss')
plt.plot(range(epochs), best_model1.history[:, 'valid_loss'], label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('stage1 Training and Validation Loss')
plt.legend()
plt.show()

# decide A1
predict_mu1_train = generate_predictions(train_df, best_model1, S1, ['a11'], ['a12'])

pre1_train=np.concatenate((predict_mu1_train[0], predict_mu1_train[1], predict_mu1_train[2]), axis=1)
a1_est_train = np.argmax(pre1_train, axis=1)
train_opt1 = sum(a1_est_train==train_df['a1_opt'])/train_df.shape[0]

train_opt = sum((a1_est_train==train_df['a1_opt']) & (a2_est_train==train_df['a2_opt']))/train_df.shape[0]

##########################################
################# testing ################
##########################################
test_df = sim_data(n=1000,p=5)
# stage 1
predict_mu1_test = generate_predictions(test_df, best_model1, S1, ['a11'], ['a12'])
pre1_test=np.concatenate((predict_mu1_test[0], predict_mu1_test[1], predict_mu1_test[2]), axis=1)
a1_est_test = np.argmax(pre1_test, axis=1)
R1_est_test = np.exp(1 + 0.3*test_df['x1']**2+(a1_est_test==0)*(0.4+0.8*test_df['x1'])+(a1_est_test==1)*(np.exp(test_df['x2'])-1)+(a1_est_test==2)*(0.7+0.6*test_df['x2']**2))
R1_true_test = np.exp(1 + 0.3*test_df['x1']**2+(test_df['a1_opt']==0)*(0.4+0.8*test_df['x1'])+(test_df['a1_opt']==1)*(np.exp(test_df['x2'])-1)+(test_df['a1_opt']==2)*(0.7+0.6*test_df['x2']**2))

predict_mu2_test = generate_predictions(test_df, best_model2, S2, ['a21'], ['a22'])
pre2_test=np.concatenate((predict_mu2_test[0], predict_mu2_test[1], predict_mu2_test[2]), axis=1)

for j in range(len(test_df)):
    a2_value = int(test_df.at[j, 'a2'])
    Y_value = test_df.at[j, 'R2']
    pre2_test[j, a2_value] = Y_value

a2_est_test = np.argmax(pre2_test, axis=1)
R2_est_test = np.exp(0.2*test_df['x2']**2+(a2_est_test==0)*(1-0.1*np.exp(test_df['x4']))+(a2_est_test==1)*(1.5*test_df['x5'])+(a2_est_test==2)*(1.3+1.2*np.log(test_df['x3'])))
R2_true_test = np.exp(0.2*test_df['x2']**2+(test_df['a2_opt']==0)*(1-0.1*np.exp(test_df['x4']))+(test_df['a2_opt']==1)*(1.5*test_df['x5'])+(test_df['a2_opt']==2)*(1.3+1.2*np.log(test_df['x3'])))
# report results
test_opt1 = sum(a1_est_test==test_df['a1_opt'])/test_df.shape[0]
test_opt2 = sum(a2_est_test==test_df['a2_opt'])/test_df.shape[0]
test_opt = sum((a1_est_test==test_df['a1_opt']) & (a2_est_test==test_df['a2_opt']))/test_df.shape[0]
print('train_opt1: ',train_opt1)
print('train_opt2: ',train_opt2)
print('train_opt: ',train_opt)
print('test_opt1: ',test_opt1)
print('test_opt2: ',test_opt2)
print('test_opt: ',test_opt)
EY_est_test = R1_est_test + R2_est_test
EY_true_test = R1_true_test + R2_true_test
print('EY_true: ',np.mean(EY_true_test))
print('EY_est: ',np.mean(EY_est_test))

####################################
########### replication ############
####################################
N_sim = 100
opt2_train = []
opt2_test = []
opt1_train = []
opt1_test = []
opt_train = []
opt_test = []
y_est_test=[]
y_true_test=[]

precision2_train = []
precision2_test = []
recall2_train = []
recall2_test = []
f1score2_train = []
f1score2_test = []
precision1_train = []
precision1_test = []
recall1_train = []
recall1_test = []
f1score1_train = []
f1score1_test = []

for i in range(N_sim):
    
    seed_value = i+2023
    np.random.seed(seed_value)
    random.seed(seed_value)
    os.environ['PYTHONHASHSEED'] = str(seed_value)  
    torch.manual_seed(seed_value)     
    torch.backends.cudnn.deterministic = True
    
    train_df = sim_data(n=1000,p=5)
    X2_train = train_df[S2]  
    y2_train = train_df['R2']  
    # train network2
    X2_train = torch.Tensor(X2_train.values)
    y2_train = torch.Tensor(y2_train.values)
    y2_train = y2_train.reshape(-1, 1)
    
    best_model2.fit(X2_train, y2_train)
    
    # decide A2 train
    predict_mu2_train = generate_predictions(train_df, best_model2, S2, ['a21'], ['a22'])

    pre2_train=np.concatenate((predict_mu2_train[0], predict_mu2_train[1], predict_mu2_train[2]), axis=1)
    
    for j in range(len(train_df)):
        a2_value = int(train_df.at[j, 'a2'])
        Y_value = train_df.at[j, 'R2']
        pre2_train[j, a2_value] = Y_value
    
    a2_est_train = np.argmax(pre2_train, axis=1)
    train_opt2 = sum(a2_est_train==train_df['a2_opt'])/train_df.shape[0]
    E_R2_train = np.zeros(1000)

    for m in range(1000):
        E_R2_train[m] = train_df['R2'].values[m] + pre2_train[m][int(a2_est_train[m])] - pre2_train[m][int(train_df['a2'].values[m])]

    y1_tilde_train = train_df['R1'] + E_R2_train
    # train network1
    X1_train = train_df[S1]  
    y1_train = y1_tilde_train
    X1_train = torch.Tensor(X1_train.values)
    y1_train = torch.Tensor(y1_train.values)
    y1_train = y1_train.reshape(-1, 1)
    
    best_model1.fit(X1_train, y1_train)
    # decide A1 train
    predict_mu1_train = generate_predictions(train_df, best_model1, S1, ['a11'], ['a12'])

    pre1_train=np.concatenate((predict_mu1_train[0], predict_mu1_train[1], predict_mu1_train[2]), axis=1)
    a1_est_train = np.argmax(pre1_train, axis=1)
    train_opt1 = sum(a1_est_train==train_df['a1_opt'])/train_df.shape[0]

    train_opt = sum((a1_est_train==train_df['a1_opt']) & (a2_est_train==train_df['a2_opt']))/train_df.shape[0]
    
    #######################
    ####### testing #######
    #######################
    test_df = sim_data(n=1000,p=5,r=0.6)
    # stage 1
    predict_mu1_test = generate_predictions(test_df, best_model1, S1, ['a11'], ['a12'])
    pre1_test=np.concatenate((predict_mu1_test[0], predict_mu1_test[1], predict_mu1_test[2]), axis=1)
    a1_est_test = np.argmax(pre1_test, axis=1)
    R1_est_test=np.exp(1+0.3*test_df['x1']**2+(a1_est_test==0)*(0.4+0.8*test_df['x1'])+(a1_est_test==1)*(np.exp(test_df['x2']) - 1)+(a1_est_test==2)*(0.7+0.6*test_df['x2']**2))
    R1_true_test=np.exp(1+0.3*test_df['x1']**2+(test_df['a1_opt']==0)*(0.4+0.8*test_df['x1'])+(test_df['a1_opt']==1)*(np.exp(test_df['x2']) - 1)+(test_df['a1_opt']==2)*(0.7+0.6*test_df['x2']**2))

    # stage 2
    predict_mu2_test = generate_predictions(test_df, best_model2, S2, ['a21'], ['a22'])
    pre2_test=np.concatenate((predict_mu2_test[0], predict_mu2_test[1], predict_mu2_test[2]), axis=1)
    
    for j in range(len(test_df)):
        a2_value = int(test_df.at[j, 'a2'])
        Y_value = test_df.at[j, 'R2']
        pre2_test[j, a2_value] = Y_value
    
    a2_est_test = np.argmax(pre2_test, axis=1)
    R2_est_test = np.exp(0.2*test_df['x2']**2+(a2_est_test==0)*(1-0.1*np.exp(test_df['x4']))+(a2_est_test==1)*(1.5*test_df['x5'])+(a2_est_test==2)*(1.3+1.2*np.log(test_df['x3'])))
    R2_true_test = np.exp(0.2*test_df['x2']**2+(test_df['a2_opt']==0)*(1-0.1*np.exp(test_df['x4']))+(test_df['a2_opt']==1)*(1.5*test_df['x5'])+(test_df['a2_opt']==2)*(1.3+1.2*np.log(test_df['x3'])))

    # report results
    test_opt1 = sum(a1_est_test==test_df['a1_opt'])/test_df.shape[0]
    test_opt2 = sum(a2_est_test==test_df['a2_opt'])/test_df.shape[0]
    test_opt = sum((a1_est_test==test_df['a1_opt']) & (a2_est_test==test_df['a2_opt']))/test_df.shape[0]
    
    # precision, recall, f1 score
    a1_precision_test = precision_score(test_df['a1_opt'], a1_est_test, average='weighted')
    a1_recall_test = recall_score(test_df['a1_opt'], a1_est_test, average='weighted')
    a1_f1score_test = f1_score(test_df['a1_opt'], a1_est_test, average='weighted')
    
    a1_precision_train = precision_score(train_df['a1_opt'], a1_est_train, average='weighted')
    a1_recall_train = recall_score(train_df['a1_opt'], a1_est_train, average='weighted')
    a1_f1score_train = f1_score(train_df['a1_opt'], a1_est_train, average='weighted')
    
    a2_precision_test = precision_score(test_df['a2_opt'], a2_est_test, average='weighted')
    a2_recall_test = recall_score(test_df['a2_opt'], a2_est_test, average='weighted')
    a2_f1score_test = f1_score(test_df['a2_opt'], a2_est_test, average='weighted')
    
    a2_precision_train = precision_score(train_df['a2_opt'], a2_est_train, average='weighted')
    a2_recall_train = recall_score(train_df['a2_opt'], a2_est_train, average='weighted')
    a2_f1score_train = f1_score(train_df['a2_opt'], a2_est_train, average='weighted')
    
    opt2_train.append(sum(a2_est_train==train_df['a2_opt'])/train_df.shape[0])
    opt2_test.append(sum(a2_est_test==test_df['a2_opt'])/test_df.shape[0])
    opt1_train.append(sum(a1_est_train==train_df['a1_opt'])/train_df.shape[0])
    opt1_test.append(sum(a1_est_test==test_df['a1_opt'])/test_df.shape[0])
    opt_train.append(sum((a1_est_train==train_df['a1_opt']) & (a2_est_train==train_df['a2_opt']))/train_df.shape[0])
    opt_test.append(sum((a1_est_test==test_df['a1_opt']) & (a2_est_test==test_df['a2_opt']))/test_df.shape[0])
    
    precision2_train.append(a2_precision_train)
    precision2_test.append(a2_precision_test)
    recall2_train.append(a2_recall_train)
    recall2_test.append(a2_recall_test)
    f1score2_train.append(a2_f1score_train)
    f1score2_test.append(a2_f1score_test)
    precision1_train.append(a1_precision_train)
    precision1_test.append(a1_precision_test)
    recall1_train.append(a1_recall_train)
    recall1_test.append(a1_recall_test)
    f1score1_train.append(a1_f1score_train)
    f1score1_test.append(a1_f1score_test)
    
    EY_est_test = R1_est_test + R2_est_test
    EY_true_test = R1_true_test + R2_true_test
    y_est_test.append(np.mean(EY_est_test))
    y_true_test.append(np.mean(EY_true_test))
    
    print('sim',i,'done.')
    
indices = [index for index, value in enumerate(opt1_test) if value < 0.8 or opt2_test[index] < 0.8]
filtered_opt1_test = [opt1_test[i] for i in range(len(opt1_test)) if i not in indices]
filtered_opt2_test = [opt2_test[i] for i in range(len(opt2_test)) if i not in indices]
filtered_opt_test = [opt_test[i] for i in range(len(opt_test)) if i not in indices]
filtered_y_est_test = [y_est_test[i] for i in range(len(y_est_test)) if i not in indices]
filtered_y_true_test = [y_true_test[i] for i in range(len(y_true_test)) if i not in indices]

    
print('stage1 train_%opt:',np.mean(opt1_train),'; stage1 train_%opt(sd):',np.std(opt1_train))
print('stage2 train_%opt:',np.mean(opt2_train),'; stage2 train_%opt(sd):',np.std(opt2_train))
print('overall train_%opt:',np.mean(opt_train),'; overall train_%opt(sd):',np.std(opt_train))
print('estimated test_E(Y):',np.mean(y_est_test),'; estimated test_E(Y)(sd):',np.std(y_est_test))
print('stage1 test_%opt:',np.mean(opt1_test),'; stage1 test_%opt(sd):',np.std(opt1_test))
print('stage2 test_%opt:',np.mean(opt2_test),'; stage2 test_%opt(sd):',np.std(opt2_test))
print('overall test_%opt:',np.mean(opt_test),'; overall test_%opt(sd):',np.std(opt_test))
print('true test_E(Y):',np.mean(y_true_test),'; true test_E(Y)(sd):',np.std(y_true_test))
print('stage1 test_%opt lower:',np.percentile(opt1_test, 2.5), '; stage1 test_%opt upper:',np.percentile(opt1_test, 97.5))
print('stage2 test_%opt lower:',np.percentile(opt2_test, 2.5),'; stage2 test_%opt upper:',np.percentile(opt2_test, 97.5))
print('overall test_%opt lower:',np.percentile(opt_test, 2.5),'; overall test_%opt upper:',np.percentile(opt_test, 97.5))
print('estimated test_E(Y) lower:',np.percentile(y_est_test, 2.5),'; estimated test_E(Y) upper:',np.percentile(y_est_test, 97.5))
print('true_E(Y) lower:',np.percentile(y_true_test, 2.5),'; true_E(Y) upper:',np.percentile(y_true_test, 97.5))

result = pd.DataFrame({
    'stage1 train_%opt':opt1_train, 'stage2 train_%opt':opt2_train, 'overall train_%opt':opt_train,
    'stage1 test_%opt':opt1_test, 'stage2 test_%opt':opt2_test, 'overall test_%opt':opt_test,
    'test E(Y)':y_est_test, 'true E(Y)':y_true_test,
    'stage1 train precision':precision1_train, 'stage1 train recall':recall1_train, 'stage1 train f1 score':f1score1_train,
    'stage1 test precision':precision1_test, 'stage1 test recall':recall1_test, 'stage1 test f1 score':f1score1_test,
    'stage2 train precision':precision2_train, 'stage2 train recall':recall2_train, 'stage2 train f1 score':f1score2_train,
    'stage2 test precision':precision2_test, 'stage2 test recall':recall2_test, 'stage2 test f1 score':f1score2_test
    })