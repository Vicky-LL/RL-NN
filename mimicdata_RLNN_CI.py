import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import RandomizedSearchCV
from skorch import NeuralNetRegressor
import torch
import torch.nn as nn
import torch.optim as optim
import random

df = pd.read_csv('data.csv')
data = df.copy(deep=True)

data['gender'] = data['gender'].apply(lambda x: 1 if x=='F' else 0)
data['ethnicity'] = data['ethnicity'].apply(lambda x: 1 if x == 'WHITE' else 0)
data['insurance'] = data['insurance'].apply(lambda x: 1 if x == 'Medicare' or x == 'Medicaid' or x == 'Government' else 0)
data['religion'] = data['religion'].apply(lambda x: 1 if x=="CATHOLIC" or x=="PROTESTANT QUAKER" or x=="EPISCOPALIAN" 
                                          or x=="CHRISTIAN SCIENTIST" or x=="ROMANIAN EAST. ORTH" or x=="GREEK ORTHODOX"
                                          or x=="JEHOVAH'S WITNESS" or x=="7TH DAY ADVENTIST" else 0)

data.loc[(data.fluid1 >= 30) ,'a1'] = 2
data.loc[(data.fluid1 >= 20) & (data.fluid1 < 30) ,'a1'] = 1
data.loc[(data.fluid1 < 20), 'a1'] = 0

data.loc[(data.fluid2 >= 30) ,'a2'] = 2
data.loc[(data.fluid2 >= 20) & (data.fluid2 < 30) ,'a2'] = 1
data.loc[(data.fluid2 < 20), 'a2'] = 0


data[['a11','a12']] = pd.get_dummies(data['a1'], drop_first=True) # dummy variables
data[['a21','a22']] = pd.get_dummies(data['a2'], drop_first=True)
data[['a11','a12','a21','a22']] = data[['a11','a12','a21','a22']].astype(int)

# sofa
plt.hist(data['sofa'], density=True, histtype='stepfilled', bins=10)
plt.show() 

data['Y'] = np.exp((25 - data['sofa'])/17)
plt.hist(data['Y'], density=True, histtype='stepfilled', bins=10)
plt.show()

# network structure
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

# notation
H1 = ['age','weight','los','gender','ethnicity','insurance','religion']
H2 = H1 + ['a11'] + ['a12'] + ['vent2','vaso2']

S1 = H1 + ['a11'] + ['a12']
S2 = H2 + ['a21'] + ['a22']
action_values = [[(0,0),(1,0),(0,1)],[(0,0),(1,0),(0,1)]]

seed_value = 686
np.random.seed(seed_value)
random.seed(seed_value)
torch.manual_seed(seed_value)
torch.cuda.manual_seed(seed_value)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# bootstrap samples
np.random.seed(42)

def bootstrap_resampling(data, n_resamples=100):
    bootstrap_samples = [data.sample(frac=1, replace=True) for _ in range(n_resamples)]
    return bootstrap_samples



N_boot = 100
boot_samples = bootstrap_resampling(data, n_resamples=N_boot)
g2_est = []
g1_est = []
mean_SOFA_est = []

for i in range(N_boot):
    data = boot_samples[i]
    # stage2 estimate
    X2 = data[S2]  
    y2 = data['Y']

    X2 = torch.Tensor(X2.values)
    y2 = torch.Tensor(y2.values)
    y2 = y2.reshape(-1, 1)

    net2 = NeuralNetRegressor(
        module=MyRegressor,
        criterion=nn.MSELoss,
        optimizer=optim.Adam,
        device='cuda' if torch.cuda.is_available() else 'cpu',
        verbose=0
    )

    param_dist2 = {
        'lr': (10**np.random.uniform(-4.5,-2,50)).tolist(),
        'max_epochs': [300, 400, 500, 600, 700, 800, 1000],
        'batch_size': [64, 128, 256],
        'optimizer__weight_decay': (10**np.random.uniform(-4,-2,50)).tolist(),
        'module__hidden_sizes': [item for sublist in [generate_hidden_sizes(X2.shape[1], num_layers, 15, 100, 5) for num_layers in range(3, 7)] for item in sublist],
    }

    search2 = RandomizedSearchCV(net2, param_dist2, scoring='neg_mean_squared_error', n_iter=50,
                                 n_jobs=-1, cv=3, random_state=seed_value)

    search2.fit(X2, y2)

    cv_results2 = search2.cv_results_
    print(cv_results2['mean_test_score'])  
    print(cv_results2['rank_test_score'])  

    best_model2 = search2.best_estimator_
    best_params2 = search2.best_params_

    print(best_params2)

    predict_mu2 = generate_predictions(data, best_model2, S2, ['a21'], ['a22'])

    pre2=np.concatenate((predict_mu2[0], predict_mu2[1], predict_mu2[2]), axis=1)
    a2_est = np.argmax(pre2, axis=1)

    y1_tilde = np.zeros(data.shape[0])

    for m in range(data.shape[0]):
        y1_tilde[m] = data['Y'].values[m] + pre2[m][int(a2_est[m])] - pre2[m][int(data['a2'].values[m])]
        
    # stage1 estimate
    X1 = data[S1]
    y1 = y1_tilde

    X1 = torch.Tensor(X1.values)
    y1 = torch.Tensor(y1)
    y1 = y1.reshape(-1, 1)

    net1 = NeuralNetRegressor(
        module=MyRegressor,
        criterion=nn.MSELoss,
        optimizer=optim.Adam,
        device='cuda' if torch.cuda.is_available() else 'cpu',
        verbose=0
    )

    param_dist1 = {
        'lr': (10**np.random.uniform(-4,-2,50)).tolist(),
        'max_epochs': [200, 300, 400, 500, 600, 700, 800],
        'batch_size': [64, 128, 256],
        'optimizer__weight_decay': (10**np.random.uniform(-3,-1,100)).tolist(),
        'module__hidden_sizes': [item for sublist in [generate_hidden_sizes(X1.shape[1], num_layers, 10, 100, 10) for num_layers in range(3, 8)] for item in sublist],
        
    }

    search1 = RandomizedSearchCV(net1, param_dist1, scoring='neg_mean_squared_error', n_iter=50,
                                        n_jobs=-1, cv=3, random_state=seed_value)

    search1.fit(X1, y1)

    cv_results1 = search1.cv_results_
    print(cv_results1['mean_test_score'])  
    print(cv_results1['rank_test_score'])

    best_model1 = search1.best_estimator_
    best_params1 = search1.best_params_

    print(best_params1)

    # decide A1
    predict_mu1 = generate_predictions(data, best_model1, S1, ['a11'], ['a12'])

    pre1=np.concatenate((predict_mu1[0], predict_mu1[1], predict_mu1[2]), axis=1)
    a1_est = np.argmax(pre1, axis=1)

    S2_post = H1 + ['a11_est'] + ['a12_est'] + ['vent2','vaso2'] + ['a21_est'] + ['a22_est']
    df_QLNN = data.copy(deep=True)
    df_QLNN['a2_est'] = a2_est
    df_QLNN['a1_est'] = a1_est

    df_QLNN['a11_est'] = df_QLNN['a1_est'].apply(lambda x: 1 if x == 1 else 0)
    df_QLNN['a12_est'] = df_QLNN['a1_est'].apply(lambda x: 1 if x == 2 else 0)
    df_QLNN['a21_est'] = df_QLNN['a2_est'].apply(lambda x: 1 if x == 1 else 0)
    df_QLNN['a22_est'] = df_QLNN['a2_est'].apply(lambda x: 1 if x == 2 else 0)

    X_predict = torch.Tensor(df_QLNN[S2_post].values)
    predict_Y = best_model2.predict(X_predict)

    predict_Esofa = np.mean(25-17*np.log(predict_Y))
    Esofa = np.mean(df_QLNN['sofa'])

    print(df_QLNN['a1'].value_counts())
    print(df_QLNN['a2'].value_counts())
    print(df_QLNN['a1_est'].value_counts())
    print(df_QLNN['a2_est'].value_counts())
    df_QLNN['QLNN_est_sofa'] = 25-17*np.log(predict_Y)
    
    print('bootstrap ',i,' done.')
    
    # save bootsrap results
    g1_est.append(a1_est)
    g2_est.append(a2_est)
    mean_SOFA_est.append(np.mean(25-17*np.log(predict_Y)))
    
np.percentile(mean_SOFA_est,2.5)

np.percentile(mean_SOFA_est,97.5)

mean_SOFA_est_df = pd.DataFrame(mean_SOFA_est, columns=['SOFA_mean'])
