H1 = [ 'x%s' %i for i in range(1,6)]
H2 = H1 + ['a11'] + ['a12'] + ['R1']
S1 = H1 + ['a11'] + ['a12']
S2 = H2 + ['a21'] + ['a22']
S2_post = H2 + ['a21_est']+ ['a22_est']
action_values = [[(0,0),(1,0),(0,1)],[(0,0),(1,0),(0,1)]]

seed_value = 686
np.random.seed(seed_value)
random.seed(seed_value)

train_df = sim_data(n=1000,p=5,r=0.6)
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
            # self.hidden_layers.append(weight_norm(nn.Linear(hidden_sizes[i+1], hidden_sizes[i+1])))
        

        self.output_layer = nn.Linear(hidden_sizes[-1], 1)
        # self.output_activation = activation()
    
    def forward(self, x):
        for layer in self.hidden_layers:
            x = layer(x)
        x = self.output_layer(x)
        # x = self.output_activation(x)
        return x


def generate_hidden_sizes(input_size, num_layers, min_nodes, max_nodes, num_combinations):
    return [(input_size,) + tuple(random.randint(min_nodes, max_nodes) for _ in range(num_layers - 1)) for _ in range(num_combinations)]


net2 = NeuralNetRegressor(
    module=MyRegressor,
    criterion=nn.MSELoss,
    optimizer=optim.Adam,
    # lr=0.00008782749767658895,
    # max_epochs=800,
    # batch_size=128,
    device='cuda' if torch.cuda.is_available() else 'cpu',
    # callbacks=[LossCallback()],
    verbose=0
    # train_split=skorch.dataset.ValidSplit(3)
)


param_dist2 = {
    'lr': (10**np.random.uniform(-4.5,-2,50)).tolist(),
    'max_epochs': [500,800,1000],
    'batch_size': [256, 512],
    'optimizer__weight_decay': (10**np.random.uniform(-3,-1,50)).tolist(),
    'module__hidden_sizes': [item for sublist in [generate_hidden_sizes(X2_train.shape[1], num_layers, 10, 100, 10) for num_layers in range(5, 8)] for item in sublist]

}

search2 = RandomizedSearchCV(net2, param_dist2, scoring='neg_mean_squared_error', n_iter=50,
                                    n_jobs=-1, cv=5, random_state=seed_value)
# search2 = GridSearchCV(net2, param_dist2, scoring='neg_mean_squared_error', n_jobs=-1, cv=5)

search2.fit(X2_train, y2_train)

cv_results2 = search2.cv_results_
# print(cv_results2['params'])
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

for i in range(len(train_df)):
    a2_value = int(train_df.at[i, 'a2'])
    Y_value = train_df.at[i, 'R2']
    pre2_train[i, a2_value] = Y_value

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
    # lr=0.0007099509982898103,
    # optimizer__momentum=0.9708209555058809,
    # max_epochs=1200,
    # batch_size=256,
    device='cuda' if torch.cuda.is_available() else 'cpu',
    verbose=0
)

param_dist1 = {
    'lr': (10**np.random.uniform(-5,-2,50)).tolist(),
    # # 'optimizer__momentum':(np.random.uniform(0.9,0.99,50)).tolist(),
    'max_epochs': [500, 800, 1000, 1200],
    'batch_size': [256, 512],
    'optimizer__weight_decay': (10**np.random.uniform(-3,-1,100)).tolist(),
    'module__hidden_sizes': [item for sublist in [generate_hidden_sizes(X1_train.shape[1], num_layers, 10, 100, 10) for num_layers in range(6, 8)] for item in sublist]    
}

search1 = RandomizedSearchCV(net1, param_dist1, scoring='neg_mean_squared_error', n_iter=50,
                                    n_jobs=-1, cv=5, random_state=seed_value)
# search1 = GridSearchCV(net1, param_dist1, scoring='neg_mean_squared_error', n_jobs=-1, cv=5)


search1.fit(X1_train, y1_train)

cv_results1 = search1.cv_results_
# print(cv_results1['params'])  
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
test_df = sim_data(n=1000,p=5,r=0.6)
# stage 1
predict_mu1_test = generate_predictions(test_df, best_model1, S1, ['a11'], ['a12'])
pre1_test=np.concatenate((predict_mu1_test[0], predict_mu1_test[1], predict_mu1_test[2]), axis=1)
a1_est_test = np.argmax(pre1_test, axis=1)
R1_est_test = np.exp(1 + 0.3*test_df['x1']**2+(a1_est_test==0)*(0.4+0.8*test_df['x1'])+(a1_est_test==1)*(np.exp(test_df['x2'])-1)+(a1_est_test==2)*(0.7+0.6*test_df['x2']**2))
R1_true_test = np.exp(1 + 0.3*test_df['x1']**2+(test_df['a1_opt']==0)*(0.4+0.8*test_df['x1'])+(test_df['a1_opt']==1)*(np.exp(test_df['x2'])-1)+(test_df['a1_opt']==2)*(0.7+0.6*test_df['x2']**2))

predict_mu2_test = generate_predictions(test_df, best_model2, S2, ['a21'], ['a22'])
pre2_test=np.concatenate((predict_mu2_test[0], predict_mu2_test[1], predict_mu2_test[2]), axis=1)

for i in range(len(test_df)):
    a2_value = int(test_df.at[i, 'a2'])
    Y_value = test_df.at[i, 'R2']
    pre2_test[i, a2_value] = Y_value

a2_est_test = np.argmax(pre2_test, axis=1)
R2_est_test = np.exp(0.2*test_df['x2']**2+(a2_est_test==0)*(1-0.1*np.exp(test_df['x4']))+(a2_est_test==1)*(1.5*test_df['x5'])+(a2_est_test==2)*(1.3+1.2*np.log(test_df['x3'])))
R2_true_test = np.exp(0.2*test_df['x2']**2+(test_df['a2_opt']==0)*(1-0.1*np.exp(test_df['x4']))+(test_df['a2_opt']==1)*(1.5*test_df['x5'])+(test_df['a2_opt']==2)*(1.3+1.2*np.log(test_df['x3'])))

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
