#Kode efter Magnus ændringer 

### main code

import torch
import pandas as pd
import numpy as np
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

df = pd.read_csv('/Users/benedicte/Documents/02461-IntelligentSystems/Januarprojekt/mentalhealthdata.csv')
patients_info = df.loc[:,'sex':'specific.disorder']

encoder = OneHotEncoder(sparse=False)
y = encoder.fit_transform(df[['specific.disorder']])

# Encoding 'sex' as 0 for M and 1 for F
df['sex'] = df['sex'].map({'M': 0, 'F': 1})  


## missing data 
# for idx, col in enumerate(df.isna().any()):
#     if col == True:
#         print(idx, col)
# print(processed_df.isna().any(axis=1))

# Replacing nan values in the columns Education and IQ  
df[["education", "IQ"]] = df[["education", "IQ"]].fillna(df[["education", "IQ"]].median())

# Deleting non-relevant columns
columns_to_delete = ['main.disorder','eeg.date','no.',"Unnamed: 122"]
df = df.drop(columns_to_delete, axis=1)

#standardiserer alle kolonner bortset fra specific.disorder

columns_to_standardise = df.columns.difference(['specific.disorder', 'sex'])
scaler = StandardScaler()
df[columns_to_standardise] = scaler.fit_transform(df[columns_to_standardise])

#print(df)

##  test multiclasification 
X = df.loc[:, df.columns != "specific.disorder"]
#y = df.loc[:,"specific.disorder"]

device = torch.device('cpu')
#device = torch.device('cuda')


# Manually set random seed - vi har sat vores seed for reproducerbarhed 
torch.manual_seed(42)


# splitter dataen i trænings- og testsæt. 
k_folds = 10

# Number of iterations
T = 200
acc_list = list()
for train, test in KFold(n_splits=k_folds, shuffle=True, random_state=69).split(X,y):
    # Use the nn package to define our model and loss function.
    model = torch.nn.Sequential(
        torch.nn.Linear(1144, 1000),
        torch.nn.ReLU(),
        torch.nn.Linear(1000, 500),
        torch.nn.ReLU(),
        torch.nn.Linear(500, 250),
        torch.nn.ReLU(),
        torch.nn.Linear(250, 12),
        #torch.nn.Softmax()
    )

    model.to(device)
    loss_fn = torch.nn.CrossEntropyLoss(reduction='mean')


    learning_rate = 10e-4
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=0.01)

    X_train, y_train = X.iloc[train], y[train]
    X_test, y_test = X.iloc[test], y[test]
    
    X_train_tensor = torch.tensor(X_train.values, dtype=torch.float32).to(device)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32).to(device)
    
    X_test_tensor = torch.tensor(X_test.values, dtype=torch.float32).to(device)
    y_test_tensor = torch.tensor(y_test, dtype=torch.float32).to(device)
    loss_list = []
    for t in tqdm(range(T)):
        # Forward pass: compute predicted y by passing x to the model.
        y_pred = model(X_train_tensor)

        # Compute and save loss.
        # print(X_train)
        # print(y_pred, y_train)
        train_loss = loss_fn(y_pred, y_train_tensor)
        #Loss[t] = loss.item()

        optimizer.zero_grad()

        # Backward pass: compute gradient of the loss with respect to model
        train_loss.backward()
        loss_list.append(train_loss.item())
        # Calling the step function on an Optimizer makes an update to its
        # parameters
        optimizer.step()    
    plt.plot(loss_list)
        # print(train_loss)
    # print(X_test_tensor.shape)
    y_pred = model(X_test_tensor)

    a = y_pred.argmax (1)
    guess = torch.zeros(y_pred.shape).scatter(1, a.unsqueeze (1), 1.0)

    test_loss = loss_fn(guess, y_test_tensor)
    acc = (sum(torch.argmax(guess,dim=1) == torch.argmax(y_test_tensor,dim=1))/len(y_test_tensor))*100
    print(test_loss,acc)
    acc_list.append(acc)
    #break
    
plt.legend(list(range(1,11)))
plt.show()
plt.bar(list(range(1,11)),acc_list)
plt.show()





### main code

import torch
import pandas as pd
import numpy as np
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

df = pd.read_csv('EEG.machinelearing_data_BRMH.csv')
patients_info = df.loc[:,'sex':'specific.disorder']

encoder = OneHotEncoder(sparse=False)
y = encoder.fit_transform(df[['specific.disorder']])

# Encoding 'sex' as 0 for M and 1 for F
df['sex'] = df['sex'].map({'M': 0, 'F': 1})  

## missing data 
# for idx, col in enumerate(df.isna().any()):
#     if col == True:
#         print(idx, col)
# print(processed_df.isna().any(axis=1))

# Replacing nan values in the columns Education and IQ  
df[["education", "IQ"]] = df[["education", "IQ"]].fillna(df[["education", "IQ"]].median())

# Deleting non-relevant columns
columns_to_delete = ['main.disorder','eeg.date','no.',"Unnamed: 122"]
df = df.drop(columns_to_delete, axis=1)

#standardiserer alle kolonner bortset fra specific.disorder

columns_to_standardise = df.columns.difference(['specific.disorder', 'sex'])
scaler = StandardScaler()
df[columns_to_standardise] = scaler.fit_transform(df[columns_to_standardise])

#print(df)

##  test multiclasification 
X = df.loc[:, df.columns != "specific.disorder"]
#y = df.loc[:,"specific.disorder"]

device = torch.device('cpu')
#device = torch.device('cuda')


# Manually set random seed - vi har sat vores seed for reproducerbarhed 
torch.manual_seed(42)


# splitter dataen i trænings- og testsæt. 
k_folds = 10

# Number of iterations
T = 200
acc_list = list()
for train, test in KFold(n_splits=k_folds, shuffle=True, random_state=69).split(X,y):
    # Use the nn package to define our model and loss function.
    model = torch.nn.Sequential(
        torch.nn.Linear(1144, 1000),
        torch.nn.ReLU(),
        torch.nn.Linear(1000, 500),
        torch.nn.ReLU(),
        torch.nn.Linear(500, 250),
        torch.nn.ReLU(),
        torch.nn.Linear(250, 12),
        #torch.nn.Softmax()
    )

    model.to(device)
    loss_fn = torch.nn.CrossEntropyLoss(reduction='mean')


    learning_rate = 10e-4
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=0.01)




    X_train, y_train = X.iloc[train], y[train]
    X_test, y_test = X.iloc[test], y[test]
    
    X_train_tensor = torch.tensor(X_train.values, dtype=torch.float32).to(device)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32).to(device)
    
    X_test_tensor = torch.tensor(X_test.values, dtype=torch.float32).to(device)
    y_test_tensor = torch.tensor(y_test, dtype=torch.float32).to(device)
    loss_list = []
    for t in tqdm(range(T)):
        # Forward pass: compute predicted y by passing x to the model.
        y_pred = model(X_train_tensor)

        # Compute and save loss.
        # print(X_train)
        # print(y_pred, y_train)
        train_loss = loss_fn(y_pred, y_train_tensor)
        #Loss[t] = loss.item()

        optimizer.zero_grad()

        # Backward pass: compute gradient of the loss with respect to model
        train_loss.backward()
        loss_list.append(train_loss.item())
        # Calling the step function on an Optimizer makes an update to its
        # parameters
        optimizer.step()    
    plt.plot(loss_list)
        # print(train_loss)
    # print(X_test_tensor.shape)
    y_pred = model(X_test_tensor)

    a = y_pred.argmax (1)
    guess = torch.zeros(y_pred.shape).scatter(1, a.unsqueeze (1), 1.0)

    test_loss = loss_fn(guess, y_test_tensor)
    acc = (sum(torch.argmax(guess,dim=1) == torch.argmax(y_test_tensor,dim=1))/len(y_test_tensor))*100
    print(test_loss,acc)
    acc_list.append(acc)
    #break
plt.legend(list(range(1,11)))
plt.show()
plt.bar(list(range(1,11)),acc_list)
plt.show()


## guess code
# defineres i kfold
guess_dict = {}

# bruges i kfold loop 
for item in guess:
    guess_dict[str(item)] = guess_dict.get(str(item), 0 ) + 1
print(guess_dict)


y_pred_labels = torch.argmax(y_pred, dim=1).cpu().numpy()
    y_test_labels = torch.argmax(y_test_tensor, dim=1).cpu().numpy()
    
    cm = confusion_matrix(y_test_labels, y_pred_labels)
    print("Confusion Matrix for fold:", cm)

    # Visualization of Confusion Matrix
    plt.figure(figsize=(10, 7))
    sns.heatmap(cm, annot=True, fmt="d")
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.title('Confusion Matrix')
    plt.show()
