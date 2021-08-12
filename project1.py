import pandas as pd
import numpy as np
from sklearn import datasets
#from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
#1. Load and Pre-Process
dt = pd.read_csv('aptrain.csv')#Load Dataset
data = dt.drop(['ap'],axis=1)#Remove unnecessary columns
target = dt['ap']#Select Target Column
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(data,target
,test_size=0.25,random_state=0)
#2. Training Phase
knn = KNeighborsClassifier(n_neighbors=4)#Define k-value
X_test.fillna( method ='ffill', inplace = True) 
y_test.fillna( method ='ffill', inplace = True) 
y_train.fillna( method ='ffill', inplace = True) 
X_train.fillna( method ='ffill', inplace = True) 

knn.fit(X_train, y_train)#Send Data, Target to Fitness function
#3. Measuring Accuracy
pred_y = knn.predict(X_test)#Do a test prediction on actual train data
acc = accuracy_score(pred_y,y_test)#Calculate accuracy
print("Accuracy is: ", acc)
#4. Working with Test Data
test_data = pd.read_csv('aptest.csv')#Define input test data
prediction = knn.predict(test_data)#Send test data to predict function
print(prediction)#Show Predictions
import Crypto
from Crypto.PublicKey import RSA
from Crypto import Random
import ast
random_generator = Random.new().read
key = RSA.generate(1024, random_generator) #generate pub and priv key
publickey = key.publickey() # pub key export for exchange
encrypted = publickey.encrypt('encrypt this message', 32)
#message to encrypt is in the above line 'encrypt this message'
print('encrypted message:', encrypted #ciphertext
f = open ('encryption.txt', 'w')
f.write(str(encrypted)) #write ciphertext to file
f.close()
#decrypted code below
f = open('encryption.txt', 'r')
message = f.read()
decrypted = key.decrypt(ast.literal_eval(str(encrypted)))
print('decrypted', decrypted)
f = open ('encryption.txt', 'w')
f.write(str(message))
f.write(str(decrypted))
f.close()
