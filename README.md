# Developing a Neural Network Regression Model

## AIM

To develop a neural network regression model for the given dataset.

## THEORY

Explain the problem statement

## Neural Network Model

Include the neural network model diagram.

## DESIGN STEPS

### STEP 1:

Loading the dataset

### STEP 2:

Split the dataset into training and testing

### STEP 3:

Create MinMaxScalar objects ,fit the model and transform the data.

### STEP 4:

Build the Neural Network Model and compile the model.

### STEP 5:

Train the model with the training data.

### STEP 6:

Plot the performance plot

### STEP 7:

Evaluate the model with the testing data.

## PROGRAM

```
NAME:KATHIRVEL.A
REG NO:212221230047
```
```
### To Read CSV file from Google Drive :

from google.colab import auth
import gspread
from google.auth import default
import pandas as pd

### Authenticate User:

auth.authenticate_user()
creds, _ = default()
gc = gspread.authorize(creds)

### Open the Google Sheet and convert into DataFrame :

worksheet = gc.open('model 1').sheet1
rows = worksheet.get_all_values()
df = pd.DataFrame(rows[1:], columns = rows[0])
df = df.astype({'input':'int','output':'int'})

### Import the packages :

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

df.head()

X = df[['input']].values
y = df[['output']].values
X

### Split Training and testing set :

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.33,random_state = 42)

### Pre-processing the data :

Scaler = MinMaxScaler()
Scaler.fit(X_train)
Scaler.fit(X_test)

X_train1 = Scaler.transform(X_train)
X_test1 = Scaler.transform(X_test)

X_train1

### Model :

ai_brain = Sequential([
    Dense(4,activation = 'relu'),
    Dense(6,activation = 'relu'),
    Dense(1)
])

ai_brain.compile(
    optimizer = 'rmsprop',
    loss = 'mse'
)

ai_brain.fit(X_train1,y_train,epochs = 4000)

### Loss plot :

loss_df = pd.DataFrame(ai_brain.history.history)

loss_df.plot()

### Testing with the test data and predicting the output :

ai_brain.evaluate(X_test1,y_test)

X_n1 = [[38]]

X_n1_1 = Scaler.transform(X_n1)

ai_brain.predict(X_n1_1)
```

## Dataset Information

![image](https://github.com/KathirvelAIDS/basic-nn-model/assets/94911373/fa3f45a1-d60f-4c6d-a85d-87c3bac23b66)


## OUTPUT




### Training Loss Vs Iteration Plot



![image](https://github.com/KathirvelAIDS/basic-nn-model/assets/94911373/b8b19c10-7f13-486b-ab65-5ef5c9e682b2)




### Test Data Root Mean Squared Error


![image](https://github.com/KathirvelAIDS/basic-nn-model/assets/94911373/fb4692ac-5d9c-4537-be0f-aafee3db4cde)






### New Sample Data Prediction


![image](https://github.com/KathirvelAIDS/basic-nn-model/assets/94911373/fd828ee6-e97b-4e78-9267-c9223b7baabf)






## RESULT:



Thus a neural network model for regression using the given dataset is written and executed successfully.
