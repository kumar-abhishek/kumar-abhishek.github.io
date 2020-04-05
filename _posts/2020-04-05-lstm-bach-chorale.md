---
layout: post
title: "Predicting chorale notes using RNN/LSTM/GRU using Keras"
date: 2020-04-05
---
### Predicting chorale notes using RNN/LSTM/GRU using Keras


Predicting chorale notes using RNN/LSTM/GRU using Keras


I am working with Bach chorales dataset. Each chorale is ~100–500 time steps long, and each time step contains 4 integers(for example: [74, 70, 65, 58]) where each integer corresponds to a note's index on a piano.
The Bach chorales dataset can be downloaded from here. Unzip it. It is composed of 382 chorales composed by Johann Sebastian Bach. Each chorale is 100 to 640 time steps long, and each time step contains 4 integers, where each integer corresponds to a note's index on a piano (except for the value 0, which means that no note is played). Train a model - recurrent, convolutional, or both - that can predict the next time step (four notes), given a sequence of time steps from a chorale.
How does our input look like?
```python
X_train_000 = pd.read_csv('chorale_000.csv')
print(X_train_000.head())
>>> 
  note0  note1  note2  note3
0     74     70     65     58
1     74     70     65     58
2     74     70     65     58
3     74     70     65     58
4     75     70     58     55

X_valid_229 = pd.read_csv('chorale_229.csv')
print(X_valid_229.head(5))
>>>
   note0  note1  note2  note3
0     72     67     60     48
1     72     67     60     48
2     72     67     60     48
3     72     67     60     48
4     72     67     64     48

X_train_000 = X_train_000.to_numpy()
X_valid_229 = X_valid_229.to_numpy()
dataX=X_train_000[:-1] # input
dataY=X_train_000[1:]  # output
print(dataX[0:5])
print(dataY[0:5])
>>>
[[74 70 65 58]
 [74 70 65 58]
 [74 70 65 58]
 [74 70 65 58]
 [75 70 58 55]]

[[74 70 65 58]
 [74 70 65 58]
 [74 70 65 58]
 [75 70 58 55]
 [75 70 58 55]]

```


Let's try to standardize the dataset 
```python
scaler = MinMaxScaler(feature_range=(0, 1))
dataXScaler = scaler.fit_transform(dataX)
dataYScaler = scaler.fit_transform(dataY)
print(dataXScaler[0:5])
print(dataYScaler[0:5])
>>>
[[0.44444444 0.77777778 1.         0.88235294]
 [0.44444444 0.77777778 1.         0.88235294]
 [0.44444444 0.77777778 1.         0.88235294]
 [0.44444444 0.77777778 1.         0.88235294]
 [0.55555556 0.77777778 0.41666667 0.70588235]]

[[0.44444444 0.77777778 1.         0.88235294]
 [0.44444444 0.77777778 1.         0.88235294]
 [0.44444444 0.77777778 1.         0.88235294]
 [0.55555556 0.77777778 0.41666667 0.70588235]
 [0.55555556 0.77777778 0.41666667 0.70588235]]
```




Let's use TimeSeriesGenerator
```python
from keras.preprocessing.sequence import TimeseriesGenerator
# make length as a hyperparameter and find it's optimal value via GridSearchCV?
generator = TimeseriesGenerator(dataXScaler, dataYScaler, length=3, batch_size=32)
validation_generator = TimeseriesGenerator(dataValidationX, dataValidationY, length=3, batch_size=32)
print(generator[0])
```


[TODO: More explanation here about what we did]
Let's try to train a model that can predict the next time step(4 notes), given a sequence of time steps from the chorale.
```python
from keras.models import Sequential
from keras import optimizers
from keras.layers import LSTM
from keras.layers import Dense, Dropout, BatchNormalization, TimeDistributed

n_features = 4
# choose a number of time steps
n_steps = None

model = Sequential()
model.add(TimeDistributed(Dense(128), input_shape=(None, n_features)))
model.add(LSTM(64, activation='tanh', input_shape=(None, n_features), return_sequences=True))
model.add(BatchNormalization())

model.add(LSTM(32 , activation = 'tanh'))
model.add(BatchNormalization())
model.add(Dropout(0.2))
model.add(Dense(n_features, activation='softmax'))
model.compile(optimizer='adam', loss='categorical_crossentropy')

# fit model
model.fit_generator(generator, epochs=500, validation_data=validation_generator)
```


Let's try to predict something by throwing an input of different sizes and see how if it does some sensible prediction
```python
x_input = dataXScaler[2:30]
x_input = x_input.reshape((1, len(x_input), 4))
print(scaler.inverse_transform(dataXScaler)[2:30])
yhat = model.predict(x_input, verbose=0)
print(scaler.inverse_transform(yhat))
print('expected: ', scaler.inverse_transform(dataYScaler)[30])
>>>
[[72.33333333 70.22222222 67.         58.23529412]
 [72.33333333 70.22222222 67.         58.23529412]
 [73.66666667 70.22222222 60.         55.58823529]
 [73.66666667 70.22222222 60.         55.58823529]
 [73.66666667 70.22222222 62.         55.58823529]
 [73.66666667 70.22222222 62.         55.58823529]
 [76.33333333 69.33333333 64.         51.17647059]
 [76.33333333 69.33333333 64.         51.17647059]
 [76.33333333 69.33333333 64.         51.17647059]
 [76.33333333 69.33333333 64.         51.17647059]
 [76.33333333 70.22222222 64.         55.58823529]
 [76.33333333 70.22222222 64.         55.58823529]
 [76.33333333 69.33333333 64.         55.58823529]
 [76.33333333 69.33333333 64.         55.58823529]
 [73.66666667 67.55555556 65.         49.41176471]
 [73.66666667 67.55555556 65.         49.41176471]
 [73.66666667 69.33333333 65.         49.41176471]
 [73.66666667 69.33333333 65.         49.41176471]
 [72.33333333 70.22222222 67.         47.64705882]
 [72.33333333 70.22222222 67.         47.64705882]
 [72.33333333 70.22222222 67.         47.64705882]
 [72.33333333 70.22222222 67.         47.64705882]
 [69.66666667 69.33333333 67.         53.82352941]
 [69.66666667 69.33333333 67.         53.82352941]
 [69.66666667 69.33333333 67.         53.82352941]
 [69.66666667 69.33333333 67.         53.82352941]
 [69.66666667 69.33333333 67.         53.82352941]
 [69.66666667 69.33333333 67.         53.82352941]]
[[70.109344 65.59594  58.57434  48.653004]]
expected:  [69.66666667 69.33333333 67.         53.82352941]
```


Seems like it did a pretty good job of predicting the next chord. 
Now, let's use this model to generate Bach-like music, one note at a time: we can do this by giving the model the start of a chorale and asking it to predict the next time step, then appending these time steps to the input sequence and asking the model for the next note, and so on. 
```python
n_steps = None
# convert into input/output
i=1
n=1
x_input = np.array([dataXScaler[0]])

x_input = x_input.reshape((1, len(x_input), n_features))
while i<len(dataXScaler):
  # demonstrate prediction
  print('Input: ')
  for input in x_input:
    print(scaler.inverse_transform(input))
  print('---------------')
  yhat = model.predict(x_input, verbose=1)
  print('Predicted Output: ', scaler.inverse_transform(yhat))
  print('expected: ', scaler.inverse_transform(dataYScaler)[i])
  print('\n\n')

  yhat = yhat.reshape((1, len(yhat), n_features))
  x_input = np.concatenate([x_input, yhat], axis=1)
  i += 1
  if i>5: # predict only 5 times, you can continue forever if you like
    break
>>>
Input: 
[[72.33333333 70.22222222 67.         58.23529412]]
---------------
1/1 [==============================] - 0s 5ms/step
Predicted Output:  [[75.26765  65.903206 55.87507  45.003094]]
expected:  [72.33333333 70.22222222 67.         58.23529412]

Input: 
[[72.33333333 70.22222222 67.         58.23529412]
 [75.26764822 65.90320635 55.87506586 45.00309411]]
---------------
1/1 [==============================] - 0s 2ms/step
Predicted Output:  [[78.15341 64.03464 55.79461 45.00003]]
expected:  [72.33333333 70.22222222 67.         58.23529412]

Input: 
[[72.33333333 70.22222222 67.         58.23529412]
 [75.26764822 65.90320635 55.87506586 45.00309411]
 [78.15341401 64.03463399 55.79460961 45.0000323 ]]
---------------
1/1 [==============================] - 0s 4ms/step
Predicted Output:  [[75.54515  64.27181  58.038734 45.010487]]
expected:  [73.66666667 70.22222222 60.         55.58823529]

Input: 
[[72.33333333 70.22222222 67.         58.23529412]
 [75.26764822 65.90320635 55.87506586 45.00309411]
 [78.15341401 64.03463399 55.79460961 45.0000323 ]
 [75.54515028 64.27181584 58.03873682 45.0104869 ]]
---------------
1/1 [==============================] - 0s 2ms/step
Predicted Output:  [[71.08536  65.97478  57.158997 48.49185 ]]
expected:  [73.66666667 70.22222222 60.         55.58823529]

Input: 
[[72.33333333 70.22222222 67.         58.23529412]
 [75.26764822 65.90320635 55.87506586 45.00309411]
 [78.15341401 64.03463399 55.79460961 45.0000323 ]
 [75.54515028 64.27181584 58.03873682 45.0104869 ]
 [71.08535445 65.97477663 57.15900016 48.49185035]]
---------------
1/1 [==============================] - 0s 2ms/step
Predicted Output:  [[70.36781  65.637566 58.040543 48.91912 ]]
expected:  [73.66666667 70.22222222 62.         55.58823529]
```


You can use SimpleRNN 
```python
from keras.layers import Dense, SimpleRNN

model_rnn = Sequential()
model_rnn.add(TimeDistributed(Dense(128), input_shape=(None, n_features)))
model_rnn.add(SimpleRNN(100, input_shape=[None, n_features], return_sequences=True ))
model_rnn.add(SimpleRNN(100))
model_rnn.add(Dense(n_features, activation='softmax'))

model_rnn.compile(optimizer='adam', loss='categorical_crossentropy')
model_rnn.fit_generator(generator, epochs=500, validation_data=validation_generator)
```


As well as GRU
```python
from keras.models import Sequential
from keras.layers import GRU, Dense

n_features = 4
model_gru = Sequential()
model_gru.add(TimeDistributed(Dense(128), input_shape=(None, n_features)))
model_gru.add(GRU(64, activation='tanh', input_shape=(None, n_features), return_sequences=True))
model_gru.add(BatchNormalization())

model_gru.add(GRU(32 , activation = 'tanh'))
model_gru.add(BatchNormalization())
model_gru.add(Dropout(0.2))
model_gru.add(Dense(n_features, activation='softmax'))
model_gru.compile(optimizer='adam', loss='categorical_crossentropy')

model_gru.fit_generator(generator, epochs=500, validation_data=validation_generator)
```

##### References
[How to Use the TimeseriesGenerator for Time Series Forecasting in Keras](https://medium.com/r/?url=https%3A%2F%2Fmachinelearningmastery.com%2Fhow-to-use-the-timeseriesgenerator-for-time-series-forecasting-in-keras%2F)
