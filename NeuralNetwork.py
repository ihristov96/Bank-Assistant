# Importing the libraries

import time
import keras
import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder

from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split

# Importing the Keras libraries and packages
from keras.layers import Dense, Dropout
from keras import Sequential


class NeuralNetwork:
    def __init__(self, batch_size, epoches):
        self.model = Sequential()
        self.dataset = None
        self.data_scaler = StandardScaler()
        self.batch_size = batch_size
        self.epochs = epoches
        self.x_test = None;
        self.y_test = None;
    def predict(self, client):
        data = self.dataset.iloc[:, 3:13].values
        y = self.dataset.iloc[:, 13].values

        X_client = np.array([client[3:13]])

        data = np.vstack([data, X_client])

        label_encoder_X_1 = LabelEncoder()
        data[:, 1] = label_encoder_X_1.fit_transform(data[:, 1])

        label_encoder_X_2 = LabelEncoder()
        data[:, 2] = label_encoder_X_2.fit_transform(data[:, 2])

        ct = ColumnTransformer(
            [('one_hot_encoder', OneHotEncoder(), [1])],
            # The column numbers to be transformed (here is [0] but can be [0, 1, 3])
            remainder='passthrough'  # Leave the rest of the columns untouched
        )

        data = np.array(ct.fit_transform(data), dtype=np.float)
        data = data[:, 1:]
        data = self.data_scaler.transform(data)

        user = data[-1:]

        result = self.model.predict(user)
        return result[0, 0]

    def learn(self, file, callback):
        X, y = self.preprocess_data(file)

        # Splitting the dataset into the Training set and Test set
        X_train, self.x_test, y_train, self.y_test = train_test_split(X, y, train_size=0.8, random_state=0, shuffle=False)

        # Build the ANN
        # Adding the input layer and the first hidden layer
        self.model.add(Dense(units=6, kernel_initializer='uniform', activation='relu', input_dim=11))

        # Adding the output layer
        self.model.add(Dense(units=1, kernel_initializer='uniform', activation='sigmoid'))

        # Compiling the ANN
        self.model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy'])
        start = time.time()

        # Fitting the ANN to the Training set
        history = self.model.fit(X_train, y_train, batch_size=self.batch_size, epochs=self.epochs,callbacks=[callback])
        end = time.time()

        print("Time: {0}".format(( end - start)))
        # result = self.model.evaluate(X_test, y_test, verbose=1);

        y_pred = self.model.predict(self.x_test)
        y_pred = (y_pred > 0.5)

        cm = confusion_matrix(self.y_test, y_pred)

        return history, cm

    def preprocess_data(self, file):
        # Importing the dataset
        self.dataset = pd.read_csv(file)

        X = self.dataset.iloc[:, 3:13].values
        y = self.dataset.iloc[:, 13].values

        # Encoding categorical data
        labelencoder_X_1 = LabelEncoder()
        X[:, 1] = labelencoder_X_1.fit_transform(X[:, 1])

        labelencoder_X_2 = LabelEncoder()
        X[:, 2] = labelencoder_X_2.fit_transform(X[:, 2])

        ct = ColumnTransformer(
            [('one_hot_encoder', OneHotEncoder(), [1])],
            # The column numbers to be transformed (here is [0] but can be [0, 1, 3])
            remainder='passthrough'  # Leave the rest of the columns untouched
        )

        X = np.array(ct.fit_transform(X), dtype=np.float)
        X = X[:, 1:]

        X = self.data_scaler.fit_transform(X)

        return X, y

