import pandas as pd
from numpy import array
from numpy import split
from pandas import read_csv
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.optimizers import RMSprop

class Data:
    
    def __init__(self, fileName, multivariate):
        self.fileName = fileName
        self.multivariate = multivariate
        
        
    def dataPrep(self):
        data = read_csv(self.fileName, header=0)
        data.columns = ['timestamp','wind_speed','wind_direction','energy']
        data['wind_speed'] = data['wind_speed'].str.replace(',','.')
        data['energy'] = data['energy'].str.replace('.','')
        data['energy'] = data['energy'].str.replace(',','.')
        #replace empty data by 0
        data.replace('-', '0', inplace=True)
        data['timestamp'] = pd.to_datetime(data['timestamp'],dayfirst=True)
        data['wind_speed'] = pd.to_numeric(data['wind_speed'])
        data['wind_direction'] = pd.to_numeric(data['wind_direction'])
        data['energy'] = pd.to_numeric(data['energy'])
        data['week'] = data['timestamp'].dt.week
        data.set_index('timestamp', drop=True, append=False, inplace=True, verify_integrity=False)
        data = data[['energy','wind_speed','wind_direction','week']]
        return(data)
    
    # split dataset into train/test sets
    def split_dataset(self, data_x, data_y, startTrain, endTrain, startTest, endTest, n_out):
        # split into test and training datasets
        train, test = data_x[startTrain:endTrain], data_x[startTest:endTest]
        train_y, test_y = data_y[startTrain:endTrain], data_y[startTest:endTest]
        
        #normalize data
        scaler_x = self.minmax_scaler(train)
        scaler_y = self.minmax_scaler(train_y)
        train_x = scaler_x.transform(train)
        test_x = scaler_x.transform(test)
        train_y = scaler_y.transform(train_y)
        test_y = scaler_y.transform(test_y)
        
        # restructure into windows of n_out hours
        train_x = array(split(array(train_x), len(train_x)/n_out))
        test_x = array(split(array(test_x), len(test_x)/n_out))
        train_y = array(split(array(train_y), len(train_y)/n_out))
        test_y = array(split(array(test_y), len(test_y)/n_out))
        return train_x, test_x, train_y, test_y, scaler_x, scaler_y
    

        
    def minmax_scaler(self, data):
        scaler = MinMaxScaler(feature_range=(-1, 1))
        scaler = scaler.fit(data)
        return(scaler)

    def invert_scale(self, scaler, normalized):
        inversed = scaler.inverse_transform(normalized)
        return(inversed)
    
        
