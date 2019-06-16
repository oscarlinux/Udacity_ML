from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.layers import Flatten
from tensorflow.compat.v1.keras.layers import CuDNNLSTM
from tensorflow.keras.layers import RepeatVector, TimeDistributed
from tensorflow.keras.layers import Input, LSTMCell, RNN, Bidirectional, concatenate
from tensorflow.keras.optimizers import Adam#RMSprop
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.metrics import RootMeanSquaredError

from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

from numpy import array, asarray
from sklearn.metrics import mean_squared_error
from math import sqrt



class Model:
   
    def __init__(self, multivariate):
        self.multivariate = multivariate


    def to_supervised(self, train_x, train_y, n_input, y_col, n_out, multivariate):
        
        #Flatten data
        data_x = train_x.reshape((train_x.shape[0]*train_x.shape[1], train_x.shape[2]))
        data_y = train_y.reshape((train_y.shape[0]*train_y.shape[1], train_y.shape[2]))

        X, y = list(), list()
        in_start = 0
        
        #step over the entire history one time step at a time
        for _ in range(len(data_x)):
            #define the end of the input sequence
            in_end = in_start + n_input
            out_end = in_end + n_out
            #ensure we have enough data for this instance
            if out_end < len(data_x):
                if not multivariate:
                    x_input = data_x[in_start:in_end]
                    x_input = x_input.reshape((len(x_input), 1))
                    X.append(x_input)
                    y.append(data_y[in_end:out_end])
                else:
                    X.append(data_x[in_start:in_end, :])
                    y.append(data_y[in_end:out_end])
            #move along one time step
            in_start += 1
        
        train_x, train_y = array(X), array(y)
        

        n_timesteps, n_features, n_outputs = train_x.shape[1], train_x.shape[2], train_y.shape[1]
        
        # reshape output into [samples, timesteps, features]
        train_y = train_y.reshape((train_y.shape[0], train_y.shape[1], 1))
 
        return train_x, train_y, n_timesteps, n_features, n_outputs 



    def LSTMEncoderDecoder(self, enconderUnits, decoderUnits, denseUnits, dropout, n_timesteps, n_features, n_outputs):
        
        

        model = Sequential()
           #ENCODER
        model.add(CuDNNLSTM(enconderUnits, input_shape=(n_timesteps, n_features)))
        model.add(Dropout(dropout))
        model.add(RepeatVector(n_outputs))
        
            #DECODER
        model.add(CuDNNLSTM(decoderUnits, return_sequences=True))
        model.add(Dropout(dropout))
        model.add(CuDNNLSTM(int(decoderUnits/2), return_sequences=True))
        model.add(Dropout(dropout))
            #OUTPUT
        model.add(TimeDistributed(Dense(denseUnits, activation='relu')))
        model.add(TimeDistributed(Dense(1)))
        
        
        return model
 
    def modelCompile(self, model, learning_rate, decay, amsgrad):
        #compile the model
        adam = Adam(learning_rate=learning_rate, decay=decay, amsgrad=amsgrad)
        model.compile(loss='mean_squared_error', optimizer=adam, metrics=[RootMeanSquaredError(),'mae','acc'])
        return model
    
    def modelfit(self, model, train_x, train_y, epochs, batch_size, validation_split, verbose, checkpointFilePath):
        checkpointer = ModelCheckpoint(filepath=checkpointFilePath,
                                      verbose=verbose, save_best_only=True)
        model.fit(train_x, train_y, epochs=epochs, batch_size=batch_size,
                  validation_split=validation_split, callbacks=[checkpointer],
                  verbose=verbose, shuffle=True)
        return model
        
    def protoConfig(self):
        config = ConfigProto()
        config.gpu_options.allow_growth = True
        session = InteractiveSession(config=config)
        return session
    
    #evaluate model
    def evaluate_model(self, train_x, test_x, test_y, model, n_input, weightsFile, scaler_x, scaler_y, multivariate):
        model.load_weights(weightsFile)
        #history is a list of n_input data
        history = [x for x in train_x]

        #walk-forward validation 
        predictions = list()
        for i in range(len(test_x)):
            #predict the week
            yhat_sequence = self.forecast(model, history, n_input, multivariate)
            
            #store the predictions
            predictions.append(yhat_sequence)
            #get real observations and add to history to predicting
            history.append(test_x[i, :])
        #evaluate predictions for each time step
        predictions = array(predictions)
        
        #invert scale
        test_y = asarray([scaler_y.inverse_transform(step) for step in test_y])
        predictions = asarray([scaler_y.inverse_transform(step) for step in predictions])
        
        #Evaluate forecast
        score, scores = self.evaluate_forecast(test_y, predictions)
        return score, scores, model, test_y, predictions
    

        #make a forecast
    def forecast(self, model, history, n_input, multivariate):
        #flatten data
        data = array(history)
        data = data.reshape((data.shape[0]*data.shape[1], data.shape[2]))

        if not multivariate:
            #retrieve last observation for input data
            input_x = data[-n_input:]
            #reshape into [1, n_input, 1]
            input_x = input_x.reshape((1, len(input_x), 1))
        else:
            #retrieve last observation for input data
            input_x = data[-n_input:,:]
            #reshape into [1, n_input, 1]
            input_x = input_x.reshape((1, input_x.shape[0], input_x.shape[1]))
            

        #forecast the next step sequence
        yhat = model.predict(input_x, verbose=0)
        #we only want the vector forecast
        yhat = yhat[0]
        return yhat
    
        #evaluate one or more sequence's forecast against expected values
    def evaluate_forecast(self, actual, predicted):
        scores = list()

        #calculate an RMSE score for each sequence
        for i in range(actual.shape[1]):
            #calculate mse
            mse = mean_squared_error(actual[:, i], predicted[:, i])
            #calculate rmse
            rmse = sqrt(mse)
            #store
            scores.append(rmse)
        #calculate overall RMSE
        s = 0
        for row in range(actual.shape[0]):
            for col in range(actual.shape[1]):
                s += (actual[row, col] - predicted[row, col])**2
        score = sqrt(s / (actual.shape[0] * actual.shape[1]))
        return score, scores
    
        #summarize scores
    def summarize_scores(self, name, score, scores):
        s_scores = ', '.join(['%.1f' % s for s in scores])
        print('%s: [%.3f] %s' % (name, score, s_scores))
        
    def invert_scale(self, scaler, normalized):
        inversed = scaler.inverse_transform(normalized)
        return(inversed)
        