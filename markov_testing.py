import sys
sys.path.append(r'../')
import numpy as np
from functions import *

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense



algorithm = 'BunDLeNet'
worm_num = 0
state_names = ['Dorsal turn', 'Forward', 'No state', 'Reverse-1', 'Reverse-2', 'Sustained reversal', 'Slowing', 'Ventral turn']
Y0_ = np.loadtxt('data/generated/saved_Y/Y0__' + algorithm + '_worm_' + str(worm_num))
B_ = np.loadtxt('data/generated/saved_Y/B__' + algorithm + '_worm_' + str(worm_num)).astype(int)
#plot_latent_timeseries(np.c_[Y0, Y1, Y2], B_, state_names)

Y0 = Y0_[0:-2]
Y1 = Y0_[1:-1]
Y2 = Y0_[2:] 
Y_diff = Y2 - Y1

### Define the model architecture
model = Sequential()
model.add(Dense(12, activation='relu', input_shape=(6,)))
model.add(Dense(6, activation='relu'))
model.add(Dense(3, activation='linear'))  
model.compile(optimizer='adam', loss='mse', metrics=['mse'])
model.save_weights('data/generated/markov_testing_model.h5')

### Without conditioning
idx = np.arange(Y0.shape[0])
np.random.shuffle(idx)
Y_rand = Y0[idx]
Y_in = np.c_[Y_rand*0+1., Y1]
Y_out = Y_diff

Y_in_train, Y_in_test, Y_out_train, Y_out_test = timeseries_train_test_split(Y_in, Y_out)

### Scaling input and output data
Yinmax = (np.abs(Y_in_train)).max() # Parameters for scaling
Y_in_train, Y_in_test = Y_in_train/Yinmax, Y_in_test/Yinmax
Youtmax = (np.abs(Y_out_train)).max() # Parameters for scaling
Y_out_train, Y_out_test = Y_out_train/Youtmax, Y_out_test/Youtmax

history = model.fit(Y_in_train, Y_out_train, epochs=10, batch_size=100, validation_data=(Y_in_test, Y_out_test))
plt.plot(history.history['mse'])
plt.plot(history.history['val_mse'])

# Evaluation
Y_out_pred = model(Y_in_test).numpy()
mse_1 = mean_squared_error(flat_partial(Y_out_pred), flat_partial(Y_out_test))


### With conditioning
Y_in = np.c_[Y0, Y1]
Y_out = Y_diff
Y_in_train, Y_in_test, Y_out_train, Y_out_test = timeseries_train_test_split(Y_in, Y_out)

### Scaling input and output data
Yinmax = (np.abs(Y_in_train)).max() # Parameters for scaling
Y_in_train, Y_in_test = Y_in_train/Yinmax, Y_in_test/Yinmax
Youtmax = (np.abs(Y_out_train)).max() # Parameters for scaling
Y_out_train, Y_out_test = Y_out_train/Youtmax, Y_out_test/Youtmax


model.load_weights('data/generated/markov_testing_model.h5')
history = model.fit(Y_in_train, Y_out_train, epochs=10, batch_size=100, validation_data=(Y_in_test, Y_out_test))
plt.plot(history.history['mse'])
plt.plot(history.history['val_mse'])
#plt.show()

# Evaluation
Y_out_pred = model(Y_in_test).numpy()
mse_2 = mean_squared_error(flat_partial(Y_out_pred), flat_partial(Y_out_test))
print(mse_1, mse_2)

exit()

