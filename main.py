import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

reloaded = tf.keras.models.load_model('rssi_to_distance')
data = {'rssi': -58.333333333333336, 'snr': 12.166666666666666, 'tx_power': 15.0}
data = pd.DataFrame(data, index=[0])
print(reloaded.predict(data)[0][0])

