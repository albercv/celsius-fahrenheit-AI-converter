import tensorflow as tf
import numpy as np

#Create Model
celsisus = np.array([-40, -10, 0, 8, 15, 22, 38], dtype=float)
fahrenheit = np.array([-40, 14, 32, 46, 59, 72, 100], dtype=float)

layer = tf.keras.layers.Dense(units=1, input_shape=[1])
model = tf.keras.Sequential([layer])

model.compile(optimizer=tf.keras.optimizers.Adam(0.1), loss='mean_squared_error')
print("Starting training")

history = model.fit(celsisus, fahrenheit, epochs=1000, verbose=False)
print("Training finished")

# Learning curves
import matplotlib.pyplot as plt
plt.xlabel('Epoch Number')
plt.ylabel('Loss Magnitude')
plt.plot(history.history['loss'])

#Predict
results = []

data = [10.0, 15.0, 20.0, 25.0, 30.0]
datos_para_predecir = np.array(data).reshape(-1, 1)
results = model.predict(data)

#[1:50, 2:59, 3:68, 4:77, 5:86] -> expected results

print(layer.get_weights())

print(results)