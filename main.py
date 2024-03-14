import tensorflow as tf
import numpy as np 

celsius = np.array([-40, -10, 0, 8, 15, 22, 38], dtype=float)
fahrenheit = np.array([-40, 14, 32, 46, 59, 72, 100], dtype=float)

layer = tf.keras.layers.Dense(units=1, input_shape=[1])
model = tf.keras.Sequential([layer])

model.compile(
    optimizer = tf.keras.optimizers.Adam(0.1),
    loss = 'mean_squared_error'
)

results = model.fit(celsius, fahrenheit, epochs=1000, verbose=False)

import matplotlib.pyplot as plt

plt.xlabel("# epochs")
plt.ylabel("Loss")
plt.plot(results.history["loss"])

res = model.predict([100.0])

print("the result is " + str(res) + " farenheit")

