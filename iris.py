import pandas as pd
import tensorflow as tf


iris_df = pd.read_csv('iris.data', header=None)

iris_df.columns = ['sepal_length', 'sepal_width', 'petal_width', 'petal_length', 'label']
iris_df["label"] = iris_df["label"].map({'Iris-setosa':0, 'Iris-versicolor':1, 'Iris-virginica':2})

iris_df = iris_df.sample(frac=1.0, random_state=4321)
x = iris_df[["sepal_length", "sepal_width", "petal_width", "petal_length"]]
x = x - x.mean(axis=0)
y = tf.one_hot(iris_df["label"], depth=3)
print(y)


from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
import tensorflow.keras.backend as K

K.clear_session()
model = Sequential([
    Dense(32, activation='relu', input_shape=(4,)),
    Dense(16, activation='relu'),
    Dense(3, activation='softmax')
])

