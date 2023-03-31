import pandas as pd
import numpy as np

def fit(x, y):
  return np.matmul(np.matmul(np.linalg.inv(np.matmul(x.T, x)), x.T), y)

def predict(theta, x):
  return np.matmul(theta, x)

def cal_mse(y_real, y_estimado):
  return sum([(real - estimado) ** 2 for real, estimado in zip(y_real, y_estimado)]) / len(y_real)

def cal_mae(y_real, y_estimado):
  return sum([abs(estimado - real) for real, estimado in zip(y_real, y_estimado)]) / len(y_real)

df = pd.read_csv(r"C:\Users\sebas\Desktop\Facultad\Introduccion al Aprendizaje Automatico\Datasets\FuelConsumption.csv")
df['One'] = 1

msk = np.random.rand(len(df)) < 0.8
train = df[msk]
test = df[~msk]

train_x = np.asanyarray(train[['One','ENGINESIZE','CYLINDERS','FUELCONSUMPTION_COMB']])
train_y = np.asanyarray(train[['CO2EMISSIONS']])

theta = fit(train_x, train_y)

# Evaluar el modelo
test_x = np.asanyarray(test[['One','ENGINESIZE','CYLINDERS','FUELCONSUMPTION_COMB']])
test_y = np.asanyarray(test[['CO2EMISSIONS']])

test_y_hat = predict(theta.T, test_x.T)

mse=cal_mse(test_y, test_y_hat.T)
mae=cal_mae(test_y, test_y_hat.T)

print("Theta = ", theta)
print("MSE = ", mse)
print("MAE = ", mae)
