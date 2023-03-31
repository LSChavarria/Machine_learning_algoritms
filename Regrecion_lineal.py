import pandas as pd
import math
from matplotlib import pyplot as plt

class Datos:

    def __init__(self, head, list_x, list_y):
        self.head = head
        self.list_x = list_x
        self.list_y = list_y
        self.y_estimado = []
        self.prom_x = 0
        self.prom_y = 0
        self.theta_0 = 0
        self.theta_1 = 0
        self.r2 = 0
        self.mse = 0
        self.mae = 0
        self.rmse = 0
        self.x_train = []
        self.y_train = []
        self.x_test = []
        self.y_test = []
        self.set_train_n_test()

    def __repr__(self):
        return "Formula: \n\ty = " + str(self.theta_0) + " + " + str(self.theta_1) + "x \nEvaluacion:\n\tr2 = " + str(self.r2) + "\n\tr = " + str(self.r) + "\n\tmse = " + str(self.mse) + "\n\trmse = " + str(self.rmse) + "\n\tmae = " + str(self.mae)

    def set_train_n_test(self):
        for datos in enumerate(zip(list_x, list_y)):
            if datos[0] % 2 == 0:
                self.x_train.append(datos[1][0])
                self.y_train.append(datos[1][1])
            else:
                self.x_test.append(datos[1][0])
                self.y_test.append(datos[1][1])

def read_csv(archivo):
    list_x = []
    list_y = []
    datos = pd.read_csv(archivo, header = 0)
    head = list(datos.columns)
    datos = datos.sort_values(by = head[0])
    for listas in zip(list(datos[head[0]]), list(datos[head[1]])):
        try:
            float(listas[0])
            float(listas[1])
            list_x.append(float(listas[0]))
            list_y.append(float(listas[1]))
        except:
            print("Trash: ", listas)
    if len(list_x) % 2 != 0:
        list_x.pop()
        list_y.pop()
    return list_x, list_y, head

def promedio(x):
    return sum(x) / len(x)

def cal_theta_1(x, y, prom_x, prom_y):
    return sum([(i - prom_x) * (j - prom_y) for i, j in zip(x, y)]) / sum([(i - prom_x) ** 2 for i in x])

def cal_theta_0(prom_x, prom_y, theta_1):
    return prom_y - theta_1 * prom_x

def cal_y_estimado(theta_0, theta_1, x):
    return [theta_0 + theta_1 * i for i in x]

def cal_mse(y_real, y_estimado):
    return sum([(real - estimado) ** 2 for real, estimado in zip(y_real, y_estimado)]) / len(y_real)

def cal_mae(y_real, y_estimado):
    return sum([abs(estimado - real) for real, estimado in zip(y_real, y_estimado)]) / len(y_real)

def cal_r2(y_real, y_estimado, prom_y):
    return 1 - sum([(real - estimado) ** 2 for real, estimado in zip(y_real, y_estimado)]) / sum([(real - prom_y) ** 2 for real in y_real])

def regrecion_lineal(x, y, prom_x, prom_y):
    theta_1 = cal_theta_1(x, y, prom_x, prom_y)
    theta_0 = cal_theta_0(prom_x, prom_y, theta_1)
    return theta_0, theta_1

def evaluar_modelo(y_real, y_estimado, prom_y):
    mse = cal_mse(y_real, y_estimado)
    mae = cal_mae(y_real, y_estimado)
    r2 = cal_r2(y_real, y_estimado, prom_y)
    rmse = math.sqrt(mse)
    r = math.sqrt(r2)
    return mse, mae, r2, rmse, r

def graficar(datos):
    recta_x = [min(datos.list_x), max(datos.list_x)]
    recta_y = cal_y_estimado(datos.theta_0, datos.theta_1, recta_x)
    plt.plot(datos.list_x, datos.list_y, 'ro', recta_x, recta_y)
    plt.title("Estimacion " + datos.head[1] + " en base a " + datos.head[0])
    plt.xlabel(datos.head[0])
    plt.ylabel(datos.head[1])
    plt.show()

list_x, list_y, head = read_csv(r'C:\Users\sebas\Desktop\Facultad\Introduccion al Aprendizaje Automatico\Datasets\engine-size.csv')
datos = Datos(head, list_x, list_y)
datos.prom_x = promedio(datos.x_train)
datos.prom_y = promedio(datos.y_train)
datos.theta_1 = cal_theta_1(datos.x_train, datos.y_train, datos.prom_x, datos.prom_y)
datos.theta_0 = cal_theta_0(datos.prom_x, datos.prom_y, datos.theta_1)
datos.y_estimado = cal_y_estimado(datos.theta_0, datos.theta_1, datos.x_train)
datos.mse, datos.mae, datos.r2, datos.rmse, datos.r = evaluar_modelo(datos.y_train, datos.y_estimado, datos.prom_y)
print(datos)
graficar(datos)
