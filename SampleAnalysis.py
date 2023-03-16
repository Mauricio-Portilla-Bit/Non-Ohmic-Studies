import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


# Función para estructurar los datos recibidos de txt a un dataframe
def structure_data():
    print("Clean Data")
    return 0


# Función para analizar eléctricamente una determinada muestra, ajustando los datos
def sample_analysis():
    print("Sample Analysis")


data = {"E": [], "J": []}

with open('Pruebas_electricas_1.txt') as f:
    for line in f.readlines():
        line_array = line.split()
        for i in range(len(line_array)):

            if i == 1:
                data["J"].append(float(line_array[i]))

            if i == 0:
                data["E"].append(float(line_array[i]))


df = pd.DataFrame(data)

plt.plot(df["E"], df["J"])
plt.scatter(df["E"], df["J"], c="r")
plt.grid()
plt.title("CAMPO ELÉCTRICO CONTRA DENSIDAD DE CORRIENTE")
plt.xlabel("CAMPO ELÉCTRICO (V/cm)")
plt.ylabel("DENSIDAD DE CORRIENTE (A/cm^2)")
plt.show()
