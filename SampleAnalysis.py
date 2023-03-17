import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score


# Función para estructurar los datos recibidos de txt a un dataframe
def structure_data(file):
    data = {"E": [], "J": []}

    with open(file) as f:
        for line in f.readlines():
            line_array = line.split()
            for i in range(len(line_array)):

                if i == 1:
                    data["J"].append(float(line_array[i]))

                if i == 0:
                    data["E"].append(float(line_array[i]))

    return pd.DataFrame(data)


# Función para analizar eléctricamente una determinada muestra, ajustando los datos al modelo
def sample_analysis(df):

    x_inicial = 0
    x_prerruptura = 10
    x_ruptura = 20
    x_final = 70

    df = df.iloc[x_inicial:x_final]
    #df_lineal = df.iloc[x_inicial:x_prerruptura]
    #df_transicion = df.iloc[x_prerruptura:x_ruptura]
    #df_exponencial = df.iloc[x_ruptura:x_final]


    # Ajuste de curva lineal : Evaluación para determinar el inicio del punto de ruptura
    # El ajuste se explora fijando un punto en la izquierda y recorriendo a la derecha
    # la operación termina una vex que el score alcanza un valor adecuado (0.95)

    err_lineal = 100
    for i in range(len(df) - 1):

        # Generar el modelo lineal
        x_lineal = np.array(df[x_inicial: x_final - i]["E"]).reshape((-1, 1))
        y_lineal = np.array(df[x_inicial: x_final - i]["J"])
        lineal_reg = LinearRegression().fit(x_lineal, y_lineal)

        # Predecir la recta y evaluar el error recorriendo la curva a la derecha
        y_lineal_fit = lineal_reg.predict(x_lineal)
        score = "{0:.2f}".format(lineal_reg.score(x_lineal, y_lineal))
        lineal_coefficient = lineal_reg.coef_[0]
        lineal_intercept = lineal_reg.intercept_

        if float(score) >= 0.80:
            x_prerruptura = x_final - i
            break

    print("CAMPO ELÉCTRICO LINEAL: ", x_prerruptura)


    # Ajuste de curva exponencial : En este caso, el ajuste de curva se realiza en el sentido contrario,
    # Fijando el punto al final de la gráfica y recortando los datos de la izquierda hasta llegar a una
    # evaluación adecuada (0.90). Es necesario establecer que no se considera el efecto joule

    err_exponecial = 100
    for i in range(x_prerruptura + 1, len(df) - 1):

        # Calcular el valor de alfa
        alfa = ((np.log10(df["J"][i + 1]) - np.log10(df["J"][i])) / (np.log10(df["E"][i + 1]) - np.log10(df["E"][i])))
        x_exponential = df[i:x_final]["E"]
        y_exponential = df[i:x_final]["J"]

        log_y = np.log(y_exponential)

        fit = np.polyfit(x_exponential, log_y, 1)
        y_exponential_fit = np.exp(fit[1]) * np.exp(fit[0]*x_exponential)

        score = r2_score(y_exponential, y_exponential_fit)

        if score > 0.98:
            x_ruptura = i
            break

    # Definición de la zona de transición


    # Graficar los ajustes del modelo
    plt.scatter(df["E"], df["J"], c="r")
    plt.plot(x_exponential, y_exponential_fit, c="b")
    plt.plot(x_lineal, y_lineal_fit, c="b")
    plt.grid()
    plt.title("CAMPO ELÉCTRICO CONTRA DENSIDAD DE CORRIENTE")
    plt.xlabel("CAMPO ELÉCTRICO (V/cm)")
    plt.ylabel("DENSIDAD DE CORRIENTE (A/cm^2)")
    plt.show()










# Función para graficar la data que se recibe
def graph_data(df):
    plt.plot(df["E"], df["J"])
    plt.scatter(df["E"], df["J"], c="r")
    plt.grid()
    plt.title("CAMPO ELÉCTRICO CONTRA DENSIDAD DE CORRIENTE")
    plt.xlabel("CAMPO ELÉCTRICO (V/cm)")
    plt.ylabel("DENSIDAD DE CORRIENTE (A/cm^2)")
    plt.show()


file = 'Pruebas_electricas_1.txt'
df = structure_data(file)
#graph_data(df)
sample_analysis(df)


