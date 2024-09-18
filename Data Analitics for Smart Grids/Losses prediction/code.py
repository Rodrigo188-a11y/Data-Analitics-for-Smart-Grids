#!/usr/bin/env python
# coding: utf-8
# # Problem 2
import os
import pandas as pd
import numpy as np
from numpy import linalg as LA
import matplotlib.pyplot as plt

# <b>Parameters
noiseFactor = 0.0025  # noise
networkFactor = 100  # to change the characteristics of the network (Y)
PtestFactor = 3  # to obtain losses similar to the training data;

# <b>Import data (From Excel file)
direct = os.getcwd()
print(direct)
Info = np.array(pd.read_excel(r'C:\Users\linha\OneDrive\Documents\Rodrigo_Contreiras\Programação\Machine_learning'
                              r'\Analitica_redes_energia\Lab2\DASG_Prob2_new.xlsx', sheet_name='Info', header=None))

# Information about the slack bus
SlackBus = Info[0, 1]
print("Slack Bus: ", SlackBus, "\n")

# Network Information
Net_Info = np.array(pd.read_excel(r'C:\Users\linha\OneDrive\Documents\Rodrigo_Contreiras\Programação\Machine_learning'
                                  r'\Analitica_redes_energia\Lab2\DASG_Prob2_new.xlsx', sheet_name='Y_Data'))
print("Lines information (Admitances)\n", Net_Info, "\n")

# Power Information (train)
Power_Info = np.array(pd.read_excel(r'C:\Users\linha\OneDrive\Documents\Rodrigo_Contreiras\Programação\Machine_learning'
                                    r'\Analitica_redes_energia\Lab2\DASG_Prob2_new.xlsx', sheet_name='Load(t,Bus)'))
Power_Info = np.delete(Power_Info, [0], 1)
print("Power consumption information (time, Bus) - (Train)\n", Power_Info, "\n")

# Power Information (test)
Power_Test = np.array(pd.read_excel(r'C:\Users\linha\OneDrive\Documents\Rodrigo_Contreiras\Programação\Machine_learning'
                                    r'\Analitica_redes_energia\Lab2\DASG_Prob2_new.xlsx',
                                    sheet_name='Test_Load(t,Bus)'))
Power_Test = np.delete(Power_Test, [0], 1)
print("Power consumption information (time, Bus) - (Test)\n", Power_Test)

time = Power_Info.shape[0]
P = Power_Info
Ptest = Power_Test * PtestFactor

# Determine the number of Bus
nBus = max(np.max(Net_Info[:, 0]), np.max(Net_Info[:, 1]))

# Create the variable number of lines and the admitance matrix (Y)

nLines = Net_Info.shape[0]
Y = np.zeros((nBus, nBus), dtype=complex)

# Complete the Y matrix nad update the number of lines
for i in range(Net_Info.shape[0]):
    y_aux = Net_Info[i, 2].replace(",", ".")
    y_aux = y_aux.replace("i", "j")
    Y[Net_Info[i, 0] - 1, Net_Info[i, 0] - 1] = Y[Net_Info[i, 0] - 1, Net_Info[i, 0] - 1] + complex(
        y_aux) * networkFactor
    Y[Net_Info[i, 1] - 1, Net_Info[i, 1] - 1] = Y[Net_Info[i, 1] - 1, Net_Info[i, 1] - 1] + complex(
        y_aux) * networkFactor
    Y[Net_Info[i, 0] - 1, Net_Info[i, 1] - 1] = Y[Net_Info[i, 0] - 1, Net_Info[i, 1] - 1] - complex(
        y_aux) * networkFactor
    Y[Net_Info[i, 1] - 1, Net_Info[i, 0] - 1] = Y[Net_Info[i, 1] - 1, Net_Info[i, 0] - 1] - complex(
        y_aux) * networkFactor

# Remove the slack bus from the admitance matrix
Yl = np.delete(Y, np.s_[SlackBus - 1], axis=0)
Yl = np.delete(Yl, np.s_[SlackBus - 1], axis=1)

# Conductance Matrix
G = Yl.real

# Susceptance Matrix
B = Yl.imag

print("The admitance matrix Y is:\n", Y, "\n")
print("The conductance matrix G is\n", G, "\n")
print("The susceptance matrix B is\n", B, "\n")

# Create the vectors
C = np.zeros((nBus, nLines))
nLine_Aux = 0

# Determine the Incidence Matrix
for i in range(Y.shape[0]):
    for j in range(i + 1, Y.shape[1]):
        if np.absolute(Y[i, j]) != 0:
            C[i, nLine_Aux] = 1
            C[j, nLine_Aux] = -1
            nLine_Aux = nLine_Aux + 1

        # Remove the slack bus from the matrix
Cl = np.delete(C, np.s_[SlackBus - 1], axis=0)

print("The incidence matrix C (nBus,nLines) is:\n", Cl)

# <b>Definition of Matrix Gij (Diagonal and vector)

# Create the vectors
Gv = np.zeros((1, nLines))
Gd = np.zeros((nLines, nLines))
nLine_Aux = 0

# Determine the Incidence Matrix
for i in range(Y.shape[0]):
    for j in range(i + 1, Y.shape[1]):
        if np.absolute(Y[i, j]) != 0:
            Gv[0, nLine_Aux] = -np.real(Y[i, j])  # Information about the lines condutance [Vector]
            Gd[nLine_Aux, nLine_Aux] = -np.real(Y[i, j])  # Information about the lines condutance [Diagonal in matrix]
            nLine_Aux = nLine_Aux + 1

print("Gij_Diag:\n", Gd)

# Matrix creation
teta = np.zeros((nBus - 1, time))
grau = np.zeros((nLines, time))
PL = np.zeros(time)
PL2 = np.zeros(time)
PT = np.zeros(time)
rLoss = np.zeros(time)

# Losses
alfa = np.dot(np.dot(np.dot(np.dot(np.linalg.inv(B), Cl), Gd), np.transpose(Cl)), np.linalg.inv(B))  # Used in
# Equation (15)

for m in range(time):
    PL[m] = np.dot(P[m, :], np.dot(alfa, np.transpose(P[m, :])))  # Power Losses using equation (15)

    teta[:, m] = np.dot(np.linalg.inv(B), np.transpose(P[m, :]))  # Voltage angle (Teta). Equation (14)

    grau[:, m] = np.dot(np.transpose(Cl), teta[:, m])  # Voltage angle difference (Teta ij)

    PL2[m] = np.dot(2 * Gv, 1 - np.cos(grau[:, m]))  # Power Losses using equation (13)

    PT[m] = np.sum([P[m, :]])  # Total Power

    rLoss[m] = np.divide(PL2[m], PT[m])  # Power Losses (%)

print("Total Power consumption:\n", PT, "\n")
print("Power Losses obtained using the Theta:\n", PL2, "\n")
print("Power Losses obtained without using the Theta:\n", PL, "\n")


def somatorio(x):
    if x == 1:
        return 1
    else:
        return x + somatorio(x - 1)


def plot(PLosses_compare, PLosses_):
    plt.plot(np.arange(len(PLosses_compare)), PLosses_compare, drawstyle='steps-post',
             label='Power Loss using equation 16')
    plt.plot(np.arange(len(PLosses_)), PLosses_, drawstyle='steps-post', label='Power Loss using equation 13')
    plt.grid(axis='x', color='0.95')
    plt.legend()
    plt.title('Losses Plot')
    plt.xlabel('Time')
    plt.ylabel('Losses [pu]')
    plt.xlim([0, 12])
    plt.show()

    abserror = abs(np.subtract(PLosses_compare, PLosses_))
    relerror = np.divide(abserror, PLosses_) * 100
    plt.plot(np.arange(len(relerror)), relerror, drawstyle='steps-post')

    plt.grid(axis='x', color='0.95')
    plt.legend()
    plt.title('Losses Plot')
    plt.xlabel('Time')
    plt.ylabel('Error [%]')
    plt.xlim([0, 12])
    plt.show()


def plot_bus_size():
    size = 100
    var = np.zeros(size)
    for a in range(size):
        if a == 0:
            var[a] = 0
        var[a] = var[a - 1] + a
        print(var[a])
    plt.plot(np.arange(size), var, color='blue', alpha=0.4)
    plt.grid(axis='x', color='0.95')
    plt.legend()
    plt.title('Power Injection Matrix Size (X)')
    plt.xlabel('Number of Buses')
    plt.ylabel('Matrix size')
    plt.show()
    exit()


### Training data
sizeX = somatorio(nBus - 1)
X = np.zeros((time, sizeX))

for i in range(time):
    auxSize = nBus - 1
    minimum = 0
    k = 0
    for j in range(sizeX):
        if k == minimum:
            X[i, j] = P[i, k] ** 2
        else:
            X[i, j] = P[i, minimum] * P[i, k] * 2
        k += 1
        if k == auxSize:
            minimum += 1
            k = minimum

Beta = np.dot(np.linalg.inv(np.dot(np.transpose(X), X)), np.dot(np.transpose(X), PL2))

# print("Vetor beta: \n", Beta)
Yaux = np.dot(X, Beta)
mu, sigma = 0, 0.0025
s = np.zeros(len(Yaux))

for i in range(len(Yaux)):
    s[i] = (1 + np.random.normal(mu, sigma, 1))

Plossesfinal_train = Yaux * s
plot(Plossesfinal_train, PL2)

# Testing data
for i in range(time):
    auxSize = nBus - 1
    minimum = 0
    k = 0
    for j in range(sizeX):
        if k == minimum:
            X[i, j] = Ptest[i, k] ** 2
        else:
            X[i, j] = Ptest[i, minimum] * Ptest[i, k] * 2
        k += 1
        if k == auxSize:
            minimum += 1
            k = minimum

# print("Vetor beta: \n", Beta)
Yaux = np.dot(X, Beta)
Plossesfinal_test = Yaux * s

# Losses
alfa = np.dot(np.dot(np.dot(np.dot(np.linalg.inv(B), Cl), Gd), np.transpose(Cl)), np.linalg.inv(B))  # Used in
# Equation (15)

for m in range(time):
    PL[m] = np.dot(Ptest[m, :], np.dot(alfa, np.transpose(Ptest[m, :])))  # Power Losses using equation (15)

    teta[:, m] = np.dot(np.linalg.inv(B), np.transpose(Ptest[m, :]))  # Voltage angle (Teta). Equation (14)

    grau[:, m] = np.dot(np.transpose(Cl), teta[:, m])  # Voltage angle difference (Teta ij)

    PL2[m] = np.dot(2 * Gv, 1 - np.cos(grau[:, m]))  # Power Losses using equation (13)

    PT[m] = np.sum([Ptest[m, :]])  # Total Power

    rLoss[m] = np.divide(PL2[m], PT[m])  # Power Losses (%)

print("Total Power consumption:\n", PT, "\n")
print("Power Losses obtained using the Theta:\n", PL2, "\n")
print("Power Losses obtained without using the Theta:\n", PL, "\n")

plot(Plossesfinal_test, PL2)
