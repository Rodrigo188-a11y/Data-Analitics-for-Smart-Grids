import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
from scipy.stats import norm


def durbin_watson(residuals):
    """
    Compute the Durbin-Watson test statistic for residuals.

    Args:
        residuals (numpy.ndarray): Array of residuals.

    Returns:
        float: Durbin-Watson test statistic.
    """
    diff = np.diff(residuals)
    numerator = np.sum(diff ** 2)
    denominator = np.sum(residuals ** 2)
    dw = numerator / denominator
    return dw


networkFactor = 100  # To change the characteristics of the network (Y)
cosPhi = 0.95  # Value of teta
time = 24  # Training Period
timeForecast = 12  # Test Period

# Import data (From Excel file)

Info = np.array(pd.read_excel(r'C:\Users\linha\OneDrive\Documents\Rodrigo_Contreiras\Programação\Machine_learning'
                              r'\Analitica_redes_energia\Lab4\DASG_Prob2_new.xlsx', sheet_name='Info', header=None))
# Information about the slack bus
SlackBus = Info[0, 1]
print("Slack Bus: ", SlackBus, "\n")

# Network Information
Net_Info = np.array(pd.read_excel(r'C:\Users\linha\OneDrive\Documents\Rodrigo_Contreiras\Programação\Machine_learning'
                                  r'\Analitica_redes_energia\Lab4\DASG_Prob2_new.xlsx', sheet_name='Y_Data'))
print("Lines information (Admitances)\n", Net_Info, "\n")

# Power Information (Train)
Power_Info = np.array(pd.read_excel(r'C:\Users\linha\OneDrive\Documents\Rodrigo_Contreiras\Programação\Machine_learning'
                                    r'\Analitica_redes_energia\Lab4\DASG_Prob2_new.xlsx', sheet_name='Load(t,Bus)'))
Power_Info = np.delete(Power_Info, [0], 1)
print("Power consumption information (time, Bus)\n", Power_Info, "\n")

# Power Information (Test)
Power_Test = np.array(pd.read_excel(r'C:\Users\linha\OneDrive\Documents\Rodrigo_Contreiras\Programação\Machine_learning'
                                    r'\Analitica_redes_energia\Lab4\DASG_Prob2_new.xlsx',
                                    sheet_name='Test_Load(t,Bus)'))
Power_Test = np.delete(Power_Test, [0], 1)
print("Power consumption information (time, Bus)\n", Power_Test, "\n")

P = np.dot(-Power_Info, np.exp(complex(0, 1) * np.arccos(cosPhi)))
I = np.conj(P[2, :])

# Power Information (Test)
i19_2_big = np.array(pd.read_excel(r'C:\Users\linha\OneDrive\Documents\Rodrigo_Contreiras\Programação\Machine_learning'
                                   r'\Analitica_redes_energia\Lab4\IEEE33-TimeSeries.xlsx',
                                   sheet_name='IEEE33(I19-2)', header=None)).flatten()  # current that we want to
# measure
i19_2_big = i19_2_big.astype(complex)

i1w_big = np.array(pd.read_excel(r'C:\Users\linha\OneDrive\Documents\Rodrigo_Contreiras\Programação\Machine_learning'
                                 r'\Analitica_redes_energia\Lab4\IEEE33-TimeSeries.xlsx',
                                 sheet_name='IEEE33(W22)', header=None)).flatten()  # power that is injected by the
# wind farm
i1w_big = i1w_big.astype(complex)

sum2_big = np.array(pd.read_excel(r'C:\Users\linha\OneDrive\Documents\Rodrigo_Contreiras\Programação\Machine_learning'
                                  r'\Analitica_redes_energia\Lab4\IEEE33-TimeSeries.xlsx',
                                  sheet_name='IEEE33(SUM)', header=None)).flatten()  # sum of all loads of big network
sum2_big = sum2_big.astype(complex)
i12_big = np.array(pd.read_excel(r'C:\Users\linha\OneDrive\Documents\Rodrigo_Contreiras\Programação\Machine_learning'
                                 r'\Analitica_redes_energia\Lab4\IEEE33-TimeSeries.xlsx',
                                 sheet_name='IEEE33(I1-2)', header=None)).flatten()  # current between line 1-2
i12_big = i12_big.astype(complex)
i32_big = np.array(pd.read_excel(r'C:\Users\linha\OneDrive\Documents\Rodrigo_Contreiras\Programação\Machine_learning'
                                 r'\Analitica_redes_energia\Lab4\IEEE33-TimeSeries.xlsx',
                                 sheet_name='IEEE33(I3-2)', header=None)).flatten()  # current between line 3-2
i32_big = i32_big.astype(complex)

i20_19_big = np.array(pd.read_excel(r'C:\Users\linha\OneDrive\Documents\Rodrigo_Contreiras\Programação\Machine_learning'
                                    r'\Analitica_redes_energia\Lab4\IEEE33-TimeSeries.xlsx',
                                    sheet_name='IEEE33(I20-19)', header=None)).flatten()  # current between line 20-19
i20_19_big = i20_19_big.astype(complex)
i12_radial = np.array(pd.read_excel(r'C:\Users\linha\OneDrive\Documents\Rodrigo_Contreiras\Programação\Machine_learning'
                                    r'\Analitica_redes_energia\Lab4\IEEE33-TimeSeries.xlsx',
                                    sheet_name='RADIAL(I1-2)', header=None)).flatten()  # current between line 1-2
i12_radial = i12_radial.astype(complex)
i1w_radial = np.array(pd.read_excel(r'C:\Users\linha\OneDrive\Documents\Rodrigo_Contreiras\Programação\Machine_learning'
                                    r'\Analitica_redes_energia\Lab4\IEEE33-TimeSeries.xlsx',
                                    sheet_name='RADIAL(W1)', header=None)).flatten()  # current between line 1-2
i1w_radial = i1w_radial.astype(complex)

# <b>Admittance Matrix(<i>Y</i>); Conductance Matrix(<i>G</i>); Susceptance Matrix(<i>B</i>)

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

# <b> Errors Definition

# Random values considering a normal distribution

np.random.seed(50)
e1 = np.random.randn(time + timeForecast) * 0.5  # Errors associated to Wind Generation
e = np.random.randn(time + timeForecast) * 0.25  # Errors associated to Power Injection (Consumption)

# To obtain the same values of lecture notes, we should use the following errors

e1 = [0.2878, 0.0145, 0.5846, -0.0029, -0.2718, -0.1411,
      -0.2058, -0.1793, -0.9878, -0.4926, -0.1480, 0.7222,
      -0.3123, 0.4541, 0.9474, -0.1584, 0.4692, 1.0173,
      -0.0503, 0.4684, -0.3604, 0.4678, 0.3047, -1.5098,
      -0.5515, -0.5159, 0.3657, 0.7160, 0.1407, 0.5424,
      0.0409, 0.0450, 0.2365, -0.3875, 1.4783, -0.8487]

e = [-0.0106, 0.0133, 0.2226, 0.2332, 0.1600, -0.0578,
     -0.2293, -0.2843, -0.2732, -0.1203, -0.1757, -0.1891,
     0.1541, -0.0093, -0.1691, 0.2211, -0.4515, -0.1786,
     -0.2031, -0.3634, -0.1105, -0.1413, -0.5900, -0.1729,
     -0.0810, -0.0023, -0.0556, 0.1858, -0.0324, -0.1071,
     -0.0845, -0.0743, -0.0479, -0.0870, -0.1834, -0.1432]

# <b> Determine the wind generation and the load flow in <i>I<sub>12

# Creation of Matrix
II = np.zeros((nBus - 1, time + timeForecast), dtype=complex)
i12 = np.zeros(time + timeForecast)
i23 = np.zeros(time + timeForecast)
i31 = np.zeros(time + timeForecast)
i1w = np.zeros(time + timeForecast)
voltages = np.zeros((nBus - 1, time + timeForecast), dtype=complex)

# Initializing the process of data generation
II[:, 0] = I  # Power Injections
voltages[:, 0] = 1 + np.dot(np.linalg.inv(Yl), I)  # Compute the voltages
v = 1 + np.dot(np.linalg.inv(Yl), I)
i12[0] = np.absolute(np.dot(Y[0, 1], v[0] - v[1]))  # Current I12 in period t=0
i23[0] = np.absolute(np.dot(Y[2, 3], v[1] - v[2]))  # Current I23 in period t=0
i31[0] = np.absolute(np.dot(Y[0, 2], v[2] - v[0]))  # Current I31 in period t=0
i1w[0] = np.real(I[0])  # Injection in bus 1 (Wind) in period t=0

# Process of data generation
for t in range(time + timeForecast - 1):
    II[:, t + 1] = 0.95 * II[:, t] + e[t]  # Power injection based on previous periods and in the errors. T
    # the values are more or less related considering
    # the value of 0.95. This value can change between 0 and 1.
    i1w[t + 1] = 0.75 * i1w[t] + e1[t]  # Wind power based on the previous periods
    II[0, t + 1] = i1w[t + 1] + complex(0, np.imag(II[0, t + 1]))  # Add the Wind generation
    v = 1 + np.dot(np.linalg.inv(Yl), II[:, t + 1])  # Compute the voltages
    voltages[:, t + 1] = v
    I12 = np.dot(-Y[0, 1], v[0] - v[1])  # Compute the load flow in line 1-2 (Complex)
    i12[t + 1] = np.absolute(I12) * np.sign(np.real(I12))  # Compute the load flow in line 1-2 (RMS with signal)
    I23 = np.dot(-Y[2, 3], v[1] - v[2])
    i23[t + 1] = np.absolute(I23) * np.sign(np.real(I23))
    I31 = np.dot(-Y[0, 2], v[2] - v[0])
    i31[t + 1] = np.absolute(I31) * np.sign(np.real(I31))

sub = np.subtract(voltages[1, :], voltages[0, :])

print('The power injection in Bus 1 is:\n', II[0, :])
print('\nThe power flow in Line 1-2 is:\n', i12)

####  BIG NETWORK ####################################
# <b>Ordinary Least Squares OLS regression
#### Pinj1
i1w_big = np.real(i1w_big)
i19_2_big = np.absolute(i19_2_big) * np.sign(np.real(i19_2_big))

AA = np.ones((time, 2))  # Vector Xt with ones
AA[:, 1] = i1w_big[0:time]  # Vector Xt with ones in first column and wind injection in column 2
AATransp = np.transpose(AA)
beta = np.dot(np.dot(np.linalg.inv(np.dot(AATransp, AA)), AATransp), i19_2_big[0:time])  # Beta values
I12f1 = beta[0] + np.dot(beta[1], i1w_big[time:time + timeForecast])  # using Ordinary least Squares (OLS)
rr1 = i19_2_big[0:time] - beta[0] - np.dot(beta[1], i1w_big[0:time])
D1 = durbin_watson(rr1)

#### Autorregration with Loads sum
# Definition of Matrix Xt
X = np.ones((time, 2))  # Vector Xt with ones
X[:, 1] = sum2_big[0:time]  # Vector Xt with previous time current i12 injection
XTransp = np.transpose(X)
beta_sum_loads = np.dot(np.dot(np.linalg.inv(np.dot(XTransp, X)), XTransp), i19_2_big[0:time])
I12f2 = beta_sum_loads[0] + np.dot(beta_sum_loads[1], sum2_big[time:time + timeForecast])  # using Ordinary least
# Squares
rr2 = i19_2_big[0:time] - beta_sum_loads[0] - np.dot(beta_sum_loads[1], sum2_big[0:time])

#### Autorregration with Loads sum + P1inj
# Definition of Matrix Xt
X = np.ones((time, 3))  # Vector Xt with ones
X[:, 1] = i1w_big[0:time]  # Vector Xt with ones in first column and wind injection in column 2
X[:, 2] = sum2_big[0:time]  # Vector Xt with previous time current i12 injection
XTransp = np.transpose(X)
beta_sum_loads = np.dot(np.dot(np.linalg.inv(np.dot(XTransp, X)), XTransp), i19_2_big[0:time])
I12f3 = beta_sum_loads[0] + np.dot(beta_sum_loads[1], i1w_big[time:time + timeForecast]) + np.dot(beta_sum_loads[2],
                                                                                                  sum2_big[
                                                                                                  time:time + timeForecast])  # using Ordinary least
# Squares
rr3 = i19_2_big[0:time] - beta_sum_loads[0] - np.dot(beta_sum_loads[1], i1w_big[0:time]) - np.dot(beta_sum_loads[2],
                                                                                                  sum2_big[0:time])

#### Autorregration with Loads sum
# Definition of Matrix Xt
i12_big = np.real(i12_big)
i32_big = np.real(i32_big)
i20_19_big = np.real(i20_19_big)
currents = np.add(i32_big, i20_19_big)
currents = np.add(currents, i12_big)
X = np.ones((time, 2))  # Vector Xt with ones
X[:, 1] = currents[0:time]  # Vector Xt with previous time current i12 injection
XTransp = np.transpose(X)
beta_sum_loads = np.dot(np.dot(np.linalg.inv(np.dot(XTransp, X)), XTransp), i19_2_big[0:time])
I12f4 = beta_sum_loads[0] + np.dot(beta_sum_loads[1], currents[time:time + timeForecast])  # using Ordinary least
# Squares
rr4 = i19_2_big[0:time] - beta_sum_loads[0] - np.dot(beta_sum_loads[1], currents[0:time])

# Plot initial data

# Define the plots
x = range(0, time + timeForecast)
xx = range(0, time)
xxx = range(time, time + timeForecast)
yy1 = i19_2_big[time:time + timeForecast]
yy2 = I12f1
yy3 = I12f2

plt.plot(xxx, yy1, color='red', label='Measured')
plt.plot(xxx, yy2, color='black', linestyle='dashed', marker='o', label='P1_inj')
plt.plot(xxx, yy3, color='red', linestyle='-.', marker='*', label='Sum Loads')
plt.plot(xxx, I12f3, color='blue', linestyle='-.', marker='*', label='Sum Loads + Pinj')
plt.plot(xxx, I12f4, color='orange', linestyle='-.', marker='*', label='Sum currents')
plt.xlim((24, 36))
plt.legend()
plt.xlabel('Time stamp [h]')
plt.ylabel('Current I_12')
plt.show()

plt.plot(xx, rr1, 'C0*', linestyle='-.', label='P1_inj')
plt.plot(xx, rr2, 'C1*', linestyle='-.', label='Sum Loads')
plt.plot(xx, rr3, 'C2*', linestyle='-.', label='Sum Loads + Pinj')
plt.plot(xx, rr4, 'C3*', linestyle='-.', label='Sum currents')
plt.xlim((0, 23))
plt.legend()
plt.xlabel('Time stamp [h]')
plt.ylabel("Residuals")
plt.show()

plt.plot(x, i1w_big, color='red', label='P1_inj')
plt.plot(x, sum2_big, color='black', linestyle='-.', label='Sum Loads')
plt.xlim((0, 36))
plt.legend()
plt.xlabel('Time stamp [h]')
plt.ylabel('Active power [pu]')
plt.show()

####  RADIAL NETWORK  #########################
# <b>Ordinary Least Squares OLS regression
#### Pinj1
i1w_radial = np.real(i1w_radial)
i12_radial = np.absolute(i12_radial) * np.sign(np.real(i12_radial))

AA = np.ones((time, 2))  # Vector Xt with ones
AA[:, 1] = i1w_radial[0:time]  # Vector Xt with ones in first column and wind injection in column 2
AATransp = np.transpose(AA)
beta = np.dot(np.dot(np.linalg.inv(np.dot(AATransp, AA)), AATransp), i12_radial[0:time])  # Beta values
I12f5 = beta[0] + np.dot(beta[1], i1w_radial[time:time + timeForecast])  # using Ordinary least Squares (OLS)
rr5 = i12_radial[0:time] - beta[0] - np.dot(beta[1], i1w_radial[0:time])

plt.plot(xxx, i12_radial[time: time + timeForecast], color='red', label='Measured')
plt.plot(xxx, I12f5, color='black', linestyle='-.', label='P_inj')
plt.xlim((24, 36))
plt.legend()
plt.xlabel('Time stamp [h]')
plt.ylabel('Current I_12')
plt.show()

plt.plot(xx, rr5, 'C0*', linestyle='-.', label='P1_inj')
plt.legend()
plt.xlabel('Time stamp [h]')
plt.ylabel("Residuals")
plt.show()
