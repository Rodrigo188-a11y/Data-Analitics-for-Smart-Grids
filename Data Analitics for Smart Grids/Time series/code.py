import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


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

P = np.dot(-Power_Info, np.exp(complex(0, 1) * np.arccos(cosPhi)))
I = np.conj(P[2, :])

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
i1w = np.zeros(time + timeForecast)

# Initializing the process of data generation
II[:, 0] = I  # Power Injections
v = 1 + np.dot(np.linalg.inv(Yl), I)
i12[0] = np.absolute(np.dot(Y[0, 1], v[0] - v[1]))  # Current I12 in period t=0
i1w[0] = np.real(I[0])  # Injection in bus 1 (Wind) in period t=0

# Process of data generation
for t in range(time + timeForecast - 1):
    II[:, t + 1] = 0.95 * II[:, t] + e[t]  # Power injection based on previous periods and in the errors. T
    # the values are more or less related considering
    # the value of 0.95. This value can change between 0 and 1.
    i1w[t + 1] = 0.75 * i1w[t] + e1[t]  # Wind power based on the previous periods
    II[0, t + 1] = i1w[t + 1] + complex(0, np.imag(II[0, t + 1]))  # Add the Wind generation
    v = 1 + np.dot(np.linalg.inv(Yl), II[:, t + 1])  # Compute the voltages
    I12 = np.dot(-Y[0, 1], v[0] - v[1])  # Compute the load flow in line 1-2 (Complex)
    i12[t + 1] = np.absolute(I12) * np.sign(np.real(I12))  # Compute the load flow in line 1-2 (RMS with signal)

print('The power injection in Bus 1 is:\n', II[1, :])
print('\nThe power flow in Line 1-2 is:\n', i12)

# <b>Ordinary Least Squares OLS regression

# Define the OLS regression relating the Current I12 with the Pinjection I1. See Equation (30) in the lecture notes
AA = np.ones((time, 2))  # Vector Xt with ones
AA[:, 1] = i1w[0:time]  # Vector Xt with ones in first column and wind injection in column 2
AATransp = np.transpose(AA)
beta = np.dot(np.dot(np.linalg.inv(np.dot(AATransp, AA)), AATransp), i12[0:time])  # Beta values
print("AAA")
print("The value of Betas, using OLS, are:\n", beta)

# <b>Plot initial data

# Define the plots
x = range(time)
yy1 = i12[0:time]
yy2 = i1w[0:time]
rss_1 = beta[0] + np.dot(beta[1], i1w[0:time])  # OLS regresion line
yy3 = rss_1
yy4 = i12[0:time] - beta[0] - np.dot(beta[1], i1w[0:time])

# First Graph (Pinjection in bus 1 and Current I12)
fig, ax1 = plt.subplots()
ax2 = ax1.twinx()
ax1.plot(x, yy2, 'g-')
ax2.plot(x, yy1, 'b-')
ax1.set_xlabel('Time Stamp [h]')
ax1.set_ylabel('Injection P_1', color='g')
ax2.set_ylabel('Current I_12', color='b')
ax1.set_ylim([-2, 2])
ax2.set_ylim([-1, 1])
plt.xlabel("Time Stamp [h]")

plt.xlim([0, 23])
plt.show()

# Second Graph (Relation I1 vs I12 and OLS regression)
plt.plot(yy2, yy1, 'C1o', label='P1W')
plt.plot(yy2, yy3, label='Regression line')
plt.legend()
plt.xlabel("Injection P_1")
plt.ylabel("Current I_12")
plt.show()

# Third Graph (Residuals - Difference between the real current I12 and the one obtained by OLS regression)
plt.plot(x, yy4, 'C0o', label='Residuals')
plt.legend()
plt.xlabel("Time Stamp [h]")
plt.ylabel("Residuals")
plt.show()

# <b>Durbin-Watson statistic

# - The Durbin Watson statistic is a test for autocorrelation in a data set. - The DW statistic always has a value
# between zero and 4.0. - A value of 2.0 means there is no autocorrelation detected in the sample. Values from zero
# to 2.0 indicate positive autocorrelation and values from 2.0 to 4.0 indicate negative autocorrelation.
# 
# <sub>https://www.investopedia.com/terms/d/durbin-watson-statistic.asp

D = durbin_watson(yy4)
ro = 1 - D / 2
print("The value of Durdin-Watson (DW) is:", D)
print("The value of rho is: ", ro)

# <b>Cochrane Orcutt

res_1 = i12[0:time] - rss_1
for k in range(3):  # According to "Applied Linear Statistical Models" if the OC methot does not converge
    # in three iterations, we should use other method
    r2 = res_1[0:time - 1]
    r1 = res_1[1:time]
    ro = 0.97 * np.dot(np.dot((np.dot(np.transpose(r2), r2)) ** (-1), np.transpose(r2)),
                       r1)  # Estimate Rho based on(28)
    i1w_s = i1w[1:time] - np.dot(ro, i1w[0:time - 1])  # Transform yt*=yt
    i12_s = i12[1:time] - np.dot(ro, i12[0:time - 1])  # Transform xt*=Xt
    B = np.ones((time - 1, 2))
    B[:, 1] = i1w_s
    b_s = np.dot(np.dot(np.linalg.inv(np.dot(np.transpose(B), B)), np.transpose(B)),
                 np.transpose(i12_s))  # Regress yt* over xt*

    b_s[0] = np.divide(b_s[0], 1 - ro)  # Transform Beta_0
    rss_s = b_s[0] + np.dot(b_s[1], i1w_s[0:time - 1])  # Update residuals
    rss_2 = b_s[0] + np.dot(b_s[1], i1w[0:time])
    res_2 = i12[0:time] - rss_2
    res_1 = res_2[:]
b_ss = b_s

# <b>Forecast Day-ahead (Current I_12)

I12f1 = beta[0] + np.dot(beta[1], i1w[time:time + timeForecast])  # using Ordinary least Squares (OLS)
I12f2 = b_ss[0] + np.dot(b_ss[1], i1w[time:time + timeForecast])  # using Cochrane-Orcutt (CO)
print("Forecast Corrent I12 considering OLS:", I12f1, "\n")
print("Forecast Corrent I12 considering CO:", I12f2)

# Plot forecsated values

x = range(time)
xx = range(time - 1)
xxx = range(time, time + timeForecast)

yy1 = i12[0:time]
yy2 = i1w[0:time]
yy3 = rss_1
yy4 = rss_2
yy5 = i12[0:time] - rss_1
yy6 = i12[0:time - 1] - rss_2[0:time - 1]
yy7 = i12[time:time + timeForecast]
yy8 = I12f1
yy9 = I12f2
D = durbin_watson(yy6)
print("AAAAAAAAAAAAAAA")
print("The value of Durdin-Watson (DW) is:", D)
D = durbin_watson(yy5)
print("AAAAAAAAAAAAAAA")
print("The value of Durdin-Watson (DW) is:", D)

plt.plot(xxx, yy7, color='red', label='Measured')
plt.plot(xxx, yy8, color='black', linestyle='dashed', marker='o', label='OLS')
plt.plot(xxx, yy9, color='red', linestyle='-.', marker='*', label='CO')
plt.legend()
plt.xlabel('Time stamp [h]')
plt.ylabel('Current I_12')
plt.xlim([24, 35])
plt.show()

plt.plot(yy2, yy1, 'C1o', label='i12')
plt.plot(yy2, yy3, label='OLC Regression')
plt.plot(yy2, yy4, label='CO Regression')
plt.legend()
plt.xlabel("Injection P_1")
plt.ylabel("Current I_12")

plt.show()

plt.plot(x, yy5, 'C1o', label='Residuals OLS')
plt.plot(xx, yy6, 'C0*', label='Residuals CO')

plt.legend()
plt.show()

print(sum(np.square(yy5)))
print(sum(np.square(yy6)))

# Autocorrelation Method

# In this example, the input data is different because the error used to generate the values is different. To obtain
# the same results, we should use the next values. To compare with previous example, we can skip this step.

ee1 = [0.2878, 0.0145, 0.5846, -0.0029, -0.2718, -0.1411,
       -0.2058, -0.1793, -0.9878, -0.4926, -0.1480, 0.7222,
       -0.3123, 0.4541, 0.9474, -0.1584, 0.4692, 1.0173,
       -0.0503, 0.4684, -0.3604, 0.4678, 0.3047, -1.5098,
       -0.5515, -0.5159, 0.3657, 0.7160, 0.1407, 0.5424,
       0.0409, 0.0450, 0.2365, -0.3875, 1.4783, -0.8487]

ee = [0.2226, -0.2293, -0.1757, -0.1691, -0.2031, -0.5900,
      -0.0556, -0.0845, -0.1834, 0.2798, 0.1534, 0.0751,
      -0.1089, 0.3545, 0.0228, -0.2139, 0.4409, 0.6044,
      -0.2187, -0.1233, 0.0026, 0.4980, 0.3703, 0.0812,
      0.1183, 0.2486, -0.0686, -0.0727, -0.0009, -0.1180,
      0.2443, 0.6224, -0.4600, -0.3878, 0.4734, -0.4050]

II = np.zeros((nBus - 1, time + timeForecast), dtype=complex)
II[:, 0] = I
i12 = np.zeros(time + timeForecast)
i1w = np.zeros(time + timeForecast)

v = 1 + np.dot(np.linalg.inv(Yl), I)
i12[0] = np.absolute(np.dot(Y[0, 1], v[0] - v[1]))
i1w[0] = np.real(I[0])
for t in range(time + timeForecast - 1):
    II[:, t + 1] = 0.95 * II[:, t] + ee[t]  # meter ee
    i1w[t + 1] = 0.75 * i1w[t] + ee1[t]  # meter ee1
    II[0, t + 1] = i1w[t + 1] + complex(0, np.imag(II[0, t + 1]))
    v = 1 + np.dot(np.linalg.inv(Yl), II[:, t + 1])
    I12 = np.dot(-Y[0, 1], v[0] - v[1])
    i12[t + 1] = np.absolute(I12) * np.sign(np.real(I12))

# <b> Models
# - 1 - OLS 
# - 2 - Cochrane Orcutt (CO)
# - 3 - Autorregration AR(1) 
# - 4 - Autorregration with Loads AR(1)+Load Sum


## 1 - OLS
AA[:, 1] = i1w[0:time]  # Vector Xt with ones in first column and wind injection in column 2
AATransp = np.transpose(AA)
beta = np.dot(np.dot(np.linalg.inv(np.dot(AATransp, AA)), AATransp), i12[0:time])  # Beta values
print("The value of Betas, using OLS, are:\n", beta)

## 2 - Cochrane Orcutt (CO)
rss_1 = beta[0] + np.dot(beta[1], i1w[0:time])  # OLS regresion line
res_1 = i12[0:time] - rss_1
for k in range(3):  # According to "Applied Linear Statistical Models" if the OC methot does not converge
    # in three iterations, we should use other method
    r2 = res_1[0:time - 1]
    r1 = res_1[1:time]
    ro = 0.97 * np.dot(np.dot((np.dot(np.transpose(r2), r2)) ** (-1), np.transpose(r2)),
                       r1)  # Estimate Rho based on (28)
    i1w_s = i1w[1:time] - np.dot(ro, i1w[0:time - 1])  # Transform yt*=yt
    i12_s = i12[1:time] - np.dot(ro, i12[0:time - 1])  # Transform xt*=Xt
    B = np.ones((time - 1, 2))
    B[:, 1] = i1w_s
    b_s = np.dot(np.dot(np.linalg.inv(np.dot(np.transpose(B), B)), np.transpose(B)),
                 np.transpose(i12_s))  # Regress yt* over xt*

    b_s[0] = np.divide(b_s[0], 1 - ro)  # Transform Beta_0
    rss_s = b_s[0] + np.dot(b_s[1], i1w_s[0:time - 1])  # Update residuals
    rss_2 = b_s[0] + np.dot(b_s[1], i1w[0:time])
    res_2 = i12[0:time] - rss_2
    res_1 = res_2[:]
b_ss = b_s

## 3 - Autorregration AR(1)
# Definition of Matrix Xt
X = np.ones((time, 3))  # Vector Xt with ones
X[:, 1] = i1w[1:time + 1]  # Vector Xt with ones in first column and wind injection in column 2
X[:-1, 2] = i12[0:time - 1]  # Vector Xt with previous time current i12 injection
XTransp = np.transpose(X)

# Definition of Matrix y
# Compute Beta using 32
Beta_AR1 = np.dot(np.dot(np.linalg.inv(np.dot(XTransp, X)), XTransp), i12[0:time])

# Equation 31
# Residuals

## 4 - Autorregration with Loads AR(1)+Load Sum
soma = np.zeros(time + timeForecast)
for j in range(time + timeForecast):
    soma[j] = sum(np.real(II[:, j]))

# Definition of Matrix Xt
X2 = np.ones((time, 3))  # Vector Xt with ones
X2[:, 1] = i1w[0:time]  # Vector Xt with ones in first column and wind injection in column 2
X2[:, 2] = soma[0:time]  # Vector Xt with previous time current i12 injection
XTransp2 = np.transpose(X2)
# Definition of Matrix y
# Compute Beta using 32
Beta_AR2 = np.dot(np.dot(np.linalg.inv(np.dot(XTransp2, X2)), XTransp2), i12[0:time])
# Equation 31
# Residuals

# Definition of Matrix Xt
X3 = np.ones((time - 1, 4))  # Vector Xt with ones
X3[:, 1] = i1w[1:time]  # Vector Xt with ones in first column and wind injection in column 2
X3[:, 2] = i12[0:time - 1]  # Vector Xt with previous time current i12 injection
X3[:, 3] = soma[1:time]  # Vector Xt with previous time current i12 injection
XTransp3 = np.transpose(X3)
# Definition of Matrix y
# Compute Beta using 32
Beta_AR3 = np.dot(np.dot(np.linalg.inv(np.dot(XTransp3, X3)), XTransp3), i12[1:time])
# Equation 31
# Residuals

# <b>Forecast Day-ahead (Current I_12)

# 1 - OLS 

I12f1 = beta[0] + np.dot(beta[1], i1w[time:time + timeForecast])  # using Ordinary least Squares (OLS)
print("Forecast Corrent I12 considering OLS:", I12f1, "\n")

# 2 - Cochrane Orcutt (CO)
I12f2 = b_ss[0] + np.dot(b_ss[1], i1w[time:time + timeForecast])  # using Cochrane-Orcutt (CO)
print("Forecast Corrent I12 considering CO:", I12f2)

# 3 - Autorregration AR(1)
I12f3 = Beta_AR1[0] + np.dot(Beta_AR1[1], i1w[time:time + timeForecast]) + np.dot(Beta_AR1[2], i12[time: time +
                                                                                  timeForecast])

# 4 - Autorregration with Load Sum
I12f4 = np.zeros(timeForecast)
I12f4[0] = Beta_AR2[0] + Beta_AR2[1] * i1w[time] + Beta_AR2[2] * soma[time]
for i in range(1, timeForecast):
    I12f4[i] = Beta_AR2[0] + Beta_AR2[1] * i1w[time + i] + Beta_AR2[2] * soma[time + i]

# 5 - Autorregration with Loads AR(1)+Load Sum
I12f5 = np.zeros(timeForecast)
I12f5[0] = Beta_AR3[0] + Beta_AR3[1] * i1w[time] + Beta_AR3[2] * i12[time - 1] + Beta_AR3[3] * soma[time]
for i in range(1, timeForecast):
    I12f5[i] = Beta_AR3[0] + Beta_AR3[1] * i1w[time + i] + Beta_AR3[2] * I12f5[i - 1] + Beta_AR3[3] * soma[time + i]

# <b>Plot forecsated values
yy7 = i12[time:time + timeForecast]
yy8 = I12f1
yy9 = I12f2
yy10 = I12f3
yy11 = I12f4
yy12 = I12f5

plt.plot(xxx, yy7, color='red', label='Measured')
plt.plot(xxx, yy8, color='black', linestyle='dashed', marker='o', label='OLS')
plt.plot(xxx, yy9, color='red', linestyle='-.', marker='*', label='CO')
plt.plot(xxx, yy10, color='green', linestyle='dashed', marker='o', label='AR(1)')
plt.plot(xxx, yy11, color='orange', linestyle='-.', marker='*', label='Loads')
plt.plot(xxx, yy12, color='purple', linestyle='-.', marker='*', label='AR(1)+Loads')
plt.xlim((24, 36))
plt.legend()
plt.xlabel('Time stamp [h]')
plt.ylabel('Current I_12')
plt.show()


