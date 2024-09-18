import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import cvxpy as cp  # Optimization Library (https://www.cvxpy.org/install/)
from scipy.linalg import block_diag  # Compose matrix for optimization

# <b>Parameters

# In[2]:


cosPhi = 0.95  # Value of teta
time = 48  # The time will change in the different steps
networkFactor = 100  # To change the characteristics of the network (Y)

# <b>Import data (From Excel file)

# In[3]:


Info = np.array(pd.read_excel(r'C:\Users\linha\OneDrive\Documents\Rodrigo_Contreiras\Programação\Machine_learning'
                              r'\Analitica_redes_energia\Lab6\DASG_Prob2_new.xlsx', sheet_name='Info', header=None))
# Information about the slack bus
SlackBus = Info[0, 1]
print("Slack Bus: ", SlackBus, "\n")

# Network Information
Net_Info = np.array(pd.read_excel(r'C:\Users\linha\OneDrive\Documents\Rodrigo_Contreiras\Programação\Machine_learning'
                                  r'\Analitica_redes_energia\Lab6\DASG_Prob2_new.xlsx', sheet_name='Y_Data'))
print("Lines information (Admitances)\n", Net_Info, "\n")

# Power Information (Train)
Power_Info = np.array(pd.read_excel(r'C:\Users\linha\OneDrive\Documents\Rodrigo_Contreiras\Programação\Machine_learning'
                                    r'\Analitica_redes_energia\Lab6\DASG_Prob2_new.xlsx', sheet_name='Load(t,Bus)'))
Power_Info = np.delete(Power_Info, [0], 1)
print("Power consumption information (time, Bus)\n", Power_Info, "\n")

# Power Information (Test)
Power_Test = np.array(pd.read_excel(r'C:\Users\linha\OneDrive\Documents\Rodrigo_Contreiras\Programação\Machine_learning'
                                    r'\Analitica_redes_energia\Lab6\DASG_Prob2_new.xlsx',
                                    sheet_name='Test_Load(t,Bus)'))
Power_Test = np.delete(Power_Test, [0], 1)
print("Power consumption information (time, Bus)\n", Power_Test, "\n")

P = np.dot(-Power_Info, np.exp(complex(0, 1) * np.arccos(cosPhi)))
I = np.conj(P[2, :])

P = np.dot(-Power_Info, np.exp(complex(0, 1) * np.arccos(cosPhi)))
# print(P)
I = np.conj(P[2, :])

# <b>Admittance Matrix(<i>Y</i>); Conductance Matrix(<i>G</i>); Susceptance Matrix(<i>B</i>)
# 
# ![image-2.png](attachment:image-2.png)

# In[4]:


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

# In[5]:


np.random.seed(98)
e1 = np.random.randn(time) * 0.5
e = np.random.randn(4, time) * 0.15

# To obtain the same values of lecture notes, we should use the following errors

# In[6]:


e1 = [0.2878, 0.0145, 0.5846, -0.0029, -0.2718, -0.1411,
      -0.2058, -0.1793, -0.9878, -0.4926, -0.1480, 0.7222,
      -0.3123, 0.4541, 0.9474, -0.1584, 0.4692, 1.0173,
      -0.0503, 0.4684, -0.3604, 0.4678, 0.3047, -1.5098,
      -0.5515, -0.5159, 0.3657, 0.7160, 0.1407, 0.5424,
      0.0409, 0.0450, 0.2365, -0.3875, 1.4783, -0.8487,
      -0.0211, 0.0266, 0.4451, 0.4663, 0.3200, -0.1156,
      -0.4587, -0.5685, -0.5464, -0.2405, -0.3513, -0.3781]

e = [[0.0925, -0.2709, -0.0663, -0.0486, -0.0194, -0.0288,
      -0.0961, 0.0720, 0.0084, -0.0848, 0.0453, 0.1048,
      0.0801, -0.2532, 0.0692, -0.0930, 0.2247, -0.0583,
      0.1100, 0.0159, 0.1016, -0.0278, -0.0942, -0.0101,
      -0.0428, 0.0711, 0.1195, 0.0303, -0.2142, -0.0605,
      -0.0793, 0.1686, -0.0161, -0.0191, -0.3057, -0.0787,
      -0.0235, -0.0007, -0.2525, -0.1399, 0.0970, 0.0330,
      -0.3454, -0.0300, 0.1832, 0.0803, -0.0141, -0.0123],
     [-0.0056, -0.1072, -0.0848, -0.0014, -0.0642, -0.0522,
      0.0415, -0.1746, -0.0378, -0.0668, 0.1215, -0.1230,
      -0.1058, -0.1673, -0.1110, -0.0361, -0.0813, 0.0340,
      -0.2259, 0.3126, -0.0760, 0.0552, -0.0117, 0.0853,
      -0.1266, 0.0981, -0.1846, 0.0642, -0.2060, 0.0298,
      -0.0203, 0.1678, -0.1196, 0.0370, -0.2070, -0.0424,
      -0.0182, 0.2051, 0.0612, -0.0256, -0.0363, -0.1274,
      -0.0144, 0.1213, 0.0474, 0.0452, 0.1294, -0.0838],
     [-0.1015, -0.1218, -0.3540, -0.0333, -0.0507, -0.1101,
      0.1679, 0.0920, 0.0451, -0.0653, 0.2127, 0.0137,
      -0.1283, 0.2646, 0.3626, -0.1312, -0.0740, 0.0016,
      0.2988, 0.2222, 0.0487, 0.0710, 0.1491, -0.0412,
      -0.0436, -0.0005, -0.0708, 0.1466, 0.3735, -0.2760,
      -0.2327, 0.2841, -0.2430, -0.2334, -0.0864, 0.0692,
      0.0447, 0.0375, -0.1278, 0.0254, -0.1437, -0.2031,
      0.0113, -0.2351, -0.3242, -0.0833, -0.2619, 0.1001],
     [0.1327, -0.2181, -0.1038, 0.1115, -0.0446, -0.0859,
      -0.1246, 0.1313, 0.2358, -0.0910, -0.1343, -0.0258,
      0.3350, -0.0661, 0.0017, -0.0668, 0.1063, 0.1508,
      0.1718, 0.0782, -0.0191, 0.0773, 0.0475, 0.0306,
      -0.0054, -0.0963, -0.4423, 0.0650, -0.3559, -0.0976,
      -0.1173, -0.1590, -0.0633, -0.1842, 0.1666, 0.1624,
      0.2661, 0.2032, -0.2687, 0.0535, 0.0120, 0.2273,
      0.0263, -0.1072, 0.0176, -0.0135, -0.0401, -0.1862]]

e = np.array(e)

# <b> Determine the wind generation and the load flow in <i>I<sub>12

# In[7]:


# Creation of Matrix
II = np.zeros((nBus - 1, time), dtype=complex)
i12 = np.zeros(time)
i1w = np.zeros(time)

# Initializing the process of data generation
II[:, 0] = I  # Power Injections

v = 1 + np.dot(np.linalg.inv(Yl), I)
i12[0] = np.absolute(np.dot(Y[0, 1], v[0] - v[1]))  # Current I12 in period t=0
i1w[0] = np.real(I[0])  # Injection in bus 1 (Wind) in period t=0

# Process of data generation
for t in range(time - 1):
    II[:, t + 1] = 0.999 * II[:, t] + e[2, t]  # Power injection based on previous periods and in the errors. T
    # the values are more or less related considering
    # the value of 0.95. This value can change between 0 and 1.
    i1w[t + 1] = 0.75 * i1w[t] + e1[t]  # Wind power based on the previous periods
    II[0, t + 1] = i1w[t + 1] + np.complex(0, np.imag(II[0, t + 1]))  # Add the Wind generation
    v = 1 + np.dot(np.linalg.inv(Yl), II[:, t + 1])  # Compute the voltages
    I12 = np.dot(-Y[0, 1], v[0] - v[1])  # Compute the load flow in line 1-2 (Complex)
    i12[t + 1] = np.absolute(I12) * np.sign(np.real(I12))  # Compute the load flow in line 1-2 (RMS with signal)

print('The power injection in Bus 1 is:\n', II[0, :])
print('\nThe power flow in Line 1-2 is:\n', i12)

# <b> Plot the input data (Page 65 of the lectures)

# In[8]:


TPlot = 24  # Define a time to plot

# Define the plots
x = range(TPlot)
yy1 = i12[0:TPlot]
yy2 = i1w[0:TPlot]
yy3 = np.absolute(II[:, 0:TPlot]) * np.sign(np.real(II[:, 0:TPlot]))

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
plt.show()

# Power From Loads
for i in range(3):
    y4 = yy3[i + 1, :]
    #    ll=('Conso '+  str(i))
    plt.plot(x, y4)
plt.xlabel("Time Stamp [h]")
plt.ylabel("Active Power from Loads")
# plt.legend()
plt.show()

# <b><u>MARKOV Chain</u></b>
# 
# Assuming we have recent measurements of both <i>y<sub>t</sub></i> and <i>W<sub>t</sub></i>,
# <i>t = 1, . . . , t<sub>0</sub></i>, we first have to be able to estimate <i>X<sub>t0</sub></i>. Because <i>X<sub>t</sub></i> is a Markov process whose state cannot be observed directly in real-time, the problem of estimating $X_{t0}$ based on <i>y<sub>t</sub></i> and <i>W<sub>t</sub></i> is a Hidden Markov Model (HMM) problem.

# <b>Discretization of the injection data
#     ![image.png](attachment:image.png)

# In[31]:


# The variables of Markov Chain are:
#  - y -> The measurement of current in branch 1-2
#  - w -> The measurement of wind power injection
#  - X -> The power injections

# Definition of variables y and x
y = i12[:]
W = i1w[:]
X = np.transpose(np.real(II[1:4, :]))
Xd = np.round(2 * X) / 2  # To have the variable X rounded according the state limits

# Definition of the States s
s = [-1.5, -1, -0.5, 0, 0.5, 1]

# Definition of the number/percentage of measurements in each state
pp = np.zeros([6])
pp[0] = np.sum(np.sum(Xd <= s[0]))
pp[1] = np.sum(np.sum(Xd == s[1]))
pp[2] = np.sum(np.sum(Xd == s[2]))
pp[3] = np.sum(np.sum(Xd == s[3]))
pp[4] = np.sum(np.sum(Xd == s[4]))
pp[5] = np.sum(np.sum(Xd >= s[5]))
ppi = pp / np.sum(pp)

print('The number of measurements in each state are:\n', pp)
print('\nThe percentage of measurements in each state are:\n', ppi)
print('\n', X)
print('\n', Xd)

# <b>Transition matrix for Homogeneous Markov Process
# ![image.png](attachment:image.png)

# In[10]:


# Create Variables
nStates = 6  # Number of States
kPeriod = 24  # Number of periot until t0
nLoads = 3  # Number of Loads
n = np.zeros(([nStates, nStates]))

# Number of times that the state i change to state j
for t in range(time - 1):
    for m in range(nLoads):
        for i in range(nStates):
            for j in range(nStates):
                if Xd[t, m] == s[i] and Xd[t + 1, m] == s[j]:
                    n[i, j] = n[i, j] + 1

# Probability Pij (in Homogeneous process)
nij = n / (np.sum(n, axis=0))

print('The number of transitions (k=1) are:\n', n)
print('\nThe probability of transitions (k=1) are:\n', nij)


# <b>Transition matrix for Non-Homogeneous Markov Process</b>
# 
# A possible approach to parametrize a non-homogeneous Markov
# process representative of a given period is to adjust $\pi_s$ and $P$ for
# every time-period $k$. The adjusted variables are named $\pi_{s,k}$ and
# $P_k$. The values of $\pi_{s,k}$ can be obtained from $\pi_s$ with:
#     
#     
#  ![image.png](attachment:image.png)
# 

# The adjusting coefficients $\pi_{s,k}$ should force the expected load in period k to be equal to time-series
# average μk and the sum of $\pi_{s,k}$  to be unitary. These constraints can be written as $A\alpha_{s,
# k}=b_k$ where, ![image.png](attachment:image.png)

# <b>Generic Optimization Function (Quadratic Problem)
#     ![image.png](attachment:image.png)

# In[11]:


def OptFunc(A, b, C, d, lb, ub):
    # Problem data.
    n = np.shape(C)[1]

    # Construct the problem.
    x = cp.Variable(n)
    objective = cp.Minimize(cp.sum_squares(C @ x - d))
    constraints = [lb <= x, x <= ub, A @ x - b <= 0, A @ x - b >= 0]
    prob = cp.Problem(objective, constraints)

    # The optimal objective value is returned by `prob.solve()`.
    result = prob.solve(solver='OSQP')

    return x.value, prob.solve(), prob.status


# <b>Transition Matrices for each period $k$
#     ![image.png](attachment:image.png)
#     ![image-2.png](attachment:image-2.png)

# In[26]:


# Create variables
mu = np.zeros(time)
b = np.ones(([time, 2])) * 1.0
a = np.zeros(([time, nStates]))

# Everage values of load
for t in range(time):
    mu[t] = np.sum(X[t, :]) / 3  # Compute the values of micro_k (expected load)
    b[t, 0] = mu[t]  # Compute the matrix B. The elements of second column are one
    # because is the total probability

# Compute matrix A, considering the probabilities of first period
A = np.vstack((s * ppi, ppi))

# Compute matrix C (Diagonal with ones)
C = np.zeros(([nStates, nStates]))
np.fill_diagonal(C, 1)

# Compute matric D (Vector with ones)
d = np.ones(nStates)

# Lower Bound
lb = np.ones(nStates) * 0.01

# Upper Bound
ub = np.ones(nStates) * np.inf

# Compute the value of alfa (matrix a) for each time period
for t in range(time):
    a[t, :], OptValue, OptStatus = OptFunc(A, b[t, :], C, d, lb, ub)  # Optimization

    # If the solver not return a solution, we can try to relax the problem.
    # In that case, we will consider that the sum of the probabilities can be higher than 1.

    while np.isnan(a[t, 0]):
        b[t, 1] = b[t, 1] + 0.0025  # We are relaxing the problem in steps of 0.0025
        a[t, :], OptValue, OptStatus = OptFunc(A, b[t, :], C, d, lb, ub)  # Do optimization again

print('The values of Alfa are\n', a)

# <b>The adjusted variables $\pi_{s,k}$ and $P_k$ are:
#     ![image.png](attachment:image.png)

# In[13]:


# Create Matrix
pi_k = np.zeros(([time, nStates]))
pi_k1 = np.zeros(([time, nStates]))
pp = np.zeros(nStates)
pi_emp = np.zeros(([time, nStates]))

# Compute the value of Pi_s,k
for t in range(time):
    pi_k1[t, :] = np.multiply(np.transpose(ppi), np.transpose(a[t, :]))  # Compute Pi*Alpha
    # Over-right pi_k with hourly values
    pp[0] = np.sum(np.sum(Xd[t, :] <= s[0]))
    pp[1] = np.sum(np.sum(Xd[t, :] == s[1]))
    pp[2] = np.sum(np.sum(Xd[t, :] == s[2]))
    pp[3] = np.sum(np.sum(Xd[t, :] == s[3]))
    pp[4] = np.sum(np.sum(Xd[t, :] == s[4]))
    pp[5] = np.sum(np.sum(Xd[t, :] >= s[5]))
    pi_emp[t, :] = pp / 3

pi_k = pi_emp

print("The computed values of 𝜋𝑠,𝑘 are:\n", pi_k1)
print("The over-right values of 𝜋𝑠,𝑘 are:\n", pi_k)

# <b>New transition matrices $P_{ij}$ for each period $k$
#     ![image.png](attachment:image.png)

# Also see Equations 41 to 47 of lectures

# Create Matrices
g = np.zeros(([time, nStates ** 2]))  # Matrix with the results
C = np.diag(np.diag(np.ones(([nStates ** 2, nStates ** 2]))))  # Part of Objective Function
d = np.ones(nStates ** 2)  # Part of Objective Function
lb = np.zeros(nStates ** 2)  # Lower Bound
ub = np.ones(nStates ** 2) * np.inf  # Upper Bound

# Compose the matrices for the optimization
for t in range(1, time):
    Ck = 0
    D = 0
    Ck = np.multiply(pi_k[t - 1, :], np.transpose(nij[:, 0]))  # See Eq. 40 and 41 of the lectures
    D = np.diag((nij[:, 0]))  # See Eq. 44 of the lectures.
    for i in range(nStates - 1):
        F = np.multiply(pi_k[t - 1, :], np.transpose(nij[:, i + 1]))  # See Eq. 41 of the lectures
        Ck = block_diag(Ck, F)  # See Eq. 40
        G = np.diag((nij[:, i + 1]))  # See Eq. 45 of the lectures
        D = np.hstack((D, G))  # See Eq. 44 of the lectures
    Ak = np.vstack((Ck, D))
    A_aux = pi_k[t, :]
    bk = np.hstack((pi_k[t, :], np.ones(nStates)))
    # Optimization
    sol, OptValue, OptStatus = OptFunc(Ak, bk, C, d, lb, ub)  # See Eq. 47
    # Evaluate if the optimal solution is achieved
    if OptStatus == "optimal":
        g[t, :] = sol
    else:
        g[t, :] = 1

print('The value of the adjusting coefficient Gamma_ij is \n', g)

# <b>Markov chain trajectories in the state space of $X_t$

# Having determined the optimum variations, $\gamma_k$, the new matrix $P_k=(p^{ij}_k)$ is obtained by varying its elements accordingly, i.e.,$p^{ij}_k\leftarrow p^{ij}\gamma^{ij}_k$.
# 
# ![image.png](attachment:image.png)

# In[27]:


# Create matrices
nSamples = 590
pijSt = np.zeros([time, nStates, nStates])
Xs = np.zeros(([nSamples, time]))

# for each sample
for ii in range(nSamples):
    # Create matrix of states
    xx = np.zeros(time)

    # The initial state is s1
    xx[0] = 2

    # Generate a random value. This value is used to define the transition between periods k
    tj = np.random.random(time - 1)

    # Definition of the trajetories in the state space
    for t in range(1, time):
        gg = np.reshape(g[t, :], (nStates, nStates))
        pij = np.transpose(np.multiply(np.transpose(nij), gg))  # Eq.39 of the lectures
        if ii == 0:  # This is only to print the two Pij in the first two k
            pijSt[t - 1, :, :] = pij

        iis = np.int(xx[t - 1])  # State in the previous period
        cum = 0
        xx[t] = xx[t - 1]

        # Definitio of the state in next period
        for j in range(nStates):
            cum = cum + pij[iis - 1, j]
            if cum >= tj[t - 1]:
                xx[t] = j + 1
                break
    Xs[ii, :] = (xx - 4) / 2;

pijSt[1, :, :]
print('Example of first two time periods, k=1,2 (Example of page 70)\nP1=\n', pijSt[0, :, :])
print('\nP2=\n', pijSt[1, :, :])

# <b>24-period non-homogeneous Markov chain trajectories in the state space of $X_t$ obtained by sampling with $Pr(
# X_{t+1} = x  |  X_t = x_t) \quad \textrm{for} \quad x_1 = s_1$

# In[28]:


# Define the plots
x = range(kPeriod)

for i in range(nSamples):
    y4 = Xs[i, 0:kPeriod]
    plt.plot(x, y4)

plt.xlabel("Time Stamp [h]")
plt.ylabel("State Si")
statesD = ['S0', 'S1', 'S2', 'S3', 'S4', 'S5']
y_pos = np.arange(len(statesD)) / 2 - 1.5
# x_pos = [0,10,20,30,40]
# timePos = ['0', '5', '10', '15', '20']
plt.yticks(y_pos, statesD)
plt.show()

# plt.legend()
plt.show()

# <b><u>Viterbi values

# To compute the Viterbi values, $V_{t,i}$, for each state $s_i$, we need to define the output probability density
# function $Pr(y_t|s_i)$ as a function of the observed outputs, $y_t$ and $W_t$. One possible way to define such
# function is to define it as a logistic over a random variable that is an estimation error $\epsilon_{t,i}$.

# In[17]:


# Create Matrices
cs = 0.1  # Scale parameter
i12_s = np.zeros(kPeriod)
ee = np.zeros([nStates, kPeriod])
VV = np.zeros([nStates, kPeriod])
ptr = np.zeros([nStates, kPeriod])
Ss = np.zeros(kPeriod)
Vs = np.zeros(kPeriod)
e_min = np.zeros(kPeriod)

sm = np.mean(s, axis=0)  # Average of states spaces
xlm = np.mean(X, axis=0)  # Average of power injections

# Compute the current i12 considering the relation between the state value and the real value (xlm-sm)
# In the end, we can evaluate the error (variavle ee) comparing the new i12 and the previous one
for t in range(kPeriod):
    for i in range(nStates):
        inj = s[i] + xlm - sm
        III = np.divide(np.concatenate((W[t], inj), axis=None), np.dot(0.95, np.exp(complex(0, 1) * np.arccos(cosPhi))))
        vvv = 1 + np.dot(np.linalg.inv(Yl), np.conjugate(III))
        I12_s = np.dot(-Y[0, 1], vvv[0] - vvv[1])
        i12_s[t] = np.absolute(I12_s) * np.sign(np.real(I12_s))
        ee[i, t] = y[t] - i12_s[t]  # error of the original value i12 and the new one. Eq(52)

# Logistic Distribution
pv = np.multiply(np.exp(np.divide(-ee, cs)), (1 + np.exp(np.divide(-ee, cs))) ** -2)  # Eq (53)

print('The Logistic distribution is:\n', pv)

# <b>24-period output probabilities $Pr(y_t|s_i)$ computed with Eq.(53).

# In[18]:


# Define the plots
x = range(kPeriod)
yy1 = pv

for i in range(nStates):
    y4 = yy1[i, :]
    ll = ('PR(y|S' + str(i) + ')')
    plt.plot(x, y4, label=ll)

plt.xlabel("Time Stamp [h]")
plt.ylabel("Probability")
plt.legend(bbox_to_anchor=(1, 1.02), loc='upper left')
plt.show()

# <b> Viterbi Path

# Once we have computed the output probabilities, we then:
# 
#  • May use $\pi_{s_i,1}$ obtained from Eq. (35) to compute $V_{1,i} = Pr(y_1 | s_i)\pi_{s_i,1}$; 
#  
#  and then
#  
# • Recursively iterate forward in time to compute $V_{t,i} = max_{x\in S}\{Pr(y_t|s_i)p^{x,s_i}_{t-1}V_{t-1,x}\}$

# In[29]:


# Compute V1,i
VV[:, 0] = np.multiply(pv[:, 0], np.transpose(pi_k[0, :]))

# Recursively iterate forward in time to compute  𝑉𝑡,𝑖.
# The matrix VV gave the probability to be in each state in period 24
# The matrix ptr is the pointer with the information about the most probable previous state
for t in range(1, kPeriod):
    gg = np.reshape(g[t, :], (nStates, nStates))
    pij = np.transpose(np.multiply(np.transpose(nij), gg))  # Eq(35)
    for i in range(nStates):
        for j in range(nStates):
            aux = pij[j, i] * pv[j, t - 1] + VV[j, t - 1]
            if aux > VV[i, t]:
                VV[i, t] = aux
                ptr[i, t] = j + 1  # Eq(35)

# Most probable state in t=24
last = VV[:, 23]
maxprob = max(VV[:, kPeriod - 1])  # Max Probability
statemax = np.argmax(VV[:, kPeriod - 1]) + 1  # State of max probability
# Start the backward vectors

Ss[23] = statemax
Vs[23] = maxprob
e_min[23] = ee[int(Ss[23] - 1), 23]

for tt in range(kPeriod - 2, -1, -1):
    Ss[tt] = ptr[int(Ss[tt + 1] - 1), tt + 1]
    Vs[tt] = VV[int(Ss[tt] - 1), tt]
    e_min[tt] = ee[int(Ss[tt + 1] - 1), tt]

print('The most probable states are:\n', Ss, '\n')
print('The probability is:\n', Vs, '\n')
print('The error is:\n', e_min)


# <b>Markov chain trajectories in the state space of injections

# In[20]:


# Create matrices
nSamplesV = 500
Xsv = np.zeros(([nSamplesV, kPeriod]))

# for each sample
for ii in range(nSamplesV):
    # Create matrix of states
    xxv = np.zeros(kPeriod)
    # The initial state is the result of previous function
    xxv[23] = int(Ss[23] - 1)

    # Generate a random value. This value is used to define the transition between periods k
    tjj = np.random.random(time - 1)

    # Definition of the trajetories in the state space
    for t in range(kPeriod - 2, -1, -1):
        gg = np.reshape(g[t, :], (nStates, nStates))
        pij = np.transpose(np.multiply(np.transpose(nij), gg))  # Eq.39 of the lectures

        iisv = np.int(xxv[t + 1])  # State in the previous period
        cum = 0
        xxv[t] = xxv[t + 1]

        # Definition of the state in next period
        for j in range(nStates):
            cum = cum + pij[iisv - 1, j]
            if cum >= tjj[t - 1]:
                xxv[t] = j + 1
                break
    Xsv[ii, :] = (xxv - 4) / 2
# Most probable path defined in the previous period. In that case considering the state values "s"
xsv = (Ss - 4) / 2

# <b>24-period minimal estimation error $\epsilon_{t,i}$ (top plot) and the
# optimal Viterbi path in the state space of the injections that minimize
# such error (bottom plot).

# In[21]:


# Define the plots
x = range(kPeriod)
y1 = e_min
y2 = i12[0:kPeriod]
y3 = xsv

# Graph 1

fig, ax1 = plt.subplots()
ax2 = ax1.twinx()
ax1.plot(x, y1, 'g-')
ax2.plot(x, y2, 'b-')
ax1.set_xlabel('Time Stamp [h]')
ax1.set_ylabel('Error E_t', color='g')
ax2.set_ylabel('Current I_12', color='b')
ax1.set_ylim([-0.2, 0.8])
ax2.set_ylim([-0.2, 0.8])
plt.xlabel("Time Stamp [h]")
plt.show()

# Grahp 2
for i in range(nSamplesV):
    y4 = Xsv[i, :]
    plt.plot(x, y4)
plt.plot(x, xsv, linewidth=5)

# plt.legend()
plt.xlabel("Time Stamp [h]")
plt.ylabel("State Si")
statesD = ['S0', 'S1', 'S2', 'S3', 'S4', 'S5']
y_pos = np.arange(len(statesD)) / 2 - 1.5
plt.yticks(y_pos, statesD)
plt.show()

# In[ ]:

