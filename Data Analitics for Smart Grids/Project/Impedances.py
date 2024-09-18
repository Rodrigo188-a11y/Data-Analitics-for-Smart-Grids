import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Parameters

cosPhi = 0.95
# time=48
m = 12
netFactor = 0.25
# noiseFactor=0.005


# <b>Initial data

# Consumption dataset
s = [[0.0450, 0.0150, 0.0470, 0.0330],
     [0.0250, 0.0150, 0.2480, 0.0330],
     [0.0970, 0.0250, 0.3940, 0.0330],
     [0.0700, 0.0490, 0.0200, 0.4850],
     [0.1250, 0.0460, 0.0160, 0.1430],
     [0.2900, 0.0270, 0.0160, 0.0470],
     [0.2590, 0.0150, 0.0170, 0.0200],
     [0.2590, 0.0160, 0.0280, 0.0160],
     [0.4420, 0.0160, 0.0500, 0.0170],
     [0.2010, 0.0230, 0.0460, 0.0160],
     [0.2060, 0.0490, 0.0220, 0.0240],
     [0.1300, 0.0470, 0.0160, 0.0490],
     [0.0460, 0.0260, 0.0170, 0.0480]]
s = np.array(s)

# topology
topo = [[1, 2], [2, 3], [3, 4]]
nBUS = np.max(topo)

# Impedance
z = np.multiply([complex(0.397, 0.0099), complex(0.1257, 0.0085), complex(0.0868, 0.0092)], netFactor)

vr = 1  # Reference voltage
el = 1
ni = 20  # Iterations for the Power Flow


# Power Flow Function


def pf3ph(t, z, si, vr, el, ni, al):
    # Matrices creation
    t = np.array(t)
    p = t[:, 0]
    f = t[:, 1]
    w = len(p) + 1
    vp = np.zeros((nBUS - 1, w), dtype=complex)
    vn = np.zeros((nBUS - 1, w), dtype=complex)
    vp[0, 0:w] = vr
    for h in range(2, nBUS):
        vp[h - 1, :] = vp[h - 2, :] * al  # Create a three-phase system of voltages
        # Voltages will be the same in all BUS

    va = vp - vn  # Auxiliar voltage
    ia = np.conj(np.divide(np.multiply(si, np.abs(va) ** el), va))  # Auxiliar current
    for it in range(ni):  # Iterations of Power Flow
        va = vp - vn
        ip = np.conj(np.divide(np.multiply(si, np.abs(va) ** el), va))  # Phase current
        inn = -np.sum(ip, 0)  # Neutral current
        for k in range(w - 1, 0, -1):  # Backward Cycle
            n = f[k - 1]
            m = p[k - 1]
            ip[:, m - 1] = ip[:, m - 1] + ip[:, n - 1]  # Phase Current
            inn = -np.sum(ip, 0)  # Neutral Current

        eps = np.linalg.norm(
            np.max(np.abs(ia - ip), 0))  # Error, comparing the new currents and the old ones (previous iteration)

        if eps > 1e-4:
            ia = ip
            mvp = 0
            mvn = 0
            eps = np.inf
        else:  # If the error is lower than the limit, we can return the results
            mvp = (vp - vn)  # Phase Voltages to return
            mvn = vn[0, :]  # Neutral Voltage to return
            #            return mvp, mvn, eps, ip, inn;
            return mvp, ip;
        for k in range(w - 1):  # Forward Cycle
            n = f[k]
            m = p[k]
            vn[:, n - 1] = vn[:, m - 1] - z[k]/2 * inn[n - 1]  # Neutral Voltage
            vp[:, n - 1] = vp[:, m - 1] - z[k] * ip[:, n - 1]  # Phase Voltage
        ia = ip  # Save the current of previous iteration


# <b>Compute the values of voltages in function of currents

al = np.exp(np.multiply(np.multiply(complex(0, -1), 2 / 3), np.pi))  # Phase Angle
sp = np.mean(s[0:m, :], axis=0)  # Average power in each phase (i0)
# State Estimation
Y = np.zeros(12 * 3, complex)
x = np.zeros((2, 12 * 3), complex)

Beta = np.zeros((3, 3), complex)

W = [[2, 1, 1],
     [1, 2, 1],
     [1, 1, 2]]
W = np.array(W)

al_aux = np.array((1, al, np.square(al))).reshape((3, 1))

######   Computing the impedances for the first node  ###########################
mu = 0
aux5 = np.zeros(3, complex)
aux6 = np.zeros(3, complex)
error_percentage = 0.05
Z0 = np.zeros(25 * 2, complex)
Z1 = np.zeros(25 * 2, complex)
Z2 = np.zeros(25 * 2, complex)
for k in range(3):
    for b in range(25):
        a = 0
        for i in range(m):
            # Power in each instant
            si = [[0, 0, s[i, 2], 0], [0, 0, s[i, 1], 0], [0, s[i, 0], 0, s[i, 3]]]  # Power in each bus and in each
            # phase
            aux = np.array(si)
            # aux = np.multiply(np.array(si), al_aux)

            # Load Flow
            mvp, ip = pf3ph(topo, z, si, vr, el, ni, al)  # Compute the power flow
            # aux = np.divide(aux, np.conj(mvp))
            for j in range(3):
                aux = np.random.normal(mu, abs(mvp[j, k].real) * (error_percentage / 100), 1)
                aux2 = np.random.normal(mu, abs(mvp[j, k].imag) * (error_percentage / 100), 1)
                aux5[j] = complex(aux, aux2)
            mvp[:, k] = mvp[:, k] + aux5
            for j in range(3):
                aux = np.random.normal(mu, abs(mvp[j, k+1].real) * (error_percentage / 100), 1)
                aux2 = np.random.normal(mu, abs(mvp[j, k+1].imag) * (error_percentage / 100), 1)
                aux5[j] = complex(aux, aux2)
            mvp[:, k+1] = mvp[:, k+1] + aux5
            for j in range(3):
                aux = np.random.normal(mu, abs(ip[j, k+1].real) * (error_percentage / 100), 1)
                aux2 = np.random.normal(mu, abs(ip[j, k+1].imag) * (error_percentage / 100), 1)
                aux5[j] = complex(aux, aux2)
            ip[:, k+1] = ip[:, k+1] + aux5

            Y[a:a + 3] = np.subtract(mvp[:, k], mvp[:, k+1])

            x[0, a:a + 3] = ip[:, k+1]
            x[1, a:a + 3] = np.sum(ip[:, k+1])
            a += 3

        xtransp = np.transpose(x)

        Beta = np.dot(np.dot(np.linalg.inv(np.dot(x, xtransp)), x), Y.transpose())
        if k == 0:
            Z0[b * 2] = Beta[0]
            Z0[b * 2 + 1] = Beta[1]
        elif k == 1:
            Z1[b * 2] = Beta[0]
            Z1[b * 2 + 1] = Beta[1]
        elif k == 2:
            Z2[b * 2] = Beta[0]
            Z2[b * 2 + 1] = Beta[1]

error_0 = np.zeros(25 * 2)
error_1 = np.zeros(25 * 2)
error_2 = np.zeros(25 * 2)

Error_med_R = np.zeros(25)
Error_med_X = np.zeros(25)

for i in range(25):
    error_0[i * 2] = abs((z[0] - Z0[2 * i]) / z[0]) * 100
    error_0[2 * i + 1] = abs((z[0] / 2 - Z0[2 * i + 1]) / (z[0] / 2)) * 100
    error_1[2 * i] = abs((z[1] - Z1[2 * i]) / z[1])
    error_1[2 * i + 1] = abs((z[1] / 2 - Z1[2 * i + 1]) / (z[1] / 2)) * 100
    error_2[2 * i] = abs((z[2] - Z2[2 * i]) / z[2]) * 100
    error_2[2 * i + 1] = abs((z[2] / 2 - Z2[2 * i + 1]) / (z[2] / 2)) * 100
    Error_med_R[i] = abs((z[0] - Z0[2 * i]) / z[0]) * 100 + abs((z[1] - Z1[2 * i]) / z[1]) + abs(
        (z[2] - Z2[2 * i]) / z[2]) * 100
    Error_med_X[i] = abs((z[0] / 2 - Z0[2 * i + 1]) / (z[0] / 2)) * 100 + abs(
        (z[1] / 2 - Z1[2 * i + 1]) / (z[1] / 2)) * 100 + abs((z[2] / 2 - Z2[2 * i + 1]) / (z[2] / 2)) * 100

aaa = sum(Error_med_R)/(25*3)
bbb = sum(Error_med_X)/(25*3)
print(aaa)
print(bbb)

Error_med = (sum(error_0) + sum(error_1) + sum(error_2)) / (50 * 3)

z_aux = np.zeros(25, complex)
z_aux2 = np.zeros(25, complex)
for i in range(25):
    z_aux[i] = Z0[2 * i]
    z_aux2[i] = Z0[2 * i + 1]

fig = plt.figure()
plt.title("Scattered Mismatches")
# Labeled the axis
plt.xlabel("Resistance")
plt.ylabel("Reactance")
plt.scatter(z_aux.real, z_aux.imag, c='midnightblue', marker='.', label='Z1_p', alpha=0.5,
            linewidths=1.5)
plt.scatter(z_aux2.real, z_aux2.imag, c='cornflowerblue', marker='.', label='Z1_n', alpha=0.5,
            linewidths=1.5)
auxx = np.zeros(2)
auxx2 = np.zeros(2)
auxx[0] = z[0].real
auxx[1] = z[0].real / 2
auxx2[0] = z[0].imag
auxx2[1] = z[0].imag / 2
plt.scatter(auxx[0], auxx2[0], c='midnightblue', marker='*', label='Z1_p_real', linewidths=2)
plt.scatter(auxx[1], auxx2[1], c='cornflowerblue', marker='*', label='Z1_n_real', linewidths=2)

for i in range(25):
    z_aux[i] = Z1[2 * i]
    z_aux2[i] = Z1[2 * i + 1]

plt.scatter(z_aux.real, z_aux.imag, c='darkgreen', marker='.', label='Z2_phases', alpha=0.5,
            linewidths=1.5)
plt.scatter(z_aux2.real, z_aux2.imag, c='lime', marker='.', label='Z2_neutral', alpha=0.3,
            linewidths=1.5)
auxx = np.zeros(2)
auxx2 = np.zeros(2)
auxx[0] = z[1].real
auxx[1] = z[1].real / 2
auxx2[0] = z[1].imag
auxx2[1] = z[1].imag / 2
plt.scatter(auxx[0], auxx2[0], c='darkgreen', marker='*', label='Z2_p_real', linewidths=2)
plt.scatter(auxx[1], auxx2[1], c='lime', marker='*', label='Z2_n_real', linewidths=2)

for i in range(25):
    z_aux[i] = Z2[2 * i]
    z_aux2[i] = Z2[2 * i + 1]

plt.scatter(z_aux.real, z_aux.imag, c='purple', marker='.', label='Z3_phases', alpha=0.4,
            linewidths=1.5)
plt.scatter(z_aux2.real, z_aux2.imag, c='fuchsia', marker='.', label='Z3_neutral', alpha=0.3,
            linewidths=1.5)
auxx = np.zeros(2)
auxx2 = np.zeros(2)
auxx[0] = z[2].real
auxx[1] = z[2].real / 2
auxx2[0] = z[2].imag
auxx2[1] = z[2].imag / 2
plt.scatter(auxx[0], auxx2[0], c='purple', marker='*', label='Z3_p_real', linewidths=2)
plt.scatter(auxx[1], auxx2[1], c='fuchsia', marker='*', label='Z3_n_real', linewidths=2)

plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.05),
           ncol=3, fancybox=True, shadow=True)
plt.show()
