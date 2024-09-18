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
# z = np.multiply([z_lines_perkm[0] * 2, z_lines_perkm[1] * 1, z_lines_perkm[2] * 4], netFactor)
z = np.multiply([complex(0.397, 0.0099), complex(0.1257, 0.0085), complex(0.0868, 0.0092)], netFactor)
z_lines_perkm = np.multiply([z[0]/9, z[1]/3, z[2]/7], 1)

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
            vn[:, n - 1] = vn[:, m - 1] - z[k] * inn[n - 1]  # Neutral Voltage
            vp[:, n - 1] = vp[:, m - 1] - z[k] * ip[:, n - 1]  # Phase Voltage
        ia = ip  # Save the current of previous iteration


# <b>Compute the values of voltages in function of currents

al = np.exp(np.multiply(np.multiply(complex(0, -1), 2 / 3), np.pi))  # Phase Angle
sp = np.mean(s[0:m, :], axis=0)  # Average power in each phase (i0)
# State Estimation
Y = np.zeros(12 * 3, complex)
x = np.zeros((2, 12 * 3), complex)
Beta = np.zeros((3, 3), complex)
Classi = np.zeros(3)
aux5 = np.zeros(3, complex)
aux6 = np.zeros(3, complex)

######   Computing the impedances for the first node  ###########################
right = 0
wrong = 0
mu = 0
error_percentage = 0.0

for k in range(3):
    a = 0
    for i in range(m):
        # Power in each instant
        si = [[0, 0, s[i, 2], 0], [0, 0, s[i, 1], 0], [0, s[i, 0], 0, s[i, 3]]]  # Power in each bus and in
        # each phase
        aux = np.array(si)

        # Load Flow
        mvp, ip = pf3ph(topo, z, si, vr, el, ni, al)  # Compute the power flow
        for j in range(3):
            aux = np.random.normal(mu, abs(mvp[j, k].real) * (error_percentage/100), 1)
            aux2 = np.random.normal(mu, abs(mvp[j, k].imag) * (error_percentage/100), 1)
            aux5[j] = complex(aux, aux2)
        mvp[:, k] = mvp[:, k] + aux5
        for j in range(3):
            aux = np.random.normal(mu, abs(mvp[j, k+1].real) * (error_percentage/100), 1)
            aux2 = np.random.normal(mu, abs(mvp[j, k+1].imag) * (error_percentage/100), 1)
            aux3 = np.random.normal(mu, abs(ip[j, k+1].real) * (error_percentage/100), 1)
            aux4 = np.random.normal(mu, abs(ip[j, k+1].imag) * (error_percentage/100), 1)
            aux5[j] = complex(aux, aux2)
            aux6[j] = complex(aux3, aux4)
        mvp[:, k+1] = mvp[:, k+1] + aux5
        ip[:, k+1] = ip[:, k+1] + aux6

        Y[a:a + 3] = np.subtract(mvp[:, k], mvp[:, k+1])  # use 0-1; 1-2; 2-3;
        x[0, a:a + 3] = ip[:, k+1]
        x[1, a:a + 3] = np.sum(ip[:, k+1])
        a += 3

    xtransp = np.transpose(x)

    Beta = np.dot(np.dot(np.linalg.inv(np.dot(x, xtransp)), x), Y.transpose())

    Classi[0] = abs(Beta[0].real/z_lines_perkm[0].real - Beta[0].imag/z_lines_perkm[0].imag)
    Classi[1] = abs(Beta[0].real/z_lines_perkm[1].real - Beta[0].imag/z_lines_perkm[1].imag)
    Classi[2] = abs(Beta[0].real/z_lines_perkm[2].real - Beta[0].imag/z_lines_perkm[2].imag)
    minimo = min(Classi)

    if minimo == Classi[0] and k == 0:
        right += 1
    elif minimo == Classi[1] and k == 1:
        right += 1
    elif minimo == Classi[2] and k == 2:
        right += 1
    else:
        wrong += 1
print("ACC = ", (right/(right + wrong)))
