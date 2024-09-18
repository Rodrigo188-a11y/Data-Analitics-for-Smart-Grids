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
z = np.multiply([complex(0.1, 0.05), complex(0.15, 0.07), complex(0.2, 0.1)], netFactor)
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
Y = np.zeros((12-1, 3), complex)
x = np.zeros((4, 11*3), complex)
Beta = np.zeros(4, complex)
ref = np.zeros(3, complex)

######   Computing the impedances for the first node  ###########################
count = 0
for i in range(m):
    # Power in each instant
    si = [[0, 0, s[i, 2], 0], [0, 0, s[i, 1], 0], [0, s[i, 0], 0, s[i, 3]]]  # Power in each bus and in each phase
    # Load Flow
    mvp, ip = pf3ph(topo, z, si, vr, el, ni, al)  # Compute the power flow
    if i == 0:
        ref[:] = abs(np.subtract(mvp[:, 0], mvp[:, 1]))
        ref_n = abs(np.multiply(np.sum(ip[:, 0]), z[0]))
        P_phases_ref = np.multiply(mvp[:, 0], np.conj(ip[:, 0])).real
        Q_phases_ref = np.multiply(mvp[:, 0], np.conj(ip[:, 0])).imag
        mvp_n = abs(np.multiply(np.sum(ip[:, 0]), z[0]))
        P_n_ref = np.multiply(mvp_n, np.conj(np.sum(ip[:, 0]))).real
        Q_n_ref = np.multiply(mvp_n, np.conj(np.sum(ip[:, 0]))).imag
    else:
        Y[i-1, :] = np.subtract(ref, abs(np.subtract(mvp[:, 0], mvp[:, 1])))
        P_phases = np.multiply(mvp[:, 0], np.conj(ip[:, 0])).real
        Q_phases = np.multiply(mvp[:, 0], np.conj(ip[:, 0])).imag
        mvp_n = abs(np.multiply(np.sum(ip[:, 0]), z[0]))
        P_n = np.multiply(mvp_n, np.conj(np.sum(ip[:, 0]))).real
        Q_n = np.multiply(mvp_n, np.conj(np.sum(ip[:, 0]))).imag

        x[0, count:count + 3] = np.divide(np.subtract(P_phases_ref, P_phases), abs(np.subtract(mvp[:, 0], mvp[:, 1])))
        x[1, count:count + 3] = np.divide(np.subtract(P_n_ref, P_n), mvp_n)
        x[2, count:count + 3] = np.divide(np.subtract(Q_phases_ref, Q_phases), abs(np.subtract(mvp[:, 0], mvp[:, 1])))
        x[3, count:count + 3] = np.divide(np.subtract(Q_n_ref, Q_n), mvp_n)
        """
        x[0, count:count + 3] = np.divide(np.subtract(P_phases_ref, P_phases), ref)
        x[1, count:count + 3] = np.divide(np.subtract(P_n_ref, P_n), ref_n)
        x[2, count:count + 3] = np.divide(np.subtract(Q_phases_ref, Q_phases), ref)
        x[3, count:count + 3] = np.divide(np.subtract(Q_n_ref, Q_n),  ref_n)
        """

        count += 3
Y = Y.flatten()
xtransp = np.transpose(x)

Beta = np.dot(np.dot(np.linalg.inv(np.dot(x, xtransp)), x), Y.transpose())
print(z)
print(Beta)
exit()
avg = 0
for i in range(3):
    for j in range(3):
        if i != j:
            avg = Beta[i, j] + avg
avg = avg / 6

Beta[0, 0] = Beta[0, 0] - avg
Beta[1, 1] = Beta[1, 1] - avg
Beta[2, 2] = Beta[2, 2] - avg


print("ERRORS: ")
print("Error phase a = ", (abs((Beta[0, 0] - z[1])/z[1]) * 100))
print("Error phase b = ", (abs((Beta[1, 1] - z[1])/z[1]) * 100))
print("Error phase c = ", (abs((Beta[2, 2] - z[1])/z[1]) * 100))
print("Error neutral = ", (abs((avg - z[1]/2)/z[1]/2) * 100))

"""
# extract real part
x = [ele.real for ele in Beta.flatten()]

# extract imaginary part
y = [ele.imag for ele in Beta.flatten()]

plt.scatter(x, y, color='blue', marker="*", label='measured')
plt.scatter(z[1].real, z[1].imag, color='red', marker="o", label='real phases impedance')
plt.scatter(z[1].real/2, z[1].imag/2, color='orange', marker="o", label='real neutral impedance')
# plt.xlim((24, 36))
plt.legend()
plt.xlabel('Real')
plt.ylabel('Imaginary')
plt.show()
"""

x1 = np.zeros(3)
y1 = np.zeros(3)

x1[0] = Beta[0, 0].real
x1[1] = Beta[1, 1].real
x1[2] = Beta[2, 2].real

y1[0] = Beta[0, 0].imag
y1[1] = Beta[1, 1].imag
y1[2] = Beta[2, 2].imag
plt.scatter(x1, y1, color='blue', marker="*", label='measured phases impedance')
plt.scatter(z[1].real, z[1].imag, color='red', marker="o", label='real phases impedance')
# plt.xlim((24, 36))
plt.legend()
plt.xlabel('Real')
plt.ylabel('Imaginary')
plt.show()


x2 = np.zeros(6)
y2 = np.zeros(6)
count = 0
for i in range(3):
    for j in range(3):
        if i != j:
            x2[count] = Beta[i, j].real
            y2[count] = Beta[i, j].imag
            count += 1

plt.scatter(x2, y2, color='green', marker="*", label='measured neutral impedance')
plt.scatter(z[1].real/2, z[1].imag/2, color='orange', marker="o", label='real neutral impedance')
plt.legend()
plt.xlabel('Real')
plt.ylabel('Imaginary')
plt.show()

plt.scatter(x2, y2, color='green', marker="*", label='measured neutral impedance')
plt.scatter(x1, y1, color='blue', marker="*", label='measured phases impedance')
plt.scatter(z[1].real, z[1].imag, color='red', marker="o", label='real phases impedance')
plt.scatter(z[1].real/2, z[1].imag/2, color='orange', marker="o", label='real neutral impedance')
# plt.xlim((24, 36))
plt.legend()
plt.xlabel('Real')
plt.ylabel('Imaginary')
plt.show()
