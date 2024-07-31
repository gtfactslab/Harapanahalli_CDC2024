import numpy as np
from SO3 import *
import matplotlib.pyplot as plt
import interval

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "Helvetica",
    "font.size": 14
})

N = 7

fig = plt.figure(figsize=(10,10), dpi=100)
ax = fig.add_subplot(111, projection='3d')
ax.view_init(elev=40, azim=45)

Thl, Thu = np.array([-0.1,-0.1,-0.1]), np.array([0.1,0.1,0.1])

R1 = TangentInterval(np.eye(3), Thl, Thu)
R1.plot(ax, N, False, color='tab:blue', alpha=0.5)

omega = np.array([0.3, 0.3, 0.3])

R2 = TangentInterval(expm(omega), Thl, Thu)

R3 = TangentInterval(np.eye(3), omega+Thl, omega+Thu)
R3.plot(ax, N, False, color='tab:blue', alpha=0.5)

Thout = BCH(interval.get_iarray(R3.Thl, R3.Thu), -omega)
R4 = TangentInterval(R2.xc, *interval.get_lu(Thout))
R4.plot(ax, N, True, color='tab:purple', alpha=0.5)

fig.tight_layout()
fig.savefig('compare.pdf')

plt.show()
