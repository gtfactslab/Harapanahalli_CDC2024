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

# fig = plt.figure(figsize=(10,10), dpi=100)
fig = plt.figure(figsize=(15,7), dpi=100)

ax = fig.add_subplot(121, projection='3d')
ax.view_init(elev=40, azim=45)

Thl, Thu = np.array([-0.1,-0.1,-0.1]), np.array([0.1,0.1,0.1])

R1 = TangentInterval(np.eye(3), Thl, Thu)
R1.plot(ax, N, False, color='tab:blue', alpha=0.5)

omega = np.array([0.3, 0.3, 0.3])

R2 = TangentInterval(expm(omega), Thl, Thu)

R3 = TangentInterval(np.eye(3), omega+Thl, omega+Thu)
R3.plot(ax, N, False, color='tab:blue', alpha=0.5)

# Thout = BCH(interval.get_iarray(R3.Thl, R3.Thu), -omega)
Thout = BCH(-omega, interval.get_iarray(R3.Thl, R3.Thu))
R4 = TangentInterval(R2.xc, *interval.get_lu(Thout))
R4.plot(ax, N, True, color='tab:purple', alpha=0.5)

ax.set_xlim(-0.3, 1)
ax.set_ylim(-0.3, 1)
ax.set_zlim(-0.3, 1)


ax2 = fig.add_subplot(122, projection='3d')
# ax2.view_init(elev=15, azim=-70)
ax2.view_init(elev=10, azim=-130)
print(Thout)
# draw_iarray_3d(ax2, interval.get_iarray(Thl, Thu), color='tab:blue', lw=2, poly_alpha=0.1)
draw_iarray_3d(ax2, omega + interval.get_iarray(Thl, Thu), color='tab:blue', lw=2, poly_alpha=0.1)
draw_iarray_3d(ax2, Thout, color='tab:purple', lw=2, poly_alpha=0.1)


samples = np.array(np.meshgrid(*[np.linspace(Thli, Thui, N) for Thli, Thui in zip(omega+Thl, omega+Thu)]))
samples = samples.reshape(3, -1).T
points = np.array([BCH(-omega, s) for s in samples])
ax2.scatter(*points.T)

ax2.set_xlabel('$X$')
ax2.set_ylabel('$Y$')
ax2.set_zlabel('$Z$')

fig.tight_layout()
# fig.subplots_adjust(wspace=-0.25)
fig.savefig('compare.pdf')

plt.show()
