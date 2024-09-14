import numpy as np
import interval
from SO3 import *
import os
import matplotlib.pyplot as plt
from tqdm import tqdm
import time

"""
For this example, we keep Lie algebra elements in the basis.
"""

h = 0.02
tf = 5.
BCHf = 1.*h
BCHm = round(BCHf/h)

np.random.seed(0)

def ut (t) :
    return 1.*np.array([(5-t)/5, (1-(t/5)**2), np.sin(np.pi*t/2)])

upm = 0.01

def randut(t) :
    ut_t = ut(t)
    return np.random.uniform(ut_t-upm, ut_t+upm)

def utl(t) :
    return ut(t) - upm

def utu(t) :
    return ut(t) + upm

def A (u) :
    return u

def reach_rkmk4 (y:TangentInterval, t, h, recenter=False) :
    ulF = []
    olF = []
    Thl0 = y.Thl
    Thu0 = y.Thu
    ulTh = Thl0
    olTh = Thu0
    c = [0, 1/2, 1/2, 1]

    for k in range(4) :
        ulFk = np.zeros(3)
        olFk = np.zeros(3)
        uint = interval.get_iarray(A(ut(t + c[k]*h) - upm), A(ut(t + c[k]*h) + upm))

        # Embedding system
        for i in range(3) :
            # _Thi is the lower ith face
            _olTh = np.copy(olTh); _olTh[i] = ulTh[i]
            _Thi = interval.get_iarray(ulTh, _olTh)
            tmp_lower, _ = interval.get_lu(h*dexpinv(_Thi, uint))
            ulFk[i] = tmp_lower[i]
            
            # _Thi is the upper ith face
            _ulTh = np.copy(ulTh); _ulTh[i] = olTh[i]
            _Thi = interval.get_iarray(_ulTh, olTh)
            _, tmp_upper = interval.get_lu(h*dexpinv(_Thi, uint))
            olFk[i] = tmp_upper[i]

        ulTh = ulFk*h/2
        olTh = olFk*h/2
        ulF.append(ulFk)
        olF.append(olFk)

    Thl = Thl0 + (ulF[0] + 2*ulF[1] + 2*ulF[2] + ulF[3])/6
    Thu = Thu0 + (olF[0] + 2*olF[1] + 2*olF[2] + olF[3])/6

    if np.any(np.max(np.abs(Thl)) > np.pi) or np.any(np.max(np.abs(Thu)) > np.pi) :
        raise Exception(f'Left injectivity radius at {t=}.')

    if recenter :
        Thcent = (Thl + Thu)/2
        xc = y.xc@expm(Thcent)
        # Thout = BCH(interval.get_iarray(Thl, Thu), -Thcent)
        Thout = BCH(-Thcent, interval.get_iarray(Thl, Thu))
        return TangentInterval(xc, *interval.get_lu(Thout))
    else :
        return TangentInterval(y.xc, Thl, Thu)


def reach (y0, h, N, method=reach_rkmk4) :
    y = [y0]
    for i in range(N) :
        y.append(method(y[-1], i*h, h, recenter=(i+1)%BCHm==0))
    return y

y0 = TangentInterval(np.eye(3), np.array([-0.01,-0.01,-0.01]), np.array([0.01,0.01,0.01]))

times = []

for i in tqdm(range(100)) :
    _t0 = time.time()
    yt = reach(y0, h, round(tf/h), reach_rkmk4)
    _tf = time.time()
    times.append(_tf - _t0)

print(f'Computed reachable set in: {np.mean(times):.5f} +/- {np.std(times):.5f} s')


# Monte carlo simulations

def int_rkmk4 (y, t, h) :
    F = []
    Th = np.zeros(3)
    c = [0, 1/2, 1/2, 1]

    for k in range(4) :
        Fk = h*dexpinv(Th, A(randut(t + c[k]*h)))
        Th = Fk*h/2
        F.append(Fk)

    return y@expm((F[0] + 2*F[1] + 2*F[2] + F[3])/6)


def integrate (y0, h, N, method=int_rkmk4) :
    y = [y0]
    for i in range(N) :
        y.append(method(y[-1], i*h, h))
    return y

print(len(yt))
print(yt[0])
print(yt[len(yt)//2])
print(yt[-1])
# print(yt[-1]@yt[-1].T)

# input()

mc_y0s = [np.eye(3)@expm(np.random.uniform(y0.Thl, y0.Thu)) for _ in range(100)]
mc_yts = np.array([integrate(y0, h, round(tf/h), int_rkmk4) for y0 in mc_y0s])


plt.rcParams.update({
    "text.usetex": True,
    "font.family": "Helvetica",
    "font.size": 14
})

print('Creating frames...')


os.makedirs('pngs', exist_ok=True)
os.system('rm pngs/*')
os.makedirs('pdfs', exist_ok=True)
os.system('rm pdfs/*')

fig = plt.figure(figsize=(10, 10), dpi=100)
anifig = plt.figure(figsize=(10, 10), dpi=100)
ax = fig.add_subplot(111, projection='3d')
aniax = anifig.add_subplot(111, projection='3d')

plot_R(ax, yt[0].xc)

savepdfs = np.linspace(0, len(yt)-1, 9, True).astype(int)

for i, y in tqdm(enumerate(yt), total=len(yt)) :
    aniax.clear()
    y.plot(aniax)
    aniax.scatter(*mc_yts[:, i, :, 0].T, color='tab:red', s=1)
    aniax.scatter(*mc_yts[:, i, :, 1].T, color='tab:red', s=1)
    aniax.scatter(*mc_yts[:, i, :, 2].T, color='tab:red', s=1)
    aniax.set_title(f'$t = {i*h:.2f}$', fontsize=50)
    # Rotate azim for video
    # aniax.view_init(elev=40, azim=30+i*30*h)
    aniax.view_init(elev=40, azim=30)
    anifig.tight_layout()
    anifig.savefig(f'pngs/{i:04d}.png')
    if i in savepdfs :
        # Still for pdfs
        aniax.view_init(elev=40, azim=30)
        anifig.savefig(f'pdfs/{i:04d}.pdf')

os.system(f'ffmpeg -y -r {round(1/h)} -i pngs/%04d.png -vcodec h264 SO3.mp4')

plt.show()

