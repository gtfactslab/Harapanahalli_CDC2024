import numpy as np
from Tn import *
import matplotlib.pyplot as plt
import os
from tqdm import tqdm
import time

np.random.seed(0)

N = 2
h = 0.02
tf = 3.
EPS = 1e-6
mc_N = 50

def f (x) :
    return x

omega = [5., 2.]
# omega = np.zeros(N)

def A (x) :
    # Expects x to be a list of 2x2 matrices, returns thdot in Lie algebra
    return np.array([
        # omega[i] + np.sum([np.sin(log(x[j]@x[i].T)/2) for j in range(N)])
        omega[i] + np.sum([log(x[j]@x[i].T) for j in range(N)])
        for i in range(N)
    ])

def reach_rkmk4 (y:TangentInterval, t, h, recenter=True) :
    ulF = []
    olF = []
    ulOmk = y.Thl
    olOmk = y.Thu

    for k in range(4) :
        # No dexpinv here, since we have an abelian Lie group
        ulFk = h*A([y.xc[i]@exp(ulOmk[i]) for i in range(N)])
        olFk = h*A([y.xc[i]@exp(olOmk[i]) for i in range(N)])
        ulF.append(ulFk)
        olF.append(olFk)
        ulOmk = ulOmk + ulFk*h/2
        olOmk = olOmk + olFk*h/2

    ulOm = y.Thl + (ulF[0] + 2*ulF[1] + 2*ulF[2] + ulF[3])/6
    olOm = y.Thu + (olF[0] + 2*olF[1] + 2*olF[2] + olF[3])/6

    if recenter :
        Om = (ulOm + olOm)/2
        xc = [y.xc[i]@exp(Om[i]) for i in range(N)]
        return TangentInterval(xc, ulOm - Om, olOm - Om)
    else :
        return TangentInterval(y.xc, ulOm, olOm)

def reach (y0, h, N, method=reach_rkmk4) :
    y = [y0]
    for i in range(N) :
        yt = y[-1]
        Thcent = log(yt.xc[0]@yt.xc[1].T)
        # Thwidth = np.asarray(yt.Thu) - np.asarray(yt.Thl)
        Thwidth = yt.Thu - yt.Thl
        if np.any(abs(Thcent) + Thwidth > np.pi) :
            raise Exception('Left Monotone Region')
        else :
            y.append(method(y[-1], i*h, h))
    return y

# y0 = TangentInterval([exp(Th) for Th in np.random.uniform(-np.pi,np.pi,N)], 
#                      [-0.1 for _ in range(N)], [0.1 for _ in range(N)])
y0 = TangentInterval([exp(np.pi/2), exp(np.pi)], 
                     np.array([-0.6,-0.1]), np.array([0.6, 0.1]))
print(y0)

times = []

for i in tqdm(range(100)) :
    _t0 = time.time()
    yt = reach(y0, h, round(tf/h))
    _tf = time.time()
    times.append(_tf - _t0)

print(f'Computed reachable set in: {np.mean(times):.5f} +/- {np.std(times):.5f} s')

def th_rkmk4 (y, h) :
    F = []
    Th = np.zeros(len(y))
    c = [0, 1/2, 1/2, 1]

    for k in range(4) :
        Fk = h*A([y[i]@exp(Th[i]) for i in range(N)])
        Th = Fk*h/2
        F.append(Fk)

    return (F[0] + 2*F[1] + 2*F[2] + F[3])/6

def int_rkmk4 (y, t, h) :
    F = []
    Th = np.zeros(len(y))
    c = [0, 1/2, 1/2, 1]

    for k in range(4) :
        Fk = h*A([y[i]@exp(Th[i]) for i in range(N)])
        Th = Fk*h/2
        F.append(Fk)

    Om = (F[0] + 2*F[1] + 2*F[2] + F[3])/6

    return [y[i]@exp(Om[i]) for i in range(N)]

def integrate (y0, h, N, method=int_rkmk4) :
    y = [y0]
    for i in range(N) :
        y.append(method(y[-1], i*h, h))
    return y

mc_Th0s = np.random.uniform(y0.Thl, y0.Thu, (mc_N, N))
mc_y0s = [[y0.xc[i]@exp(Th[i]) for i in range(len(Th))] for Th in mc_Th0s]
mc_yts = [integrate(y0, h, round(tf/h), int_rkmk4) for y0 in mc_y0s]

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "Helvetica",
    "font.size": 14
})

print('Creating frames...')

os.makedirs('pngs1', exist_ok=True)
os.system('rm pngs1/*')
os.makedirs('pdfs1', exist_ok=True)
os.system('rm pdfs1/*')

os.makedirs('pngs2', exist_ok=True)
os.system('rm pngs2/*')
os.makedirs('pdfs2', exist_ok=True)
os.system('rm pdfs2/*')

fig1 = plt.figure(figsize=(5,10), dpi=100)
ax1 = fig1.add_subplot(211)

# fig2 = plt.figure(figsize=(10, 10), dpi=100)
ax2 = fig1.add_subplot(212, projection='3d')

savepdfs = np.linspace(0, len(yt)-1, 8, True).astype(int)

colors = ['tab:blue', 'tab:orange']
# fig1.tight_layout()

for t, y in tqdm(enumerate(yt), total=len(yt)) :
    ax1.clear()
    unit_circle(ax1)
    y.plot(ax1, alpha=0.6)
    # ax1.set_title(f't = {t*h:.2f}', fontsize=50)
    ax1.set_title(f'$t = {t*h:.2f}$', fontsize=30)

    # for i in range(N) :
    #     ax1.scatter(*(np.array([mc_yts[k][t][i][:,0] for k in range(mc_N)]).T), marker='o', facecolor='none', edgecolor=colors[i], s=80)

    # ax1.set_xlim(-1.25, 1.25)
    # ax1.set_ylim(-1.25, 1.25)

    ax2.clear()
    torus(ax2)
    # plot_torus_point(ax2, mc_yts[][t][0], mc_yts[:][t][1], color='r')
    y.plot_torus(ax2)
    # ax2.set_title(f't = {t*h:.2f}', fontsize=50)

    fig1.tight_layout()

    fig1.savefig(f'pngs1/{t:04d}.png')
    # fig2.tight_layout()
    # fig2.savefig(f'pngs2/{t:04d}.png')

    if t in savepdfs :
        fig1.savefig(f'pdfs1/{t:04d}.pdf')
        # fig2.savefig(f'pdfs2/{t:04d}.pdf') 

os.system(f'ffmpeg -y -r {round(1/h)} -i pngs1/%04d.png -vcodec h264 Tn1.mp4')
# os.system(f'ffmpeg -y -r {round(1/h)} -i pngs2/%04d.png -vcodec h264 Tn2.mp4')

plt.show()
