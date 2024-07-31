import numpy as np

def hat (v) :
    return np.array([
        [0, -v],
        [v, 0]
    ])

def vee (X) :
    return X[1, 0]

def exp (x) :
    return np.array([
        [np.cos(x), -np.sin(x)],
        [np.sin(x), np.cos(x)]
    ])

def log (R) :
    # Returns value in (-pi, pi]
    return np.arctan2(R[1, 0], R[0, 0])

R = 2
r = 1

# Plot a torus
def torus(ax, n=20, m=20):
    th1 = np.linspace(0, 2 * np.pi, n)
    th2 = np.linspace(0, 2 * np.pi, m)
    th1, th2 = np.meshgrid(th1, th2)
    x = (R + r * np.cos(th1)) * np.cos(th2)
    y = (R + r * np.cos(th1)) * np.sin(th2)
    z = r * np.sin(th1)
    ax.plot_wireframe(x, y, z, color='k', alpha=0.5)
    ax.set_xlim3d(-R, R)
    ax.set_zlim3d(-R/2, R/2)
    ax.set_ylim3d(-R, R)
    ax.view_init(45, 0)
    ax.axis('off')

def plot_torus_point (ax, th1, th2, color='r'):
    x = (R + r * np.cos(th1)) * np.cos(th2)
    y = (R + r * np.cos(th1)) * np.sin(th2)
    z = r * np.sin(th1)
    ax.scatter(x, y, z, color=color)

def plot_torus_interval(ax, ulth1, olth1, ulth2, olth2, n=10, m=10) :
    th1 = np.linspace(ulth1, olth1, n)
    th2 = np.linspace(ulth2, olth2, m)
    th1, th2 = np.meshgrid(th1, th2)
    x = (R + r * np.cos(th1)) * np.cos(th2)
    y = (R + r * np.cos(th1)) * np.sin(th2)
    z = r * np.sin(th1)
    ax.plot_surface(x, y, z, color='k', alpha=1.)

# Plot the unit circle
def unit_circle(ax, n=100):
    th = np.linspace(0, 2 * np.pi, n)
    x = np.cos(th)
    y = np.sin(th)
    ax.plot(x, y, color='k', alpha=0.5)
    ax.axis('equal')
    ax.axis('off')

def plot_circle_point (ax, th, color='r'):
    x = np.cos(th)
    y = np.sin(th)
    ax.scatter(x, y, color=color)

def plot_circle_interval(ax, ulth, olth, n=100, **kwargs) :
    th = np.linspace(ulth, olth, n)
    x = np.cos(th)
    y = np.sin(th)
    # kwargs.setdefault('color', 'k')
    kwargs.setdefault('alpha', 0.9)
    line, = ax.plot(x, y, lw=10, **kwargs)

    # endline = 0.1
 
    # Draw the line
    # ax.plot([np.cos(ulth) - endline * np.sin(ulth), np.cos(ulth) + endline * np.sin(ulth)], 
    #         [np.sin(ulth) + endline * np.cos(ulth), np.sin(ulth) - endline * np.cos(ulth)], 
    #         lw=4, color=line.get_color(), **kwargs)

    # ax.plot([np.cos(ulth) - endline * np.sin(ulth), np.sin(ulth) + endline * np.cos(ulth)], 
    #         [np.cos(ulth)  endline * np.sin(ulth), np.sin(ulth) - endline * np.cos(ulth)], 
    #         lw=4, color=line.get_color(), **kwargs)
    # ax.plot(np.cos(ulth) - endline * np.sin(ulth), np.sin(ulth) + endline * np.cos(ulth))

    ax.axis('equal')
    ax.axis('off')

from dataclasses import dataclass
@dataclass
class TangentInterval :
    xc: np.ndarray
    Thl: np.ndarray
    Thu: np.ndarray
    def plot (self, ax, N=10, **kwargs) :
        # kwargs.setdefault('color', 'black')
        kwargs.setdefault('alpha', 1.)
        for i in range(len(self.xc)) :
            lxci = log(self.xc[i])
            plot_circle_interval(ax, self.Thl[i] + lxci, self.Thu[i] + lxci, **kwargs)

    def plot_torus (self, ax, **kwargs) :
        plot_torus_interval(ax, log(self.xc[0])+self.Thl[0], log(self.xc[0])+self.Thu[0], 
                                log(self.xc[1])+self.Thl[1], log(self.xc[1])+self.Thu[1], **kwargs)


