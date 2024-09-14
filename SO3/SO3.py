import numpy as np
from numpy import sin, cos, arcsin, arccos, sqrt
import interval
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

"""
Rather than use the closed form dexpinv and BCH, we truncate to 4th order.
This is because of the trig expressions in the denominator. 
When using intervals, these denominators can include zero, and blow up the result.
"""

def brak (u, v) :
    return np.cross(u, v)

def hat (v) :
    return np.array([[0, -v[2], v[1]],
                     [v[2], 0, -v[0]],
                     [-v[1], v[0], 0]])

def vee (X) :
    return np.array([X[2, 1], X[0, 2], X[1, 0]])

def expm (v) :
    th = np.linalg.norm(v)
    vth = th / 2
    if th < 1e-6 :
        return np.eye(3) + hat(v)
    else :
        return (np.eye(3) 
            + (np.sin(th)/th)*hat(v)
            + 0.5*((np.sin(vth)/vth)**2)*hat(v)@hat(v))

def dexpinv (x, y) :
    # th = np.linalg.norm(x)
    # vth = th / 2
    # def cot(x) :
    #     return 1/np.tan(x)

    # if th < 1e-6 :
    #     return y - 0.5*brak(x, y)
    # else :
    #     return (
    #         y
    #         - 0.5*brak(x, y)
    #         - ((th*cot(vth) - 2)/(2*th**2))*brak(x, brak(x, y))
    #     )

    bxy = brak(x,y)
    return y + 0.5*bxy + (1/12)*brak(x, bxy)


def BCH (u, v) :
    # return (
    #     Th1 
    #     + Th2 
    #     + (1/2)*brak(Th1, Th2) 
    #     + (1/12)*brak(Th1, brak(Th1, Th2)) 
    #     - (1/12)*brak(Th2, brak(Th1, Th2))
    #     # + (1/12)*brak(Th1 - Th2, brak(Th1, Th2)) 
    # )
    th = sqrt(np.dot(u, u))
    ph = sqrt(np.dot(v, v))
    # if th.dtype == np.interval :
    if ph.dtype == np.interval :
        buv = brak(u, v)
        buuv = brak(u, buv)
        return (
            u
            + v 
            + (1/2)*buv
            + (1/12)*(buuv - brak(v, buv))
            - (1/24)*(brak(v, buuv))
        )
    else :
        ang = arccos(np.dot(u, v)/(th*ph))
        a = sin(th)*cos(ph/2)**2 - sin(ph)*sin(th/2)**2*cos(ang)
        b = sin(ph)*cos(th/2)**2 - sin(th)*sin(ph/2)**2*cos(ang)
        c = 0.5*sin(th)*sin(ph) - 2*sin(th/2)**2*sin(ph/2)**2*cos(ang)
        d = sqrt(a**2 + b**2 + 2*a*b*cos(ang) + c**2*sin(ang)**2)
        alpha = (arcsin(d)/d)*(a/th)
        beta  = (arcsin(d)/d)*(b/ph)
        gamma = (arcsin(d)/d)*(c/(th*ph))
        return (
            alpha*u
            + beta*v
            + gamma*brak(u, v)
        )


def plot_R (ax, R, **kwargs) :
    """Plot the rotation as a 3D arrow."""
    if kwargs.get('color', 'none') is 'none' :
        xarrow = ax.quiver(0, 0, 0, *R[:, 0], color='red', label='x')
        yarrow = ax.quiver(0, 0, 0, *R[:, 1], color='green', label='y')
        zarrow = ax.quiver(0, 0, 0, *R[:, 2], color='blue', label='z')
    else :
        xarrow = ax.quiver(0, 0, 0, *R[:, 0], **kwargs)
        yarrow = ax.quiver(0, 0, 0, *R[:, 1], **kwargs)
        zarrow = ax.quiver(0, 0, 0, *R[:, 2], **kwargs)
    # kwargs.setdefault('width', 0.01)
    # xarrow = ax.quiver(0, 0, 0, *R[:, 0], **kwargs)
    # yarrow = ax.quiver(0, 0, 0, *R[:, 1], **kwargs)
    # zarrow = ax.quiver(0, 0, 0, *R[:, 2], **kwargs)
    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)
    ax.set_zlim(-1, 1)
    ax.set_xlabel('$x$')
    ax.set_ylabel('$y$')
    ax.set_zlabel('$z$')
    return xarrow, yarrow, zarrow

def plot_points_R (ax, R, **kwargs) :
    """Plot the rotation as a 3D arrow."""
    kwargs.setdefault('color', 'black')
    # kwargs.setdefault('width', 0.01)
    ax.scatter(*R[:, 0], **kwargs)
    ax.scatter(*R[:, 1], **kwargs)
    ax.scatter(*R[:, 2], **kwargs)

def plot_tangent_int_R (ax, R, Thl, Thu, N=5, **kwargs) :
    kwargs.setdefault('color', 'black')
    plot_R(ax, R, **kwargs)
    samples = np.array(np.meshgrid(*[np.linspace(Thli, Thui, N, True) for Thli, Thui in zip(Thl, Thu)]))
    samples = samples.reshape(3, -1).T

    xpoints = []
    ypoints = []
    zpoints = []
    for s in samples :
        y = R@expm(s)
        xpoints.append(y[:,0])
        ypoints.append(y[:,1])
        zpoints.append(y[:,2])

    surfx = ax.scatter(*(np.array(xpoints).T), s=1, **kwargs)
    surfy = ax.scatter(*(np.array(ypoints).T), s=1, **kwargs)
    surfz = ax.scatter(*(np.array(zpoints).T), s=1, **kwargs)
    # surfx = ax.plot_trisurf(*(np.array(xpoints).T), **kwargs)
    # surfy = ax.plot_trisurf(*(np.array(ypoints).T), **kwargs)
    # surfz = ax.plot_trisurf(*(np.array(zpoints).T), **kwargs)


def draw_iarray_3d (ax, x, xi=0, yi=1, zi=2, **kwargs) :
    xl, xu = interval.get_lu(x)
    Xl, Yl, Zl = xl[(xi,yi,zi),]
    Xu, Yu, Zu = xu[(xi,yi,zi),]
    poly_alpha = kwargs.pop('poly_alpha', 0.)
    kwargs.setdefault('color', 'tab:blue')
    kwargs.setdefault('lw', 0.75)
    faces = [ \
        np.array([[Xl,Yl,Zl],[Xu,Yl,Zl],[Xu,Yu,Zl],[Xl,Yu,Zl],[Xl,Yl,Zl]]), \
        np.array([[Xl,Yl,Zu],[Xu,Yl,Zu],[Xu,Yu,Zu],[Xl,Yu,Zu],[Xl,Yl,Zu]]), \
        np.array([[Xl,Yl,Zl],[Xu,Yl,Zl],[Xu,Yl,Zu],[Xl,Yl,Zu],[Xl,Yl,Zl]]), \
        np.array([[Xl,Yu,Zl],[Xu,Yu,Zl],[Xu,Yu,Zu],[Xl,Yu,Zu],[Xl,Yu,Zl]]), \
        np.array([[Xl,Yl,Zl],[Xl,Yu,Zl],[Xl,Yu,Zu],[Xl,Yl,Zu],[Xl,Yl,Zl]]), \
        np.array([[Xu,Yl,Zl],[Xu,Yu,Zl],[Xu,Yu,Zu],[Xu,Yl,Zu],[Xu,Yl,Zl]]) ]
    for face in faces :
        ax.plot3D(face[:,0], face[:,1], face[:,2], **kwargs)
        kwargs['alpha'] = poly_alpha
        ax.add_collection3d(Poly3DCollection([face], **kwargs))


from dataclasses import dataclass
@dataclass
class TangentInterval :
    xc: np.ndarray
    Thl: np.ndarray
    Thu: np.ndarray
    def plot (self, ax, N=7, mesh=False, **kwargs) :
        kwargs.setdefault('color', 'none')
        plot_R(ax, self.xc, color=kwargs.get('color'))
        samples = np.array(np.meshgrid(*[np.linspace(Thli, Thui, N) for Thli, Thui in zip(self.Thl, self.Thu)]))
        samples = samples.reshape(3, -1).T

        xpoints = []
        ypoints = []
        zpoints = []
        for s in samples :
            y = self.xc@expm(s)
            xpoints.append(y[:,0])
            ypoints.append(y[:,1])
            zpoints.append(y[:,2])

        if kwargs.get('color', 'none') is 'none' :
            if mesh :
                surfx = ax.plot_trisurf(*(np.array(xpoints).T), color='red')
                surfy = ax.plot_trisurf(*(np.array(ypoints).T), color='green')
                surfz = ax.plot_trisurf(*(np.array(zpoints).T), color='blue')
            else :
                surfx = ax.scatter(*(np.array(xpoints).T), s=2, color='red')
                surfy = ax.scatter(*(np.array(ypoints).T), s=2, color='green')
                surfz = ax.scatter(*(np.array(zpoints).T), s=2, color='blue')
        else :
            if mesh :
                surfx = ax.plot_trisurf(*(np.array(xpoints).T), **kwargs)
                surfy = ax.plot_trisurf(*(np.array(ypoints).T), **kwargs)
                surfz = ax.plot_trisurf(*(np.array(zpoints).T), **kwargs)
            else :
                surfx = ax.scatter(*(np.array(xpoints).T), s=2, **kwargs)
                surfy = ax.scatter(*(np.array(ypoints).T), s=2, **kwargs)
                surfz = ax.scatter(*(np.array(zpoints).T), s=2, **kwargs)

