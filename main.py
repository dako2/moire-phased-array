import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# 1) Grid
N, radius = 256, 5.0
L = 5.0           # aperture half-size (λ)
x = np.linspace(-L, L, N)
X, Y = np.meshgrid(x, x)
dx = x[1] - x[0]
k0 = 2*np.pi

# 2) Phase generators
phi_r = np.pi/4
ks = 10*k0

def gb_linear(X, Y):
    return ks*(np.cos(phi_r)*X + np.sin(phi_r)*Y)

# --- 2. Archimedean spiral generator ---
a_spiral, b_spiral = 5.0, 10.0  # radial and angular coefficients
def gb_archimedean(X, Y):
    r = np.sqrt(X**2 + Y**2)
    theta = np.arctan2(Y, X)
    return a_spiral * r + b_spiral * theta

def gb_quadratic(X, Y, beta=0.5):
    return beta*k0**2*(X**2 + Y**2)

# 2. Spherical-wave generator (feed at z = f)
def gb_spherical(X, Y, f=.5):
    return k0 * np.sqrt(X**2 + Y**2 + f**2)

# 3) Transforms + param sweep
frames = 100
transforms = [
    ("Rotate",   lambda X,Y,a: (np.cos(a)*X - np.sin(a)*Y, np.sin(a)*X + np.cos(a)*Y),
                 {'a': np.linspace(0,2*np.pi,frames)}, gb_linear),

    ("Translate (spherical)",   # move the feed laterally under a spherical gb
        lambda X,Y,dx,dy: (X - dx, Y - dy),
        {
            'dx': 0.5 * np.cos(np.linspace(0, 2*np.pi, frames)),
            'dy': 0.5 * np.sin(np.linspace(0, 2*np.pi, frames))
        },
        gb_spherical
    ),

    ("Translate",lambda X,Y,dx,dy: (X-dx, Y-dy),
                 {'dx':0.5*np.cos(np.linspace(0,2*np.pi,frames)),
                  'dy':0.5*np.sin(np.linspace(0,2*np.pi,frames))},
                 gb_quadratic),
    ("Shear",    lambda X,Y,s: (X+s*Y, Y),
                 {'s': np.linspace(-1,1,frames)}, gb_linear),
    ("Scale",    lambda X,Y,a,b: (a*X, b*Y),
                 {'a':1+0.5*np.cos(np.linspace(0,2*np.pi,frames)),
                  'b':1+0.5*np.sin(np.linspace(0,2*np.pi,frames))},
                 gb_linear),
    ("Bend",     lambda X,Y,R: (X, R*np.arcsin(np.clip(Y/R,-1,1))),
                 {'R':np.linspace(1.2,3,frames)}, gb_linear),
    ("Twist",    lambda X,Y,tau: (X*np.cos(tau*Y)-Y*np.sin(tau*Y),
                                  X*np.sin(tau*Y)+Y*np.cos(tau*Y)),
                 {'tau':np.linspace(-20,20,frames)}, gb_archimedean)
]

# 4) Circular window
#window = (X**2+Y**2<=1).astype(float)
window = (X**2 + Y**2 <= radius**2).astype(float)

# 5) Frequency axes
fx = np.fft.fftshift(np.fft.fftfreq(N, d=dx))
u = (2*np.pi*fx)/k0
v = u.copy()

# 6) Compute pattern
def compute_pattern_db(T, params, gb):
    gs = gb(X, Y)
    Xm, Ym = T(X, Y, **params)
    gm = gb(Xm, Ym)
    A = window * np.exp(1j*(gs-gm))
    F = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(A)))
    P = np.abs(F)**2
    P /= P.max()
    return 10*np.log10(P + 1e-12)

# 7) Setup figure
fig, axes = plt.subplots(len(transforms),2,figsize=(8,3*len(transforms)))
moire_ims, beam_ims = [], []
for (ax_m, ax_b), (title,_,_,gb) in zip(axes, transforms):
    ax_m.set_title(f"{title} moiré"); ax_m.axis('off')
    im_m = ax_m.imshow(np.zeros((N,N,3)), extent=(-L,L,-L,L), origin='lower')
    moire_ims.append((im_m,gb))
    ax_b.set_title(f"{title} beam"); ax_b.set_xlabel('u'); ax_b.set_ylabel('v')
    im_b = ax_b.imshow(np.zeros((N,N)), extent=(u.min(),u.max(),v.min(),v.max()),
                       origin='lower', vmin=-40, vmax=0, cmap='viridis')
    beam_ims.append((im_b,gb))

plt.tight_layout()

# 8) Animation update
def update(fr):
    for (im_m,gb_m),(im_b,gb_b),(title,T,params,_) in zip(moire_ims, beam_ims, transforms):
        p = {k:v[fr] for k,v in params.items()}
        # moiré: overlay red=gs, blue=gm
        #gs = gb_m(X,Y) % (2*np.pi)/(2*np.pi)
        Xm,Ym = T(X,Y,**p)
        gm = gb_m(Xm,Ym) % (2*np.pi)/(2*np.pi)
        rgb = np.zeros((N,N,3))
        #rgb[...,0]=gs
        rgb[...,2]=gm
        im_m.set_data(rgb)
        # beam
        Pdb = compute_pattern_db(T,p,gb_b)
        im_b.set_data(Pdb)
    return [im for im,_ in moire_ims] + [im for im,_ in beam_ims]

ani = FuncAnimation(fig, update, frames=frames, interval=100, blit=True)
plt.close(fig)
ani.save('moire_scan.gif', writer='imagemagick', fps=30)
