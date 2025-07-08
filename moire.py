import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import os

# Set Matplotlib to use LaTeX for text rendering (optional, but can improve quality)
# You may need to have a LaTeX distribution like MiKTeX (Windows), MacTeX (macOS),
# or TeX Live (Linux) installed for this to work. If it causes errors, comment it out.
# plt.rcParams.update({
#     "text.usetex": True,
#     "font.family": "serif",
#     "font.serif": ["Computer Modern Roman"],
# })

# 1) Grid
N, radius = 64, 1.0
L = 1.0           # aperture half-size (λ)
x = np.linspace(-L, L, N)
X, Y = np.meshgrid(x, x)
dx = x[1] - x[0]
k0 = 2*np.pi

# 2) Phase generators
phi_r = np.pi/4
ks = 10*k0

def gb_linear(X, Y):
    return ks*(np.cos(phi_r)*X + np.sin(phi_r)*Y)

a_spiral, b_spiral = 2.0, 5.0
def gb_archimedean(X, Y):
    r = np.sqrt(X**2 + Y**2)
    theta = np.arctan2(Y, X)
    return a_spiral * r + b_spiral * theta

def gb_quadratic(X, Y, beta=0.5):
    return beta*k0**2*(X**2 + Y**2)

def gb_spherical(X, Y, f=.5):
    return k0 * np.sqrt(4*X**2 + 4*Y**2 + f**2)

# 3) Transforms + param sweep with LaTeX formulas
frames = 100
transforms = [
    ("Rotate",   lambda X,Y,a: (np.cos(a)*X - np.sin(a)*Y, np.sin(a)*X + np.cos(a)*Y),
                 {'a': np.linspace(0,2*np.pi,frames)}, gb_linear,
                 r"(X', Y') = (X\cos(a) - Y\sin(a), X\sin(a) + Y\cos(a))"),

    ("Translate (spherical)", lambda X,Y,dx,dy: (X - dx, Y - dy),
                 {'dx': 0.5 * np.cos(np.linspace(0, 2*np.pi, frames)),
                  'dy': 0.5 * np.sin(np.linspace(0, 2*np.pi, frames))},
                 gb_spherical, r"(X', Y') = (X - d_x, Y - d_y)"),

    ("Translate (45 deg)", lambda X, Y, dx, dy: (X - dx, Y - dy),
                 {'dx': (np.linspace(-1.0, 1.0, frames)) * np.cos(np.pi/4),
                  'dy': (np.linspace(-1.0, 1.0, frames)) * np.sin(np.pi/4)},
                 gb_quadratic, r"(X', Y') = (X - d_x, Y - d_y)"),

    ("Shear",    lambda X,Y,s: (X+s*Y, Y),
                 {'s': np.linspace(-1,1,frames)}, gb_linear,
                 r"(X', Y') = (X + sY, Y)"),

    ("Scale",    lambda X,Y,a,b: (a*X, b*Y),
                 {'a':1+0.5*np.cos(np.linspace(0,2*np.pi,frames)),
                  'b':1+0.5*np.sin(np.linspace(0,2*np.pi,frames))},
                 gb_linear, r"(X', Y') = (aX, bY)"),

    ("Bend",     lambda X,Y,R: (X, R*np.arcsin(np.clip(Y/R,-1,1))),
                 {'R':np.linspace(1.2,3,frames)}, gb_linear,
                 r"(X', Y') = (X, R \arcsin(Y/R))"),

    ("Shear-Bend", lambda X,Y,s: (X, Y + s*X),
                 {'s': np.linspace(-1,1,frames)}, gb_linear,
                 r"(X', Y') = (X, Y + sX)"),

    ("Twist",    lambda X,Y,tau: (X*np.cos(tau*Y)-Y*np.sin(tau*Y),
                                  X*np.sin(tau*Y)+Y*np.cos(tau*Y)),
                 {'tau':np.linspace(-10,10,frames)}, gb_archimedean,
                 r"(X', Y') = (X\cos(\tau Y) - Y\sin(\tau Y), X\sin(\tau Y) + Y\cos(\tau Y))")
]

# 4) Circular window
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

output_dir = './results'
os.makedirs(output_dir, exist_ok=True)

for name, T, params, gb, latex_str in transforms:
    fig, (ax_m, ax_b) = plt.subplots(1, 2, figsize=(8, 4.5))

    # --- MODIFICATION: Add LaTeX formula as the title ---
    # Combine the name and the LaTeX string. The '$' signs activate math mode.
    title_with_formula = f"{name}\n${latex_str}$"

    # Add the text to the figure.
    fig.text(0.5, 0.97, title_with_formula,
             ha='center', va='top',
             fontsize=14,  # Increased font size for clarity
             bbox=dict(boxstyle="round,pad=0.5", fc='aliceblue', ec='k', lw=1))
    # --- END OF MODIFICATION ---

    ax_m.set_title(f"Moiré Pattern")
    ax_m.axis('off')
    im_m = ax_m.imshow(np.zeros((N,N,3)), extent=(-L,L,-L,L), origin='lower')

    ax_b.set_title(f"Beam Pattern")
    ax_b.set_xlabel('u'); ax_b.set_ylabel('v')
    im_b = ax_b.imshow(np.zeros((N,N)), extent=(u.min(),u.max(),v.min(),v.max()),
                       origin='lower', vmin=-40, vmax=0, cmap='viridis')

    plt.tight_layout(rect=[0, 0, 1, 0.82]) # Adjust rect to leave space for title

    gs = gb(X,Y)
    gs_norm = (gs % (2*np.pi)) / (2*np.pi)

    def update(fr):
        p = {k: v[fr] for k, v in params.items()}
        Xm, Ym = T(X,Y,**p)
        gm = gb(Xm,Ym)
        gm_norm = (gm % (2*np.pi)) / (2*np.pi)

        rgb = np.zeros((N,N,3))
        rgb[...,0] = gs_norm
        rgb[...,2] = gm_norm
        rgb *= window[..., np.newaxis]
        im_m.set_data(rgb)

        Pdb = compute_pattern_db(T, p, gb)
        im_b.set_data(Pdb)
        return [im_m, im_b]

    ani = FuncAnimation(fig, update, frames=frames, interval=50, blit=True)

    safe_name = name.replace(' ', '_').replace('(', '').replace(')', '')
    output_path = os.path.join(output_dir, f'{safe_name}.gif')

    print(f"Saving animation for '{name}' to {output_path}...")
    ani.save(output_path, writer='imagemagick', fps=30)
    plt.close(fig)

print("\nAll animations saved successfully.")