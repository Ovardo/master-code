import numpy as np

# cell at index 0: plot a 2D Gaussian (bell surface = PDF)
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # registers 3D projection
import matplotlib.cm as cm
import matplotlib as mpl

mpl.rcParams.update({
    "figure.figsize": (6.5, 4.5),   # width x height in inches
    "font.size": 14,                 # base font size
    "axes.labelsize": 13,
    "axes.titlesize": 13,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "legend.fontsize": 11,
})

def multivariate_gaussian(pos, mu, Sigma):
    """Vectorized multivariate Gaussian PDF for pos shaped (..., 2)."""
    det = np.linalg.det(Sigma)
    inv = np.linalg.inv(Sigma)
    norm = 1.0 / (2 * np.pi * np.sqrt(det))
    diff = pos - mu
    exponent = np.einsum('...i,ij,...j->...', diff, inv, diff)
    return norm * np.exp(-0.5 * exponent)

# parameters (change as desired)
mu = np.array([0.0, 0.0])
Sigma = np.array([[1, 0],
                  [0, 1]])

# grid
lim = 3.0
n = 200
x = np.linspace(-lim, lim, n)
y = np.linspace(-lim, lim, n)
X, Y = np.meshgrid(x, y)
pos = np.dstack((X, Y))

# pdf values
Z = multivariate_gaussian(pos, mu, Sigma)

import matplotlib as mpl
mpl.rcParams['pdf.fonttype'] = 42   # keep TrueType fonts
mpl.rcParams['ps.fonttype']  = 42
mpl.rcParams['svg.fonttype'] = 'none'  # keep text as text in SVG

# plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

ax.contourf(X, Y, Z, zdir='z', offset=0, cmap='gray_r', levels=5, alpha=1.0)

# Mahalanobis radii (k-sigma) → PDF levels (strictly increasing after sort)
k_sigma = np.array([1, 2, 3])
det = np.linalg.det(Sigma)
norm_const = 1.0 / (2 * np.pi * np.sqrt(det))
levels_pdf = np.sort(norm_const * np.exp(-0.5 * k_sigma**2))  # [f(3σ), f(2σ), f(1σ)]

# Filled bands: need 4 increasing bounds for 3 bands (outer→inner)
filled_levels = np.r_[levels_pdf, Z.max()]  # [f(3σ), f(2σ), f(1σ), peak]

band_colors = ["#d6d6d6", "#919191", "#444444"]  # 3 bands


# # filled annuli on z=0
# ax.contourf(
#     X, Y, Z,
#     zdir='z', offset=0,
#     levels=filled_levels,        # strictly increasing
#     colors=band_colors,
#     alpha=0.9, antialiased=True
# )

# # outlines (strictly increasing levels: 3σ, 2σ, 1σ)
# cs = ax.contour(
#     X, Y, Z,
#     zdir='z', offset=0,
#     levels=levels_pdf,           # [f(3σ), f(2σ), f(1σ)]
#     colors='black',
#     linestyles=[':', '--', '-'],
#     linewidths=1.0
# )

# # helper (define once, outside the loop)
# L = np.linalg.cholesky(Sigma)
# def ellipse_point(mu, Sigma, k, theta=0.0):
#     v = np.array([np.cos(theta), np.sin(theta)])
#     return mu + k * (L @ v)

# # label height and a tiny x-offset so the label doesn't sit on the line
# z_label = 0.12
# x_offset = 0.04 * np.sqrt(Sigma[0, 0])  # scale with ellipse size

# for k, txt in [(1, r'$1\sigma$'), (2, r'$2\sigma$'), (3, r'$3\sigma$')]:
#     px, py = ellipse_point(mu, Sigma, k, theta=0.0)

#     # vertical connector from label down to the contour (z=0)
#     ax.plot([px, px], [py, py], [0.0, z_label-0.011],
#             linestyle='--', linewidth=1.2, color='black', alpha=0.9, zorder=5)

#     # place the label slightly to the +x side, at height z_label
#     ax.text(px + x_offset, py, z_label, txt, zdir=None,
#             ha='left', va='center', fontsize=12,
#             bbox=dict(boxstyle='round,pad=0.2', fc='lightgray', ec='none', alpha=1.0))




surf = ax.plot_surface(X, Y, Z, color='white', edgecolor='black', linewidth=0.3, antialiased=True, shade=False, rcount=30, ccount=30, alpha=0.7)


# remove gridlines and make 3D axes panes transparent
ax.grid(False)
# Make panes transparent
ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))


ax.set_xlabel(r'$x$', fontsize=12)
ax.set_ylabel(r'$y$', fontsize=12)
ax.set_zlabel(r'$P(x,y)$', fontsize=12)
ax.set_zlim(0, Z.max() * 1.05)
ax.view_init(elev=27.290806086083744, azim=-52.22792277075064, roll=1.742012537233759)

#ax.set_title('2D Gaussian PDF (bell surface)')
#fig.colorbar(surf, shrink=0.6, aspect=10, label='pdf value')

plt.tight_layout()
plt.show()

# print('ax.azim {}'.format(ax.azim))
# print('ax.elev {}'.format(ax.elev))
# print('ax.roll {}'.format(ax.roll))

#fig.savefig("figures/gaussian_standard.pdf", transparent=True)   # vector PDF
#fig.savefig("figures/gaussian_standard.svg", transparent=True)   # vector SVG




