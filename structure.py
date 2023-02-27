import matplotlib
from matplotlib import pyplot as plt
from scipy.spatial import Voronoi, voronoi_plot_2d
import numpy as np
import nucleation as nc

def plot_cone_crys(latt: nc.Crystal, plot_initial=False):
    '''Plot particle positions of a crystal on a conic surface.'''
    latt_plot = latt.pos.reshape(latt.N, 2)
    latt_init = latt.init_pos.reshape(latt.N, 2)
    plt.scatter(latt_plot[:, 0], latt_plot[:, 1])
    if plot_initial:
        plt.scatter(latt_init[:, 0], latt_init[:, 1], color='lightgray',
                    zorder=-1)
    format_plot(plt.gca(), latt)
    plt.title('Relaxed crystal structure for $2 \pi \sin \\alpha = {:.3}$'.format(latt.ctheta))

def format_plot(ax: matplotlib.axes.Axes, latt):
    '''Format plot of unrolled conic surface.'''
    latt_plot = latt.pos.reshape(latt.N, 2)
    xlim = np.max(np.abs(latt_plot[:, 0])) * 1.1
    ylimp = np.max(latt_plot[:, 1])
    ylimn = np.min(latt_plot[:, 1])
    xs = np.linspace(-xlim, xlim, 100)
    ax.plot(xs, np.abs(xs * np.tan(np.pi/2 - latt.ctheta/2)), '--k', zorder=-1)
    ax.axis('Equal')
    ax.set_xlim([-xlim, xlim])
    ax.set_ylim([ylimn, ylimp])
    ax.set_xlabel('x')
    ax.set_ylabel('y')

def plot_coordination(latt: nc.Crystal):
    '''Plot coordination number of each particle in the crystal.'''
    points = get_image_pts(latt)
    vor = Voronoi(points)
    vnum = [len(v) for v in vor.regions]
    coord_num = [vnum[i] for i in vor.point_region]
    # normalize colormap and plot voronoi
    norm = matplotlib.colors.Normalize(vmin=4, vmax=8, clip=True)
    mapper = matplotlib.cm.ScalarMappable(norm=norm, cmap='coolwarm')
    voronoi_plot_2d(vor, show_points=True, show_vertices=False, s=1,
                    point_size=10)
    # color finite regions according to coordination number
    for r in range(len(vor.point_region)):
        region = vor.regions[vor.point_region[r]]
        if not -1 in region:
            polygon = [vor.vertices[i] for i in region]
            plt.fill(*zip(*polygon), color=mapper.to_rgba(coord_num[r]))
    format_plot(plt.gca(), latt)

def get_image_pts(latt: nc.Crystal):
    '''Get all points in the crystal, including periodic images.'''
    points_c = latt.pos.reshape(latt.N, 2)
    xs = points_c[:,0]
    ys = points_c[:,1]
    xl, yl = nc.rotate_coord(xs, ys, latt.cth, latt.sth)
    xr, yr = nc.rotate_coord(xs, ys, latt.cth, -latt.sth)
    rpts_all = np.array([xr, yr]).T
    lpts_all = np.array([xl, yl]).T
    rpts = np.array([pt for pt in rpts_all if angr(pt, latt.ctheta)])
    lpts = np.array([pt for pt in lpts_all if angl(pt, latt.ctheta)])
    points = np.concatenate((points_c, rpts, lpts))
    return points

def angl(pt, ctheta):
    '''Filter left-rotated image points that overlap with original points.'''
    ang = np.angle(pt[0]+1j*pt[1])%(2*np.pi)
    if 0.97*(np.pi/2+0.5*ctheta) <= ang <= np.pi*1.2:
        return True
    return False

def angr(pt, ctheta):
    '''Filter right-rotated image points that overlap with original points.'''
    ang = np.angle(pt[0]+1j*pt[1])%(2*np.pi)
    if ang >= np.pi*(2-0.2):
        return True
    if ang <= 1.*(np.pi/2 - ctheta/2):
        return True
    return False