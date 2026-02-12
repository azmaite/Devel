
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import os
import pickle

from scipy.spatial import Delaunay
import shapely.geometry as geom
import shapely.ops as ops



## Set colors and plot order for different body parts

# set plot order
leg_parts = ['COXA', 'TROCHANTER', 'FEMUR', 'TIBIA', 'TARSUS']
PLOT_ORDER = []
PLOT_ORDER.append('Pupa')
for leg in ['Hind', 'Mid', 'Front']:
    for part in leg_parts:
        PLOT_ORDER.append(f'Left {leg} leg {part}')
        PLOT_ORDER.append(f'Right {leg} leg {part}')
PLOT_ORDER.append('Left Wing')

# set colors for all parts
# for each leg, a different shade of red, green, blue
COLORS = {}
COLORS['Pupa'] = 'yellow'
COLORS['Wing'] = 'black'

def shade(base, factor):
    """Interpolate base color toward white (factor in [0,1])."""
    base = np.array(base)
    return tuple(base + (1 - base) * factor)

# how light each segment should be
factors = np.linspace(0.0, 0.6, len(leg_parts))  # darker → lighter

for i in range(len(leg_parts)):
    COLORS[f'Front_{leg_parts[i]}'] = shade(mcolors.to_rgb('firebrick'),   factors[i])
    COLORS[f'Mid_{leg_parts[i]}']   = shade(mcolors.to_rgb('yellowgreen'),  factors[i])
    COLORS[f'Hind_{leg_parts[i]}']  = shade(mcolors.to_rgb('dodgerblue'), factors[i])




## ACCESSORY FUNCTIONS ##

# function to rotate along the X, Y or Z axis
def rotate_around_xyz(mat, angle_deg, xyz='z'):

    angle_rad = np.deg2rad(angle_deg)
    
    c, s = np.cos(angle_rad), np.sin(angle_rad)

    if xyz == 'z':
        Rz = np.array([
            [ c, -s, 0],
            [ s,  c, 0],
            [ 0,  0, 1],
        ])
    elif xyz == 'y':
        Rz = np.array([
            [ c, 0, s],
            [ 0, 1, 0],
            [-s, 0, c],
        ])
    elif xyz == 'x':
        Rz = np.array([
            [1,  0,  0],
            [0,  c, -s],
            [0,  s,  c],
        ])
    else:
        raise ValueError("xyz must be 'x', 'y' or 'z'")

    return mat @ Rz.T


# function to get OUTLINE of each point-cloud (plus accesory function)
def alpha_shape(points, alpha):
    tri = Delaunay(points)
    edges = set()

    def add_edge(i, j):
        if (i, j) in edges or (j, i) in edges:
            edges.discard((j, i))
        else:
            edges.add((i, j))

    for ia, ib, ic in tri.simplices:
        pa, pb, pc = points[[ia, ib, ic]]
        a = np.linalg.norm(pb - pc)
        b = np.linalg.norm(pa - pc)
        c = np.linalg.norm(pa - pb)
        s = (a + b + c) / 2.0
        area = np.sqrt(max(s*(s-a)*(s-b)*(s-c), 0))
        if area == 0:
            continue
        circum_r = a*b*c / (4.0*area)
        if circum_r < 1.0 / alpha:
            add_edge(ia, ib)
            add_edge(ib, ic)
            add_edge(ic, ia)

    edge_points = [(points[i], points[j]) for i, j in edges]
    m = geom.MultiLineString(edge_points)
    return ops.polygonize(m)


def get_outlines(coordinates, plot_what, alpha=None):
    if alpha is None:
        alpha = 0.04  # smaller = tighter boundary

    outlines = {}

    # ensure plot_what is a list
    if isinstance(plot_what, str):
        plot_what = [plot_what]

    # select what to plot - ex: 'Pupa', 'leg', 'Wing', 'Mid'...
    names = PLOT_ORDER
    names_plot = [
        name for name in names
        if any(p in name for p in plot_what)
        ]
    if not names_plot:
        print(f"No names found with {plot_what}")
        return

    
    # get overal xyz lims from coordinates['Pupa']
    xyz_lims = np.array(coordinates['Pupa']).min(axis=0), np.array(coordinates['Pupa']).max(axis=0)

    xyz_combos = [(0,2), (1,2), (0,1)]

    for xyz_pair in xyz_combos:
        for name in names_plot:
            x = coordinates[name][:,xyz_pair[0]]
            y = coordinates[name][:,xyz_pair[1]]

            points_2d = np.column_stack((x, y))
            polys = list(alpha_shape(points_2d, alpha))
            if len(polys) > 1:
                print(f"Warning: multiple polygons found for {name} in plane {xyz_pair}")

            # save outline polygons as coordinate arrays - only for first xyz_pair
            if xyz_pair == xyz_combos[0]:
                outlines[name] = np.array(polys[0].exterior.coords)

    return outlines




### MAIN

def align_pupa_outlines(img, traced_coordinates, scale, x_shift, y_shift, x_rotation, y_rotation, z_rotation, plot_what='', old_ax=None, old_title=None, save_path=None):
    

    # load traced parts coordinates if not provided
    if traced_coordinates is None:

        main_path = '/mnt/labserver/data/MA/Development_project/'
        traces_folder = os.path.join(main_path, 'CT_PupaP15_traced')
        traces_file = os.path.join(traces_folder, 'pupa_traced_coordinates_rot.pkl')

        with open(traces_file, 'rb') as f:
            traced_coordinates = pickle.load(f)


    # center traced coordinates around mean of pupa
    traced_coordinates_aligned = {}
    pupa_coords = traced_coordinates['Pupa']
    pupa_mean = np.mean(pupa_coords, axis=0)
    for part in traced_coordinates.keys():
        coords = traced_coordinates[part]
        coords_centered = coords - pupa_mean
        traced_coordinates_aligned[part] = coords_centered

    # scale traced coordinates
    for part in traced_coordinates_aligned.keys():
        coords = traced_coordinates_aligned[part]
        coords_scaled = coords * scale
        traced_coordinates_aligned[part] = coords_scaled

    # rotate trace coordinates in x, y and z
    for part in traced_coordinates_aligned.keys():
        coords = traced_coordinates_aligned[part]
        coords_rotated = rotate_around_xyz(coords, x_rotation, 'x')
        coords_rotated = rotate_around_xyz(coords_rotated, y_rotation, 'y')
        coords_rotated = rotate_around_xyz(coords_rotated, z_rotation, 'z')
        traced_coordinates_aligned[part] = coords_rotated

    # default centering around middle of image
    img_center = np.array(img.shape) / 2
    for part in traced_coordinates_aligned.keys():
        coords = traced_coordinates_aligned[part]
        coords[:,0] += img_center[1]  
        coords[:,1] += 0 
        coords[:,2] += img_center[0]           
        traced_coordinates_aligned[part] = coords

    # shift trace coordinates in x and y
    for part in traced_coordinates_aligned.keys():
        coords = traced_coordinates_aligned[part]
        coords[:,0] += x_shift
        coords[:,2] += y_shift
        traced_coordinates_aligned[part] = coords

    # get outlines
    outlines = get_outlines(traced_coordinates_aligned, plot_what=plot_what, alpha=0.04)


    ## PLOT!
    fig = plt.figure(figsize=(10,10))

    # plot previous image if provided - if not, just plot the image with no lines
    ax = plt.subplot(1,2,1)
    plt.imshow(img, cmap='gray')

    if old_ax is not None:
        for line in old_ax.get_lines():
            ax.plot(line.get_xdata(), line.get_ydata(), color=line.get_color(), linewidth=line.get_linewidth(), label=line.get_label())

        if old_title is not None:
            plt.title(old_title)

    # plot average image
    new_ax = plt.subplot(1,2,2)
    plt.imshow(img, cmap='gray')

    # plot outlines
    for name in outlines.keys():
        outline = outlines[name]
    
        # find the correct color
        color = 'black'  # default
        for key in COLORS.keys():
            # split by '_' to avoid partial matches
            if '_' in key:
                key_list = key.split('_')
            else:
                key_list = [key]
            # check if all parts of the key are in the name
            if all(part in name for part in key_list):
                color = COLORS[key]
                break

        plt.plot(outline[:,0], outline[:,1], color=color, linewidth=0.5, label=name)
    
    # title
    new_title = f's={scale}, x,y_shi=({x_shift},{y_shift}), x,y,z_rot=({x_rotation},{y_rotation},{z_rotation})'
    plt.title(new_title)

    # save figure if save_path provided
    if save_path is not None:
        plt.savefig(save_path, dpi=300)

    plt.show()
    return new_ax, new_title



### EXAMPLE USAGE:
# from align_pupa_outlines import align_pupa_outlines
# scale = 0.8
# x_shift, y_shift = -125, 174
# x_rotation, y_rotation, z_rotation = 0, 185, -20
# plot_what = ['Pupa', 'Front', 'Wing']
# if 'old_ax' not in globals():
#     old_ax = None
#     old_title = None
# old_ax, old_title = align_pupa_outlines(image, coordinates=None, scale, x_shift, y_shift, x_rotation, y_rotation, z_rotation, plot_what, old_ax, old_title)
