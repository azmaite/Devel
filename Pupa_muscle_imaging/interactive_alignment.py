import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button
import matplotlib.colors as mcolors
from scipy.spatial import Delaunay
import shapely.geometry as geom
import shapely.ops as ops


def _rotate_points(points, rx, ry, rz):
    # Rotation matrices (Degrees to Radians)
    cx, sx = np.cos(np.radians(rx)), np.sin(np.radians(rx))
    cy, sy = np.cos(np.radians(ry)), np.sin(np.radians(ry))
    cz, sz = np.cos(np.radians(rz)), np.sin(np.radians(rz))
    
    Rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
    Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    Rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
    
    return points @ Rx.T @ Ry.T @ Rz.T


def _shade(base, factor):
    """Interpolate base color toward white (factor in [0,1])."""
    base = np.array(base)
    return tuple(base + (1 - base) * factor)

def _set_colors_by_segment():
    """ Define a color scheme for each leg segment, with a gradient from darker to lighter for proximal to distal parts."""
    
    leg_parts = ['COXA', 'TROCHANTER', 'FEMUR', 'TIBIA', 'TARSUS']

    # how light each segment should be
    factors = np.linspace(0.0, 0.6, len(leg_parts))  # darker → lighter

    colors_dict = {}
    for i in range(len(leg_parts)):
        colors_dict[f'Front {leg_parts[i]}'] = _shade(mcolors.to_rgb('firebrick'),   factors[i])
        colors_dict[f'Mid {leg_parts[i]}']   = _shade(mcolors.to_rgb('yellowgreen'),  factors[i])
        colors_dict[f'Hind {leg_parts[i]}']  = _shade(mcolors.to_rgb('dodgerblue'), factors[i])

    return colors_dict

def _get_points_and_colors(coordinates_dict):
    """ Define which segments to plot and in which order,
    joining the corresponding coordinates into a single array for plotting 
    and assign colors based on segment type to each point. """

    # set plot order    
    plot_order = ['Hind COXA', 'Hind TROCHANTER', 'Mid COXA', 'Mid TROCHANTER', 'Mid TIBIA', 
                  'Front COXA', 'Front TROCHANTER', 'Front FEMUR', 'Front TIBIA']
    
    # edit segment names to match those in coordinates_dict
    plot_order = [s.replace(' ', ' leg ') for s in plot_order]

    # repeat for Left and Right legs
    plot_order = [f"{side} {seg}" for seg in plot_order for side in ['Left', 'Right']]

    # get color dictionary
    colors_dict = _set_colors_by_segment()

    # join coordinates in plot order, and assign colors to each point based on segment type
    points_3d = []
    points_color = []
    for seg in plot_order:
        segment_points = coordinates_dict[seg]

        # subsample to half the points randomly to reduce clutter (except COXA which has fewer points)
        if 'COXA' not in seg:
            indices = np.random.choice(segment_points.shape[0], size=segment_points.shape[0]//2, replace=False)
            segment_points = segment_points[indices]

        points_3d.append(segment_points)
        segment_name_short = f'{seg.split()[1]} {seg.split()[3]}'   # e.g. "Hind COXA" from "Left Hind leg COXA"
        # assign same color to all points in segment
        color = colors_dict[segment_name_short]
        points_color.append(np.tile(color, (segment_points.shape[0], 1)))  

    # concatenate all points and colors into single arrays
    points_3d = np.concatenate(points_3d, axis=0)
    points_color = np.concatenate(points_color, axis=0)
    
    return points_3d, points_color


def _alpha_shape(points, alpha=0.04):
    """ Obtain the alpha shape (concave hull) of a set of points."""
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

    polys = list(ops.polygonize(m))
    outline = np.array(polys[0].exterior.coords)  # get coordinates of the first polygon's exterior as the outline

    return outline




def interactive_alignment(image, points_3d):

    # check if points_3d is a dict of segments or already a single array of points
    if isinstance(points_3d, dict):
        points_3d, colors = _get_points_and_colors(points_3d)
    else:
        colors = None


    fig, ax = plt.subplots(figsize=(18, 8))
    plt.subplots_adjust(right=0.75, left=0)
    
    # Initial Plot
    ax.imshow(image, cmap='gray')
    scatter = ax.scatter([], [], s=1, alpha=0.6, c=[] if colors is not None else 'red')
    ax.set_title("Align 3D Object (Red) to Image\nClose window or click 'Done' to finish")


    # create a persistant reference for the outline plot
    outline_plot, = ax.plot([], [], 'k-', lw=2, zorder=5)

    # create a timer
    timer = fig.canvas.new_timer(interval=200)  

    def compute_and_draw_hull():
        """ only runs when the timer finishes """

        # get current transformation values from sliders
        s = s_scale.val
        rx, ry, rz = s_rx.val, s_ry.val, s_rz.val
        tx, ty = s_tx.val, s_ty.val

        # Transform points
        p_transformed = _rotate_points(points_3d * s, rx, ry, rz)
        x_proj = p_transformed[:, 0] + tx
        y_proj = p_transformed[:, 1] + ty
        points_2d = np.column_stack((x_proj, y_proj))

        # Compute alpha shape and plot
        outline = _alpha_shape(points_2d, alpha=0.04)
        outline_plot.set_data(outline[:, 0], outline[:, 1])  

        fig.canvas.draw_idle()  # redraw the figure to show the updated hull

    # link the timer to the hull function
    timer.add_callback(compute_and_draw_hull)
        

    # Slider Axes
    left, width, height = 0.74, 0.21, 0.025 # right is different for each
    ax_scale = plt.axes([left, 0.80, width, height])
    ax_rx    = plt.axes([left, 0.70, width, height])
    ax_ry    = plt.axes([left, 0.65, width, height])
    ax_rz    = plt.axes([left, 0.60, width, height])
    ax_tx    = plt.axes([left, 0.50, width, height])
    ax_ty    = plt.axes([left, 0.45, width, height])

    # Sliders
    s_scale = Slider(ax_scale, 'Scale', 0.3, 1.5, valinit=0.6, valstep=0.01, valfmt='%.2f')
    s_rx = Slider(ax_rx, 'Rot X', -90, 270, valinit=90, valstep=1, valfmt='%d°')
    s_ry = Slider(ax_ry, 'Rot Y', -180, 180, valinit=0, valstep=1, valfmt='%d°')
    s_rz = Slider(ax_rz, 'Rot Z', -270, 90, valinit=-90, valstep=1, valfmt='%d°')
    s_tx = Slider(ax_tx, 'Shift X', 0, image.shape[1], valinit=image.shape[1]//2, valstep=1, valfmt='%d')
    s_ty = Slider(ax_ty, 'Shift Y', 0, image.shape[0], valinit=image.shape[0]//2, valstep=1, valfmt='%d')

    def update(val):
        """ this runs continuoulsy as sliders are moved, but we will only compute the hull after a short delay to avoid lag """

        # stop the timer every time the sliders are moved
        timer.stop()

        # Transform
        p_transformed = _rotate_points(points_3d * s_scale.val, s_rx.val, s_ry.val, s_rz.val)
        
        # Project and Shift
        x_proj = p_transformed[:, 0] + s_tx.val
        y_proj = p_transformed[:, 1] + s_ty.val
        
        scatter.set_offsets(np.column_stack((x_proj, y_proj)))

        # add colors if available
        if colors is not None:
            scatter.set_color(colors)

        # restart the stimer. If 5000ms pass without the sliders being moved again, the hull will be computed and drawn
        timer.start()

        fig.canvas.draw_idle()

    # Register update function
    for s in [s_scale, s_rx, s_ry, s_rz, s_tx, s_ty]:
        s.on_changed(update)

    # Reset Button
    button_height = 0.35
    first_button_x = 0.775
    reset_ax = plt.axes([first_button_x, button_height, 0.07, 0.04])
    btn_reset = Button(reset_ax, 'Reset', color='lightgray', hovercolor='0.975')

    def reset(event):
        for s in [s_scale, s_rx, s_ry, s_rz, s_tx, s_ty]:
            s.reset()
    btn_reset.on_clicked(reset)

    # --- THE DONE BUTTON ---
    done_ax = plt.axes([first_button_x + 0.08, button_height, 0.07, 0.04])
    btn_done = Button(done_ax, 'Done', color='#90EE90', hovercolor='#32CD32')

    def done(event):
        plt.close(fig) # This closes the window and resumes script execution
    btn_done.on_clicked(done)
    # -----------------------

    update(None)
    plt.show() # Execution blocks here until the window is closed

    # Return the values captured at the moment the window was closed
    return {
        "scale": s_scale.val,
        "rotation_deg_xyz": (s_rx.val, s_ry.val, s_rz.val),
        "shift_px_xy": (s_tx.val, s_ty.val)
    }
