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
    # points_3d is a dictionary with keys = segments

    # set what segments to plot and in what order order   
    #plot_order = ['Hind COXA', 'Hind TROCHANTER', 'Mid COXA', 'Mid TROCHANTER', 'Mid TIBIA', 
    #              'Front COXA', 'Front TROCHANTER', 'Front FEMUR', 'Front TIBIA']
    plot_order = ['Mid TIBIA', 'Front COXA', 'Front TROCHANTER', 'Front FEMUR', 'Front TIBIA']
    plot_order = [s.replace(' ', ' leg ') for s in plot_order]
    plot_order = [f"{side} {seg}" for seg in plot_order for side in ['Left', 'Right']]

    # get colors for each segment
    colors_dict_0 = _set_colors_by_segment()
    colors_dict = {seg: colors_dict_0[f"{seg.split()[1]} {seg.split()[3]}"] for seg in plot_order}  # e.g. "Hind COXA" from "Left Hind leg COXA"

    # initial plot
    fig, ax = plt.subplots(figsize=(18, 8))
    plt.subplots_adjust(right=0.75, left=0)
    ax.imshow(image, cmap='gray')
    ax.set_title("Align 3D legs to min projection image.  Close window or click 'Done' to finish")


    # create a persistant reference for the outline plots for each segment
    outline_plots = {segment: ax.plot([], [], color=colors_dict[segment], lw=1, zorder=5)[0] for segment in plot_order} 

    # create a timer (to avoid lag while dragging sliders, we will only update the outlines after a short delay once the sliders stop moving)
    timer = fig.canvas.new_timer(interval=50)  

    def compute_and_draw_outlines():
        """ only runs when the timer finishes """

        # get current transformation values from sliders
        s = s_scale.val
        rx, ry, rz = s_rx.val, s_ry.val, s_rz.val
        tx, ty = s_tx.val, s_ty.val

        # Transform points for each segment
        for segment in plot_order:
            segment_points = points_3d[segment]
            p_transformed = _rotate_points(segment_points * s, rx, ry, rz)
            x_proj = p_transformed[:, 0] + tx
            y_proj = p_transformed[:, 1] + ty
            points_2d = np.column_stack((x_proj, y_proj))

            outline = _alpha_shape(points_2d, alpha=0.04)
            outline_plots[segment].set_data(outline[:, 0], outline[:, 1])  # update the existing Line2D object with new outline coordinates

        fig.canvas.draw_idle()  # redraw the figure to show the updated hull

    # link the timer to the outline function
    timer.add_callback(compute_and_draw_outlines)



    # create a single-shot timer for peek-a-boo effect (hide outlines when button is pressed)
    peek_timer = fig.canvas.new_timer(interval=200)
    peek_timer.single_shot = True

    # set function to show outlines after peek_timer finishes
    def show_outlines():
        for plot in outline_plots.values():
            plot.set_visible(True)
        fig.canvas.draw_idle()

    # link the peek_timer to the show_outlines function
    peek_timer.add_callback(show_outlines)

        

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
    s_rx = Slider(ax_rx, 'Rot X', 0, 180, valinit=90, valstep=1, valfmt='%d°')
    s_ry = Slider(ax_ry, 'Rot Y', -90, 90, valinit=0, valstep=1, valfmt='%d°')
    s_rz = Slider(ax_rz, 'Rot Z', -270, 90, valinit=-90, valstep=1, valfmt='%d°')
    s_tx = Slider(ax_tx, 'Shift X', 0, image.shape[1], valinit=image.shape[1]//2, valstep=1, valfmt='%d')
    s_ty = Slider(ax_ty, 'Shift Y', 0, image.shape[0], valinit=image.shape[0]//2, valstep=1, valfmt='%d')

    def update(val):
        """ this runs continuoulsy as sliders are moved, but we will only compute the hull after a short delay to avoid lag """

        # stop the timer every time the sliders are moved
        timer.stop()

        # restart the stimer. If 5000ms pass without the sliders being moved again, the hull will be computed and drawn
        timer.start()

        fig.canvas.draw_idle()

    # Register update function
    for s in [s_scale, s_rx, s_ry, s_rz, s_tx, s_ty]:
        s.on_changed(update)



    # Reset Button
    button_height = 0.35
    first_button_x = 0.735
    reset_ax = plt.axes([first_button_x, button_height, 0.07, 0.04])
    btn_reset = Button(reset_ax, 'Reset', color='lightgray', hovercolor='gray')

    def reset(event):
        for s in [s_scale, s_rx, s_ry, s_rz, s_tx, s_ty]:
            s.reset()
    btn_reset.on_clicked(reset)

    # Peek Button
    peek_ax = plt.axes([first_button_x + 0.08, button_height, 0.07, 0.04])
    btn_peek = Button(peek_ax, 'Hide outlines', color='lightblue', hovercolor='dodgerblue')

    def peek(event):
        # hide outlines
        for plot in outline_plots.values():
            plot.set_visible(False)
        fig.canvas.draw_idle()

        # start timer to show outlines again after a short delay
        peek_timer.start()
    
    btn_peek.on_clicked(peek)

    # Done button
    done_ax = plt.axes([first_button_x + 0.08*2, button_height, 0.07, 0.04])
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
