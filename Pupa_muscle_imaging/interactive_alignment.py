import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button
import matplotlib.colors as mcolors


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
        points_3d.append(coordinates_dict[seg])
        segment_name_short = f'{seg.split()[1]} {seg.split()[3]}'   # e.g. "Hind COXA" from "Left Hind leg COXA"
        # assign same color to all points in segment
        color = colors_dict[segment_name_short]
        points_color.append(np.tile(color, (coordinates_dict[seg].shape[0], 1)))  

    # concatenate all points and colors into single arrays
    points_3d = np.concatenate(points_3d, axis=0)
    points_color = np.concatenate(points_color, axis=0)
    
    return points_3d, points_color




def interactive_alignment(image, points_3d):

    # check if points_3d is a dict of segments or already a single array of points
    if isinstance(points_3d, dict):
        points_3d, colors = _get_points_and_colors(points_3d)
    else:
        colors = None

    # Subsample points if the cloud is too dense for smooth interaction
    if len(points_3d) > 5000:
        indices = np.random.choice(len(points_3d), 5000, replace=False)
        points_3d = points_3d[indices]
        if colors is not None:
            colors = colors[indices]


    fig, ax = plt.subplots(figsize=(10, 9))
    plt.subplots_adjust(bottom=0.35)
    
    # Initial Plot
    ax.imshow(image, cmap='gray')
    scatter = ax.scatter([], [], s=1, alpha=0.6, c=[] if colors is not None else 'red')
    ax.set_title("Align 3D Object (Red) to Image\nClose window or click 'Done' to finish")

    # Slider Axes
    ax_color = 'lightgoldenrodyellow'
    ax_scale = plt.axes([0.15, 0.25, 0.7, 0.025], facecolor=ax_color)
    ax_rx    = plt.axes([0.15, 0.21, 0.7, 0.025], facecolor=ax_color)
    ax_ry    = plt.axes([0.15, 0.17, 0.7, 0.025], facecolor=ax_color)
    ax_rz    = plt.axes([0.15, 0.13, 0.7, 0.025], facecolor=ax_color)
    ax_tx    = plt.axes([0.15, 0.09, 0.7, 0.025], facecolor=ax_color)
    ax_ty    = plt.axes([0.15, 0.05, 0.7, 0.025], facecolor=ax_color)

    # Sliders
    s_scale = Slider(ax_scale, 'Scale', 0.1, 5, valinit=0.6)
    s_rx = Slider(ax_rx, 'Rot X', -180, 180, valinit=90)
    s_ry = Slider(ax_ry, 'Rot Y', -180, 180, valinit=0)
    s_rz = Slider(ax_rz, 'Rot Z', -180, 180, valinit=-90)
    s_tx = Slider(ax_tx, 'Shift X', 0, image.shape[1], valinit=image.shape[1]//2)
    s_ty = Slider(ax_ty, 'Shift Y', 0, image.shape[0], valinit=image.shape[0]//2)

    def update(val):
        # Transform
        p_transformed = _rotate_points(points_3d * s_scale.val, s_rx.val, s_ry.val, s_rz.val)
        
        # Project and Shift
        x_proj = p_transformed[:, 0] + s_tx.val
        y_proj = p_transformed[:, 1] + s_ty.val
        
        scatter.set_offsets(np.column_stack((x_proj, y_proj)))

        # add colors if available
        if colors is not None:
            scatter.set_color(colors)

        fig.canvas.draw_idle()

    # Register update function
    for s in [s_scale, s_rx, s_ry, s_rz, s_tx, s_ty]:
        s.on_changed(update)

    # Reset Button
    reset_ax = plt.axes([0.65, 0.29, 0.1, 0.04])
    btn_reset = Button(reset_ax, 'Reset', color=ax_color, hovercolor='0.975')

    def reset(event):
        for s in [s_scale, s_rx, s_ry, s_rz, s_tx, s_ty]:
            s.reset()
    btn_reset.on_clicked(reset)

    # --- THE DONE BUTTON ---
    done_ax = plt.axes([0.77, 0.29, 0.1, 0.04])
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
        "rotation_deg": (s_rx.val, s_ry.val, s_rz.val),
        "shift_px": (s_tx.val, s_ty.val)
    }
