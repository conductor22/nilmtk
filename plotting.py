import matplotlib.pyplot as plt

def draw_plot(meters, title="Title"):

    fig, ax = plt.subplots()

    if not isinstance(meters, list):
        meters = [meters]
    
    #meter_lines = []
    for meter_group in meters:
        meter_group.plot(ax=ax)
        
    lines = ax.get_lines()
    legend_picker(ax, lines, title)
    plt.show()

def legend_picker(ax, lines, title):

    ax.set_title(title)
    leg = ax.legend(fancybox=True, shadow=True)

    map_legend_to_ax = {}  # Will map legend lines to original lines.

    pickradius = 5  # Points (Pt). How close the click needs to be to trigger an event.

    for legend_line, ax_line in zip(leg.get_lines(), lines):
        legend_line.set_picker(pickradius)  # Enable picking on the legend line.
        map_legend_to_ax[legend_line] = ax_line


    def on_pick(event):
        # On the pick event, find the original line corresponding to the legend
        # proxy line, and toggle its visibility.
        legend_line = event.artist

        # Do nothing if the source of the event is not a legend line.
        if legend_line not in map_legend_to_ax:
            return

        ax_line = map_legend_to_ax[legend_line]
        visible = not ax_line.get_visible()
        ax_line.set_visible(visible)
        # Change the alpha on the line in the legend, so we can see what lines
        # have been toggled.
        legend_line.set_alpha(1.0 if visible else 0.2)
        ax.figure.canvas.draw()

    def on_close(event):
        ax.figure.canvas.mpl_disconnect(on_pick_connection)

    on_pick_connection = ax.figure.canvas.mpl_connect('pick_event', on_pick)
    ax.figure.canvas.mpl_connect('close_event', on_close)

    # Works even if the legend is draggable. This is independent from picking legend lines.
    leg.set_draggable(True)

    return map_legend_to_ax
