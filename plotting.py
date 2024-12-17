import matplotlib.pyplot as plt
import pandas as pd
import nilmtk as nilmtk

def draw_plot(input_data, title="Title"):

    fig, ax = plt.subplots()

    if not isinstance(input_data, list):
        input_data = [input_data]

    for i, item in enumerate(input_data):
    
        if isinstance(item, pd.DataFrame):
            for appliance in item.columns:
                ax.plot(item.index, item[appliance], label=appliance)

        # label geht für metergroup so nicht
        elif isinstance(item, nilmtk.elecmeter.ElecMeter):
            meter_name = f"Meter {item.instance()}"
            # meter_name = "Meter " + str(item.instance())
            item.plot(ax=ax, plot_kwargs={'label': meter_name}) 
            # for meter_group in input_data:
            #     meter_group.plot(ax=ax)
    
        elif isinstance(item, nilmtk.metergroup.MeterGroup):
            item.plot(ax=ax)


    lines = ax.get_lines()
    ax.set_title(title)
    legend_picker(ax, lines)
    plt.show()

def legend_picker(ax, lines):

    leg = ax.legend(fancybox=True, shadow=True)

    map_legend_to_ax = {}  # map legend lines to original lines

    pickradius = 5  # radius to trigger event

    for legend_line, ax_line in zip(leg.get_lines(), lines):
        legend_line.set_picker(pickradius)
        map_legend_to_ax[legend_line] = ax_line


    def on_pick(event):
        # find original line corresponding to picked line
        legend_line = event.artist

        # Do nothing if the source of the event is not a legend line.
        if legend_line not in map_legend_to_ax:
            return

        ax_line = map_legend_to_ax[legend_line]
        visible = not ax_line.get_visible()
        ax_line.set_visible(visible)
        # set alpha to indicate what has been toggled (off)
        legend_line.set_alpha(1.0 if visible else 0.2)
        ax.figure.canvas.draw()

    # sonst gehen die nächsten plots nicht
    def on_close(event):
        ax.figure.canvas.mpl_disconnect(on_pick_connection)

    on_pick_connection = ax.figure.canvas.mpl_connect('pick_event', on_pick)
    ax.figure.canvas.mpl_connect('close_event', on_close)

    # große legenden besser platzieren
    leg.set_draggable(True)

    return map_legend_to_ax
