import matplotlib.pyplot as plt
import pandas as pd
import nilmtk as nilmtk
import numpy as np

def draw_stackedplot(df_list, title):
    
    stacked_power = np.zeros(len(df_list[0]))
    for i, df in enumerate(df_list):
        print("i: ", i)
        if i == 12:
            plt.plot(df.index, df.iloc[:, 0].values, label=f"SiteMeter", linewidth=3.5)
            # plt.plot(df.index, df.iloc[:, 0].values, label=f"SiteMeter", linewidth=3.5, alpha=0.5)
            continue
        stacked_power += df.iloc[:, 0].values
        plt.plot(df.index, stacked_power, label=f"Appliance {i+1}", linewidth=3.5)
    plt.xlabel("Time [yyyy-mm-dd]", fontsize=33, labelpad=15)
    plt.ylabel("Power [W]", fontsize=33, labelpad=15)
    plt.title(title, fontsize=50)
    plt.grid()
    plt.legend(fontsize=20)
    plt.xlim([df.index.min(), df.index.max()])
    plt.tick_params(axis="both", labelsize=23)
    plt.show()

def draw_plot(input_data, title="", metergroup=None, lim="", top_k=None):
    # print()
    # print("metergroup: ", metergroup)
    fig, ax = plt.subplots()
    

    if not isinstance(input_data, list):
        input_data = [input_data]
    
    meter_counters = 1


    for i, item in enumerate(input_data):
        # print("i: ", i)
        length = len(item)
        # print("head: ", item.head())
        # print("tail: ", item.tail())
        if isinstance(item, nilmtk.elecmeter.ElecMeter):
            item = item.power_series_all_data(ac_type="active").to_frame()
        if isinstance(item, pd.DataFrame):
            if top_k != None:
                if i == 0:
                    label = "FHMM"
                    zorder = 3
                elif i == 1:
                    label = "CO"
                    zorder = 2
                elif i == 2:
                    label = "GT"
                    zorder = 1
                for appliance, i in zip(item.columns, top_k):
                    label = metergroup.meters[i-1].label()
                    print("label: ", label)
                    ax.plot(item.index, item[appliance], label=label, linewidth=3, alpha=0.7)
                    ax.set_xlim([item.index.min(), item.index.max()])
                # if lim == "FHMM" or lim == "CO":
                    # ax.set_ylim([-145, 3133])
            else:
                col = 1 if i == 0 else col
                for appliance in item.columns:
                    if appliance == ('power', 'active'):
                        # label = f"ElecMeter {meter_counters}"
                        if metergroup != None:
                            label = metergroup.meters[i].label()
                        else:
                            # label=f"ElecMeter {meter_counters}"
                            print(i)
                            label = "Sum of Submeters"
                            if i == 1:
                                label = "Site Meter"
                        meter_counters += 1
                        if i == 12:
                            label = "SiteMeter"
                    # if lim == "FHMM" or lim == "CO":
                    else:
                        label = appliance
                    ax.plot(item.index, item[appliance], label=label, linewidth=3, alpha=0.8)
                    ax.set_xlim([item.index.min(), item.index.max()])
                    if lim == "FHMM" or lim == "CO":
                        ax.set_ylim([-145, 3133])
        
        elif isinstance(item, nilmtk.metergroup.MeterGroup):
            print("metergroup detected")
            for meter in item:
                df = meter.power_series_all_data(ac_type='active').to_frame()
                ax.plot(df.index, df)
            # item.plot(ax=ax)


    manager = plt.get_current_fig_manager()
    manager.window.state('zoomed')


    lines = ax.get_lines()
    # ax.set_title(title)   
    legend_picker(ax, lines)
    import matplotlib.dates as mdates
    from matplotlib.ticker import MultipleLocator 
    plt.xlabel("$t$ in dd HH:MM", fontsize=30, labelpad=15)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%d %H:00'))
    print(length)
    if 0 < length <= 60*60*24:
        ax.xaxis.set_major_locator(MultipleLocator(base=1/6))   # hä wieso ist mit base stunde gemeint
        print("base 1/6")
    elif 60*60*24 < length <= 60*60*24*3:
        ax.xaxis.set_major_locator(MultipleLocator(base=1/3))
        print("base 1/3")
    elif 60*60*24*3 < length <= 60*60*24*9:
        ax.xaxis.set_major_locator(MultipleLocator(base=1))
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%d.%m'))
        plt.xlabel("$t$ in DD.MM", fontsize=30, labelpad=15)
        print("base 1")
    elif 60*60*24*9 < length <= 60*60*24*30:
        ax.xaxis.set_major_locator(MultipleLocator(base=3))
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%d.%m'))
        plt.xlabel("$t$ in DD.MM", fontsize=30, labelpad=15)
        print("base 3")
    else:
        ax.xaxis.set_major_locator(MultipleLocator(base=5))
    plt.ylabel("$P$ in W", fontsize=30, labelpad=15)
    plt.tick_params(axis="both", labelsize=23)
    if title != "":
        plt.title(title, fontsize=40)
    ax.grid(True)
    plt.show()

def legend_picker(ax, lines, invisible=False):

    if invisible:
        for line in lines:
            line.set_visible(False)

    # leg = ax.legend(loc='upper left', bbox_to_anchor=(1, 1), fancybox=True, shadow=True, fontsize=15)
    leg = ax.legend(fontsize=23)
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


'''
def draw_plot(input_data, title="Title"):

    fig, ax = plt.subplots()

    if not isinstance(input_data, list):
        input_data = [input_data]

    for i, item in enumerate(input_data):
        # print("head: ", item.head())
        # print("tail: ", item.tail())
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
'''

def draw_plot2(input_data, title="", metergroup=None, lim="", top_k=None):
    # print()
    # print("metergroup: ", metergroup)
    fig, ax = plt.subplots()
    

    if not isinstance(input_data, list):
        input_data = [input_data]
    
    meter_counters = 1


    for i, item in enumerate(input_data):
        # print("i: ", i)
        length = len(item)
        # print("head: ", item.head())
        # print("tail: ", item.tail())
        if isinstance(item, nilmtk.elecmeter.ElecMeter):
            item = item.power_series_all_data(ac_type="active").to_frame()
        if isinstance(item, pd.DataFrame):
            if top_k != None:
                if i == 0:
                    label = "FHMM"
                    zorder = 3
                elif i == 1:
                    label = "CO"
                    zorder = 2
                elif i == 2:
                    label = "GT"
                    zorder = 1
                for appliance, i in zip(item.columns, top_k):
                    label = metergroup.meters[i-1].label()
                    print("label: ", label)
                    ax.plot(item.index, item[appliance], label=label, linewidth=3, alpha=0.7)
                    ax.set_xlim([item.index.min(), item.index.max()])
                # if lim == "FHMM" or lim == "CO":
                    # ax.set_ylim([-145, 3133])
            else:
                col = 1 if i == 0 else col
                for appliance in item.columns:
                    if appliance == ('power', 'active'):
                        # label = f"ElecMeter {meter_counters}"
                        if metergroup != None:
                            label = metergroup.meters[i].label()
                        else:
                            # label=f"ElecMeter {meter_counters}"
                            print(i)
                            label = "Sum of Submeters"
                            if i == 1:
                                label = "Site Meter"
                        meter_counters += 1
                        if i == 12:
                            label = "SiteMeter"
                    # if lim == "FHMM" or lim == "CO":
                    else:
                        if i == 0:
                            label = "FHMM"
                        elif i == 1:
                            label = "CO"
                        elif i == 2:    
                            label = "GT"
                        elif i == 3:
                            label = "FHMM-Ext"
                    ax.plot(item.index, item[appliance], label=label, linewidth=3, alpha=0.8)
                    ax.set_xlim([item.index.min(), item.index.max()])
                    if lim == "FHMM" or lim == "CO":
                        ax.set_ylim([-145, 3133])
        
        elif isinstance(item, nilmtk.metergroup.MeterGroup):
            print("metergroup detected")
            for meter in item:
                df = meter.power_series_all_data(ac_type='active').to_frame()
                ax.plot(df.index, df)
            # item.plot(ax=ax)


    manager = plt.get_current_fig_manager()
    manager.window.state('zoomed')


    lines = ax.get_lines()
    # ax.set_title(title)   
    legend_picker(ax, lines)
    import matplotlib.dates as mdates
    from matplotlib.ticker import MultipleLocator 
    plt.xlabel("$t$ in dd HH:MM", fontsize=30, labelpad=15)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%d %H:00'))
    print(length)
    if 0 < length <= 60*60*24:
        ax.xaxis.set_major_locator(MultipleLocator(base=1/6))   # hä wieso ist mit base stunde gemeint
        print("base 1/6")
    elif 60*60*24 < length <= 60*60*24*3:
        ax.xaxis.set_major_locator(MultipleLocator(base=1/3))
        print("base 1/3")
    elif 60*60*24*3 < length <= 60*60*24*9:
        ax.xaxis.set_major_locator(MultipleLocator(base=1))
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%d.%m'))
        plt.xlabel("$t$ in DD.MM", fontsize=30, labelpad=15)
        print("base 1")
    elif 60*60*24*9 < length <= 60*60*24*30:
        ax.xaxis.set_major_locator(MultipleLocator(base=3))
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%d.%m'))
        plt.xlabel("$t$ in DD.MM", fontsize=30, labelpad=15)
        print("base 3")
    else:
        ax.xaxis.set_major_locator(MultipleLocator(base=5))
    plt.ylabel("$P$ in W", fontsize=30, labelpad=15)
    plt.tick_params(axis="both", labelsize=23)
    if title != "":
        plt.title(title, fontsize=40)
    ax.grid(True)
    plt.show()



def draw_plot3(input_data, title="", metergroup=None, lim="", top_k=None):
    # print()
    # print("metergroup: ", metergroup)
    fig, ax = plt.subplots()
    

    if not isinstance(input_data, list):
        input_data = [input_data]
    
    meter_counters = 1


    for i, item in enumerate(input_data):
        # print("i: ", i)
        # print("head: ", item.head())
        # print("tail: ", item.tail())
        if isinstance(item, nilmtk.elecmeter.ElecMeter):
            item = item.power_series_all_data(ac_type="active").to_frame()
        if isinstance(item, pd.DataFrame):
            if top_k != None:
                if i == 0:
                    label = "FHMM"
                    zorder = 3
                elif i == 1:
                    label = "CO"
                    zorder = 2
                elif i == 2:
                    label = "GT"
                    zorder = 1
                for appliance, i in zip(item.columns, top_k):
                    label = metergroup.meters[i-1].label()
                    print("label: ", label)
                    ax.plot(item.index, item[appliance], label=label, linewidth=3, alpha=0.7)
                    ax.set_xlim([item.index.min(), item.index.max()])
                # if lim == "FHMM" or lim == "CO":
                    # ax.set_ylim([-145, 3133])
            else:
                col = 1 if i == 0 else col
                for appliance in item.columns:
                    if appliance == ('power', 'active'):
                        # label = f"ElecMeter {meter_counters}"
                        if metergroup != None:
                            label = metergroup.meters[i].label()
                        else:
                            # label=f"ElecMeter {meter_counters}"
                            print(i)
                            label = "Sum of Submeters"
                            if i == 1:
                                label = "Site Meter"
                        meter_counters += 1
                        if i == 12:
                            label = "SiteMeter"
                    # if lim == "FHMM" or lim == "CO":
                    else:
                        label = appliance
                    ax.plot(item.index, item[appliance], label=label, linewidth=3, alpha=0.8)
                    ax.set_xlim([item.index.min(), item.index.max()])
                    if lim == "FHMM" or lim == "CO":
                        ax.set_ylim([-145, 3133])
        
        elif isinstance(item, nilmtk.metergroup.MeterGroup):
            print("metergroup detected")
            for meter in item:
                df = meter.power_series_all_data(ac_type='active').to_frame()
                ax.plot(df.index, df)
            # item.plot(ax=ax)


    manager = plt.get_current_fig_manager()
    manager.window.state('zoomed')


    lines = ax.get_lines()
    # ax.set_title(title)   
    legend_picker(ax, lines)

    ax.grid(True)
    plt.show()