import matplotlib
matplotlib.use('TkAgg')  # Ensure interactive backend
import matplotlib.pyplot as plt
import numpy as np

# Example data
time = np.linspace(0, 10, 500)
y1 = np.sin(time)
y2 = np.cos(time)
y3 = np.sin(2 * time)

# Create a plot
fig, ax = plt.subplots()

# Plot data with labels
line1, = ax.plot(time, y1, label="Sine Wave")
line2, = ax.plot(time, y2, label="Cosine Wave")
line3, = ax.plot(time, y3, label="Double Sine Wave")

# Add legend
legend = ax.legend(loc="upper right", fancybox=True, shadow=True)

# Make plots interactive
def on_pick(event):
    # Get the picked legend item (Line2D object)
    legend_line = event.artist

    # Find the corresponding plotted line (legend_line represents the legend handle)
    for line, legend_handle in zip([line1, line2, line3], legend.legendHandles):
        if legend_line is legend_handle:
            # Toggle the visibility of the plotted line
            visible = not line.get_visible()
            line.set_visible(visible)

            # Update the legend entry's transparency
            legend_handle.set_alpha(1.0 if visible else 0.2)

            # Redraw the canvas
            fig.canvas.draw()
            break

# Connect the event handler
fig.canvas.mpl_connect("pick_event", on_pick)

# Enable legend picking
for legend_handle in legend.legendHandles:
    legend_handle.set_picker(True)

# Show the plot
plt.show(block=True)