import numpy as np
import matplotlib.pyplot as plt
from nilm_metadata import get_appliance_types
import pprint

appliance_types = get_appliance_types()
appliance_type_list = list(appliance_types.keys())
pprint.pprint(appliance_type_list)

# Time axis
time = np.linspace(0, 24, 1000)  # Simulating 24 hours with 1000 points

power_permanent = np.full_like(time, 1500)  # Constant 100W consumption

power_on_off = np.where((time % 4) < 2, 1500, 0)  # On for 2h, off for 2h

fsm_states = [1000, 500, 1300, 400, 1500]  # Example states


state_durations = [0.5, 0.5, 2.2, 0.8, 2]
fsm_period = sum(state_durations)

state_start_times = np.cumsum([0] + state_durations)

def get_fsm_state(t):
    if 3 <= t <= 9 or 15 <= t <= 21:
        t_mod = (t - 3) % fsm_period  # Normalize to FSM cycle starting at 0
        state_index = np.searchsorted(state_start_times, t_mod, side="right") - 1
        return fsm_states[state_index]
    return 0

power_fsm = np.array([get_fsm_state(t) for t in time])

power_continuous = np.array([200 + 150 * np.sin(0.5 * t) + 400 * np.random.rand() if 0.2 <= t <= 4.8 else 0 for t in time])

# Create plots
fig, axes = plt.subplots(2, 2, figsize=(8, 10), sharex=False)

axes[0, 0].plot(time, power_permanent, label="Permanent Consumption", color='b', linewidth=2)
axes[0, 1].plot(time, power_on_off, label="On/Off Appliance", color='b', linewidth=2)
axes[1, 0].plot(time, power_fsm, label="FSM Appliance", color='b', linewidth=2)
axes[1, 1].plot(time, power_continuous, label="Continuous Variable Consumer", color='b', linewidth=2)
power_data = [power_permanent, power_on_off, power_fsm, power_continuous]

titles = ["Permanent Consumption", "On/Off", "Finite-State Machine (FSM)", "Continuously Variable"]
for ax, title, data in zip(axes.flatten(), titles, power_data):
    ax.set_title(title, fontsize=35)
    ax.set_ylabel("Power in W", fontsize=20)
    ax.set_xlabel("Time in s", fontsize=20)
    ax.grid(True)
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_xlim([0, 24])
    ax.set_ylim(bottom=0, top=2000)


plt.tight_layout(h_pad=5, w_pad=0.2)
plt.subplots_adjust(top=0.95, bottom=0.05, left=0.05, right=0.95, hspace=0.4, wspace=0.3)
plt.show()