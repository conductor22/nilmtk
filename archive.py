fig, axs = plt.subplots(2, 2, figsize=(20,10))

dataset.buildings[1].elec.plot(ax=axs[0 ,0])
axs[0, 0].set_title("whole data")
axs[0, 0].legend().remove()



dataset.buildings[1].elec.plot(ax=axs[1 ,0])
axs[0, 1].set_title("windowed data")
axs[0, 1].legend().remove()

train.buildings[1].elec.plot(ax=axs[0 ,1])
axs[1, 0].set_title("train data")
axs[1, 0].legend().remove()
test.buildings[1].elec.plot(ax=axs[1 ,1])
axs[1, 1].set_title("test data")
axs[1, 1].legend().remove()

plt.subplots_adjust(hspace=0.6)
plt.tight_layout()
plt.show()