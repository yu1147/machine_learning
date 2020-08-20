import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib as mpl


def sinplot(flip=1):
    x = np.linspace(0, 14, 100)
    for i in range(1, 7):
        plt.plot(x, np.sin(x + i * .5) * (7 - i) * flip)

with sns.axes_style("darkgrid"):
    plt.subplot(211)
    sinplot()
plt.subplot(212)
sns.despine(left=True)
sinplot(-1)
#sns.set()
#sns.despine(offset=10)
#sinplot()
plt.show()




