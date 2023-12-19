import matplotlib.colors as colors
import numpy as np

from matplotlib import cm


class MidpointRangeNormalize(colors.Normalize):

    def __init__(self, vmin=None, vmax=None, midrange=None, clip=False):
        self.midrange = midrange
        colors.Normalize.__init__(self, vmin, vmax, clip)

    def __call__(self, value, clip=None):
        x = [self.vmin, self.midrange[0], self.midrange[1], self.midrange[2], self.vmax]
        y = [0, 0.25, 0.5, 0.75, 1]
        return np.ma.masked_array(np.interp(value, x, y))


def get_colors(colormap='hsv', n=10):

    cmap = cm.get_cmap(colormap)

    # First color, always black
    rgb = [(0, 0, 0, 1)]

    for c in np.arange(1, n):
        rgb.append(cmap(c/n))

    rgb = np.asarray(rgb)

    # Randomize the other colors
    np.random.shuffle(rgb[1:])

    return rgb

