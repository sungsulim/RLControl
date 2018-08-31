import os
import matplotlib as mpl
if os.environ.get('DISPLAY','') == '':
    print('no display found. Using non-interactive Agg backend')
    mpl.use('Agg')
mpl.use('Agg')
from matplotlib import pyplot as plt
import numpy as np

# bit messy but contains all plot functions (for AE_CCEM, DDPG, NAF)
def plotFunction(agent_name, func_list, state, mean, x_min, x_max, resolution=1e2, display_title='', save_title='',
                 save_dir='', linewidth=2.0, ep_count=0, grid=True, show=False, equal_aspect=False):
    fig, ax = plt.subplots(2, sharex=True)
    # fig, ax = plt.subplots(figsize=(10, 5))

    x = np.linspace(x_min, x_max, resolution)
    y1 = []
    y2 = []

    max_point_x = x_min
    max_point_y = np.float('-inf')

    if agent_name == 'AE_CCEM':
        func1, func2 = func_list[0], func_list[1]
        for point_x in x:
            point_y1 = np.squeeze(func1([point_x]))  # reduce dimension
            point_y2 = func2(point_x)

            if point_y1 > max_point_y:
                max_point_x = point_x
                max_point_y = point_y1

            y1.append(point_y1)
            y2.append(point_y2)

        ax[0].plot(x, y1, linewidth=linewidth)
        ax[1].plot(x, y2, linewidth=linewidth)

        if grid:
            ax[0].grid(True)
            ax[1].grid(True)
            ax[1].axhline(y=0, linewidth=1.5, color='darkslategrey')
            ax[1].axvline(x=0, linewidth=1.5, color='darkslategrey')

        if display_title:
            display_title += ", maxA: {:.3f}".format(max_point_x) + ", maxQ: {:.3f}".format(
                max_point_y) + "\n state: " + str(state)
            fig.suptitle(display_title, fontsize=11, fontweight='bold')
            top_margin = 0.95

            mode_string = ""
            for i in range(len(mean)):
                mode_string += "{:.3f}".format(np.squeeze(mean[i])) + ", "
            ax[1].set_title("modes: " + mode_string)
        else:
            top_margin = 1.0

    elif agent_name == 'DDPG':
        func1 = func_list[0]
        for point_x in x:
            point_y1 = np.squeeze(func1([point_x]))  # reduce dimension

            if point_y1 > max_point_y:
                max_point_x = point_x
                max_point_y = point_y1

            y1.append(point_y1)
        ax[0].plot(x, y1, linewidth=linewidth)

        if grid:
            ax[0].grid(True)
            ax[1].grid(True)
            ax[1].axhline(y=0, linewidth=1.5, color='darkslategrey')
            ax[1].axvline(x=mean[0], linewidth=1.5, color='red')

        if display_title:

            display_title += ", maxA: {:.3f}".format(max_point_x) + ", maxQ: {:.3f}".format(
                max_point_y) + "\n state: " + str(state)
            fig.suptitle(display_title, fontsize=11, fontweight='bold')
            top_margin = 0.95
            ax[1].set_title("mean: " + str(mean[0]))

        else:
            top_margin = 1.0

    elif agent_name == 'NAF':
        func1 = func_list[0]
        for point_x in x:
            point_y1 = np.squeeze(func1([point_x]))  # reduce dimension

            if point_y1 > max_point_y:
                max_point_x = point_x
                max_point_y = point_y1

            y1.append(point_y1)

        ax[0].plot(x, y1, linewidth=linewidth)

        if grid:
            ax[0].grid(True)
            # ax[0].axhline(y=0, linewidth=1.5, color='darkslategrey')
            # ax[0].axvline(x=0, linewidth=1.5, color='darkslategrey')

            ax[1].grid(True)
            ax[1].axhline(y=0, linewidth=1.5, color='darkslategrey')
            ax[1].axvline(x=mean[0], linewidth=1.5, color='red')

        if display_title:

            display_title += ", maxA: {:.3f}".format(max_point_x) + ", maxQ: {:.3f}".format(
                max_point_y) + "\n state: " + str(state)
            fig.suptitle(display_title, fontsize=11, fontweight='bold')
            top_margin = 0.95

            ax[1].set_title("mean: " + str(mean[0]))

        else:
            top_margin = 1.0


    # common
    if equal_aspect:
        ax.set_aspect('auto')

    if show:
        plt.show()

    else:
        # print(save_title)
        save_dir = save_dir + '/figures/'
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        plt.savefig(save_dir + save_title)
        plt.close()
