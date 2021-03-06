import os
import matplotlib as mpl
if os.environ.get('DISPLAY','') == '':
    print('no display found. Using non-interactive Agg backend')
    mpl.use('Agg')
mpl.use('Agg')
from matplotlib import pyplot as plt
import numpy as np

# bit messy but contains all plot functions (for AE_CCEM, DDPG, NAF)
def plotFunction(agent_name, func_list, state, greedy_action, expl_action, x_min, x_max, resolution=1e3, display_title='', save_title='',
                 save_dir='', linewidth=2.0, ep_count=0, grid=True, show=False, equal_aspect=False):
    fig, ax = plt.subplots(2, sharex=True)
    # fig, ax = plt.subplots(figsize=(10, 5))

    x = np.linspace(x_min, x_max, resolution)
    y1 = []
    y2 = []

    max_point_x = x_min
    max_point_y = np.float('-inf')

    if agent_name == 'SoftActorCritic':
        func1, func2 = func_list[0], func_list[1]


        for point_x in x:
            point_y1 = np.squeeze(func1(point_x))  # reduce dimension
            point_y2 = np.squeeze(func2(point_x))

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


            ax[1].axvline(x=greedy_action[0], linewidth=1.5, color='grey')
            ax[1].axvline(x=expl_action[0], linewidth=1.5, color='blue')

        if display_title:
            fig.suptitle(display_title, fontsize=11, fontweight='bold')
            top_margin = 0.95

            mode_string = ""
            for i in range(len(greedy_action)):
                mode_string += "{:.2f}".format(np.squeeze(greedy_action[i]))

            ax[0].set_title("Action-values, argmax Q(S,A): {:.2f}".format(max_point_x))
            ax[1].set_title("Policy, greedy action: " + mode_string)
        else:
            top_margin = 1.0

    elif agent_name == 'ActorExpert_PICNN':  # Identical to ActorExpert except when passing func1(point_x)
        func1, func2 = func_list[0], func_list[1]

        modal_mean = [mean for mean in greedy_action[2]]

        old_greedy_action = greedy_action[1]
        greedy_action = greedy_action[0]

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

            for mean in modal_mean:
                ax[1].axvline(x=mean[0], linewidth=1.5, color='grey')

            ax[1].axvline(x=old_greedy_action[0], linewidth=1.5, color='pink')
            ax[1].axvline(x=greedy_action[0], linewidth=1.5, color='red')
            ax[1].axvline(x=expl_action[0], linewidth=1.5, color='blue')

        if display_title:
            fig.suptitle(display_title, fontsize=11, fontweight='bold')
            top_margin = 0.95

            mode_string = ""
            for i in range(len(greedy_action)):
                mode_string += "{:.2f}".format(np.squeeze(greedy_action[i]))

            ax[0].set_title("Action-values, argmax Q(S,A): {:.2f}".format(max_point_x))
            ax[1].set_title("Policy, greedy action: " + mode_string)
        else:
            top_margin = 1.0

    elif agent_name == 'ActorCritic':
        func1, func2 = func_list[0], func_list[1]

        modal_mean = [mean for mean in greedy_action[1]]

        old_greedy_action = greedy_action[0]
        greedy_action = greedy_action[0]

        for point_x in x:
            point_y1 = np.squeeze(func1(point_x))  # reduce dimension
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

            for mean in modal_mean:
                ax[1].axvline(x=mean[0], linewidth=1.5, color='grey')

            ax[1].axvline(x=old_greedy_action[0], linewidth=1.5, color='pink')
            ax[1].axvline(x=greedy_action[0], linewidth=1.5, color='red')
            ax[1].axvline(x=expl_action[0], linewidth=1.5, color='blue')

        if display_title:
            fig.suptitle(display_title, fontsize=11, fontweight='bold')
            top_margin = 0.95

            mode_string = ""
            for i in range(len(greedy_action)):
                mode_string += "{:.2f}".format(np.squeeze(greedy_action[i]))

            ax[0].set_title("Action-values, argmax Q(S,A): {:.2f}".format(max_point_x))
            ax[1].set_title("Policy, greedy action: " + mode_string)
        else:
            top_margin = 1.0

    elif agent_name == 'ActorCritic_unimodal':
        func1, func2 = func_list[0], func_list[1]

        modal_mean = greedy_action[1]

        old_greedy_action = greedy_action[0]
        greedy_action = greedy_action[0]

        for point_x in x:
            point_y1 = np.squeeze(func1(point_x))  # reduce dimension
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

            for mean in modal_mean:
                ax[1].axvline(x=mean, linewidth=1.5, color='grey')

            ax[1].axvline(x=old_greedy_action, linewidth=1.5, color='pink')
            ax[1].axvline(x=greedy_action, linewidth=1.5, color='red')
            ax[1].axvline(x=expl_action[0], linewidth=1.5, color='blue')

        if display_title:
            fig.suptitle(display_title, fontsize=11, fontweight='bold')
            top_margin = 0.95

            mode_string = ""
            for i in range(len(greedy_action)):
                mode_string += "{:.2f}".format(np.squeeze(greedy_action[i]))

            ax[0].set_title("Action-values, argmax Q(S,A): {:.2f}".format(max_point_x))
            ax[1].set_title("Policy, greedy action: " + mode_string)
        else:
            top_margin = 1.0

    elif agent_name == 'ActorExpert_PICNN':  # Identical to ActorExpert except when passing func1(point_x)
        func1, func2 = func_list[0], func_list[1]

        modal_mean = [mean for mean in greedy_action[2]]

        old_greedy_action = greedy_action[1]
        greedy_action = greedy_action[0]

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

            for mean in modal_mean:
                ax[1].axvline(x=mean[0], linewidth=1.5, color='grey')

            ax[1].axvline(x=old_greedy_action[0], linewidth=1.5, color='pink')
            ax[1].axvline(x=greedy_action[0], linewidth=1.5, color='red')
            ax[1].axvline(x=expl_action[0], linewidth=1.5, color='blue')

        if display_title:
            fig.suptitle(display_title, fontsize=11, fontweight='bold')
            top_margin = 0.95

            mode_string = ""
            for i in range(len(greedy_action)):
                mode_string += "{:.2f}".format(np.squeeze(greedy_action[i]))

            ax[0].set_title("Action-values, argmax Q(S,A): {:.2f}".format(max_point_x))
            ax[1].set_title("Policy, greedy action: " + mode_string)
        else:
            top_margin = 1.0

    elif agent_name == 'ActorExpert':
        func1, func2 = func_list[0], func_list[1]

        modal_mean = [mean for mean in greedy_action[2]]

        old_greedy_action = greedy_action[1]
        greedy_action = greedy_action[0]

        for point_x in x:
            point_y1 = np.squeeze(func1(point_x))  # reduce dimension
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

            for mean in modal_mean:
                ax[1].axvline(x=mean[0], linewidth=1.5, color='grey')

            ax[1].axvline(x=old_greedy_action[0], linewidth=1.5, color='pink')
            ax[1].axvline(x=greedy_action[0], linewidth=1.5, color='red')
            ax[1].axvline(x=expl_action[0], linewidth=1.5, color='blue')

        if display_title:
            fig.suptitle(display_title, fontsize=11, fontweight='bold')
            top_margin = 0.95

            mode_string = ""
            for i in range(len(greedy_action)):
                mode_string += "{:.2f}".format(np.squeeze(greedy_action[i]))

            ax[0].set_title("Action-values, argmax Q(S,A): {:.2f}".format(max_point_x))
            ax[1].set_title("Policy, greedy action: " + mode_string)
        else:
            top_margin = 1.0

    # elif agent_name == 'ActorExpert_Plus':
    #     func1, func2 = func_list[0], func_list[1]
    #
    #     old_greedy_action = greedy_action[1]
    #     greedy_action = greedy_action[0]
    #
    #     for point_x in x:
    #         point_y1 = np.squeeze(func1([point_x]))  # reduce dimension
    #         point_y2 = func2(point_x)
    #
    #         if point_y1 > max_point_y:
    #             max_point_x = point_x
    #             max_point_y = point_y1
    #
    #         y1.append(point_y1)
    #         y2.append(point_y2)
    #
    #     ax[0].plot(x, y1, linewidth=linewidth)
    #     ax[1].plot(x, y2, linewidth=linewidth)
    #
    #     if grid:
    #         ax[0].grid(True)
    #         ax[1].grid(True)
    #         ax[1].axhline(y=0, linewidth=1.5, color='darkslategrey')
    #         ax[1].axvline(x=old_greedy_action[0], linewidth=1.5, color='pink')
    #         ax[1].axvline(x=greedy_action[0], linewidth=1.5, color='red')
    #         ax[1].axvline(x=expl_action[0], linewidth=1.5, color='blue')
    #
    #     if display_title:
    #         display_title += ", argmax Q(S,A): {:.2f}".format(max_point_x)
    #         fig.suptitle(display_title, fontsize=11, fontweight='bold')
    #         top_margin = 0.95
    #
    #         mode_string = ""
    #         for i in range(len(greedy_action)):
    #             mode_string += "{:.2f}".format(np.squeeze(greedy_action[i])) + ", "
    #         ax[1].set_title("greedy actions: " + mode_string)
    #     else:
    #         top_margin = 1.0

    elif agent_name == 'DDPG' or agent_name == 'WireFitting' or agent_name == 'SoftQlearning':
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
            ax[1].axvline(x=greedy_action[0], linewidth=1.5, color='red')
            ax[1].axvline(x=expl_action[0], linewidth=1.5, color='blue')

        if display_title:

            fig.suptitle(display_title, fontsize=11, fontweight='bold')
            top_margin = 0.95
            ax[0].set_title("Action-values, argmax Q(S,A): {:.2f}".format(max_point_x))
            ax[1].set_title("Policy, greedy action: " + np.format_float_positional(greedy_action[0], precision=2))

        else:
            top_margin = 1.0

    elif agent_name == 'NAF':

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
            # ax[0].axhline(y=0, linewidth=1.5, color='darkslategrey')
            # ax[0].axvline(x=0, linewidth=1.5, color='darkslategrey')

            ax[1].grid(True)
            ax[1].axhline(y=0, linewidth=1.5, color='darkslategrey')
            ax[1].axvline(x=greedy_action[0], linewidth=1.5, color='red')
            ax[1].axvline(x=expl_action[0], linewidth=1.5, color='blue')

        if display_title:

            fig.suptitle(display_title, fontsize=11, fontweight='bold')
            top_margin = 0.95

            ax[0].set_title("Action-values, argmax Q(S,A): {:.2f}".format(max_point_x))
            ax[1].set_title("Policy, greedy action: " + np.format_float_positional(greedy_action[0], precision=2))

        else:
            top_margin = 1.0




        # OLD
        # func1 = func_list[0]
        # for point_x in x:
        #     point_y1 = np.squeeze(func1([point_x]))  # reduce dimension
        #
        #     if point_y1 > max_point_y:
        #         max_point_x = point_x
        #         max_point_y = point_y1
        #
        #     y1.append(point_y1)
        #
        # ax[0].plot(x, y1, linewidth=linewidth)
        #
        # if grid:
        #     ax[0].grid(True)
        #     # ax[0].axhline(y=0, linewidth=1.5, color='darkslategrey')
        #     # ax[0].axvline(x=0, linewidth=1.5, color='darkslategrey')
        #
        #     ax[1].grid(True)
        #     ax[1].axhline(y=0, linewidth=1.5, color='darkslategrey')
        #     ax[1].axvline(x=mean[0], linewidth=1.5, color='red')
        #
        # if display_title:
        #
        #     display_title += ", maxA: {:.3f}".format(max_point_x) + ", maxQ: {:.3f}".format(
        #         max_point_y) + "\n state: " + str(state)
        #     fig.suptitle(display_title, fontsize=11, fontweight='bold')
        #     top_margin = 0.95
        #
        #     ax[1].set_title("mean: " + str(mean[0]))
        #
        # else:
        #     top_margin = 1.0

    elif agent_name == 'PICNN':
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
            ax[1].axvline(x=greedy_action[0], linewidth=1.5, color='red')
            ax[1].axvline(x=expl_action[0], linewidth=1.5, color='blue')

        if display_title:

            fig.suptitle(display_title, fontsize=11, fontweight='bold')
            top_margin = 0.95

            ax[0].set_title("Action-values, argmax Q(S,A): {:.2f}".format(max_point_x))
            ax[1].set_title("Policy, greedy action: " + np.format_float_positional(greedy_action[0], precision=2))

        else:
            top_margin = 1.0

    elif agent_name == 'QT_OPT':

        # utils.plot_utils.plotFunction("QT_OPT", [func1, func2], state, greedy_action, chosen_action, self.action_min,
        #                               self.action_max,
        #                               display_title='ep: ' + str(
        #                                   self.train_ep_count) + ', steps: ' + str(
        #                                   self.train_global_steps),
        #                               save_title='steps_' + str(self.train_global_steps),
        #                               save_dir=self.writer.get_logdir(), ep_count=self.train_ep_count,
        #                               show=False)

        func1, func2 = func_list[0], func_list[1]

        greedy_action, modal_mean = greedy_action

        greedy_action = greedy_action[0]

        for point_x in x:
            point_y1 = np.squeeze(func1(point_x))  # reduce dimension
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

            for mean in modal_mean:
                ax[1].axvline(x=mean, linewidth=1.5, color='grey')

            ax[1].axvline(x=greedy_action, linewidth=1.5, color='red')
            ax[1].axvline(x=expl_action[0], linewidth=1.5, color='blue')

        if display_title:
            fig.suptitle(display_title, fontsize=11, fontweight='bold')
            top_margin = 0.95

            mode_string = ""
            for i in range(len(modal_mean)):
                mode_string += "{:.2f}".format(np.squeeze(modal_mean[i])) + ", "

            ax[0].set_title("Action-values, argmax Q(S,A): {:.2f}".format(max_point_x))
            ax[1].set_title("Policy, modal means: " + mode_string)
        else:
            top_margin = 1.0

    # set y range in the Q val plot
    # ax[0].set_ylim([-0.1, 1.6])

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
