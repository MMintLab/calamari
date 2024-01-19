import cv2
import numpy as np
import imageio

metric_thres = 10
front_rgbs = []
metrics = []
metrics_final = []
binary_metrics = []
goal_steps = []



# folder = 'out/March_result/wipe_200_op_2'
folder = 'out/april_result/wipe/three'
folder = 'out/sweep_hor_reb'
# folder = 'out/wipe_test2_calamari'
folder = 'out/peract/22-19-48'
folder = 'out/peract/2023-10-06/18-38-06'
folder = 'out/peract/2023-10-06/23-52-34'
folder = 'out/1013_wipe1'
# task = 'wipe_desk' #'wipe'
# task = 'sweep1' #'wipe'
task = 'wipe'
# task = 'cliport_sweep'
#
st = 125
N = 25
perf_goal_steps = np.ones((N, 21)) * -1
for i_ori in range(N):
    i = i_ori + st
    # demo = np.load(f'{folder}/demo_{task}_{0}.npy', allow_pickle=True)

    try:
        # demo = np.load(f'{folder}/demo_scoop_{i}.npy', allow_pickle=True)
        # demo = np.load(f'{folder}/demo_sweep_{i}.npy', allow_pickle=True)
        demo = np.load(f'{folder}/demo_{task}_{i}.npy', allow_pickle=True)

    except:

        print(f'{folder}/demo_{task}_{i}.npy not found')
        continue

    front_rgb = []
    metric = []
    for i_traj, traj in enumerate(demo):
        ## Make a gif using front cam
        try:
            rgb = traj.front_rgb
        except:
            rgb = traj['front_rgb']
            rgb = np.transpose(rgb, (1, 2, 0))
            # peract
        # blend with contact goal (magenta)


        # Try save RGB
        try:
            if traj.contact_goal is None:
                cnt_goal = np.zeros_like(rgb).astype(np.uint8)
            else:
                cnt_goal = traj.contact_goal.astype(np.uint8)
            cnt_goal = cv2.addWeighted(rgb, 1.0, cnt_goal, 0.5, 0)
            print( cnt_goal.shape)
            front_rgb.append(cnt_goal.astype(np.uint8))

        except:

            front_rgb.append(rgb.astype(np.uint8))

        try:
            ## Plot the number of particles
            metric.append(traj.sim_cost)

        except:
            continue

        ## Plot the number of particles
        try:            # if perf_goal_steps[i_ori, traj.goal_steps] < 0:
            perf_goal_steps[i_ori, traj.goal_steps:] = traj.sim_cost
        except:
            pass

    ## count success & fail.
    imageio.mimsave(f'{folder}/front_rgb_{i:02d}.gif', front_rgb, fps=10)

    # Try save final score
    try:
        metrics.append(metric)
        print(metric)
        metrics_final.append(metric[-1])
        binary_metrics.append(1 if metric[-1] < metric_thres + 1 else 0)

        goal_steps.append(demo[-1].goal_steps)
        perf_goal_steps[i_ori, traj.goal_steps:] = traj.sim_cost
    except:
        pass



print(metrics_final, np.mean(metrics_final))
# plot graph given n trajectories
import matplotlib.pyplot as plt
for i in range(len(metrics)):
    plt.plot(np.array(metrics[i]))
# put a legend at the end of the line


# plt.legend([f'run_{i}' for i in range(15)])
# plt.axhline(y=10, color='black', linestyle=':')
plt.xlabel('time step')
plt.ylabel('metric')
plt.show()

# Binary Success graph
plt.bar(range(2), (binary_metrics.count(0), binary_metrics.count(1)))
plt.xticks(range(2), ('success', 'fail'))
plt.show()
print(perf_goal_steps)
# success rate with the number of contact goals
import matplotlib.ticker as mticker
plt.plot(range(1, len(perf_goal_steps[0])), np.mean(perf_goal_steps, axis=0)[1:], 'o-')
# plt.bar(range(1, len(perf_goal_steps[0])), np.mean(100 - perf_goal_steps, axis=0)[1:])
plt.gca().xaxis.set_major_locator(mticker.MultipleLocator(1))
plt.gca().yaxis.set_major_locator(mticker.MultipleLocator(10))
plt.xlabel('# of contact goal generated')
plt.ylabel('Mean success rate (%)')
plt.show()



# Histogram of metrics
plt.bar(range(len(binary_metrics)), binary_metrics)
plt.show()

## KDE of metrics
import seaborn as sns
sns.kdeplot(metrics_final)
plt.xlim(0, 100)
plt.xlabel('metric')
plt.ylabel('density')
plt.title('KDE of metric with mean :{:.3f} and std:{:.3f}'.format(np.mean(metrics_final), np.std(metrics_final)))
plt.show()

# histogram of goal steps

plt.hist(goal_steps, bins= np.amax(goal_steps) - np.amin(goal_steps))
plt.gca().xaxis.set_major_locator(mticker.MultipleLocator(1))
plt.gca().yaxis.set_major_locator(mticker.MultipleLocator(1))

plt.xlabel('# of contact goal generated')
plt.ylabel('# of demos')
plt.title('# goals :{:.3f} and std:{:.3f}'.format(np.mean(goal_steps), np.std(goal_steps)))
plt.show()



