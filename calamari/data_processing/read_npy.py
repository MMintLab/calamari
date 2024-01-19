import copy
import os

import matplotlib.pyplot as plt
import numpy as np
import cv2
import imageio

from .utils import *
def check_and_make(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)

def process_demo(demo, folder):
    cnt_traj = []
    hull_traj = []
    front_rgb = []
    ee_history2d = []
    ee_historyx = []
    cnt_force = []
    hull_hist = {'idx': [], 'hull_topdown': [], 'hull_front': []}
    start_contact = False

    check_and_make(os.path.join(f'{folder}/rgb'))
    check_and_make(os.path.join(f'{folder}/contact_front'))
    check_and_make(os.path.join(f'{folder}/rgb'))


    for i_traj, traj in enumerate(demo):
        intrinsics = traj.misc['front_camera_intrinsics']
        extrinsics = traj.misc['front_camera_extrinsics']
        C = np.expand_dims(extrinsics[:3, 3], 0).T

        R = extrinsics[:3, :3]
        R_inv = R.T  # inverse of rot matrix is transpose
        R_inv_C = np.matmul(R_inv, C)
        extrinsics = np.concatenate((R_inv, -R_inv_C), -1)
        P = np.matmul(intrinsics, extrinsics)

        # reset.
        if i_traj == 0:
            hull_front_init = None
            rgb = traj.front_rgb
            cv2.imwrite(f'{folder}/rgb/rgb_{i_traj:03d}.png', rgb[:, :, [2, 1, 0]])
            continue

        cnt_force.append(traj.gripper_touch_forces)
        rgb = traj.front_rgb

        # TODO: Use contact points directly!
        if traj.contact_info != None :

            # Directly read contact points as contact patch convex hull
            if 'points_estimated' in traj.contact_info.keys():
                if len(traj.contact_info['points_estimated']) == 0 and len(traj.contact_info['points']) != 0:
                    # # TODO: figure out compatiblity with scooping task.
                    # continue
                    # 1) using coppeliasim's contact detection.
                    cntpxls, cntpxls_front = extract_contact_points(traj.contact_info['points'], P)
                    hull = cntpxls2img(cntpxls, imgsize=(rgb.shape[0], rgb.shape[1]))
                    hull_front = cntpxls2img(cntpxls_front, imgsize=(rgb.shape[0], rgb.shape[1]))

                    hull = cv2.dilate(hull, np.ones((3, 3), np.uint8), iterations=2)
                    hull_front = cv2.dilate(hull_front, np.ones((3, 3), np.uint8), iterations=2)


                elif len(traj.contact_info['points_estimated']) != 0:
                    # 1) using our own contact detection.
                    cntpxls, cntpxls_front = extract_contact_points(traj.contact_info['points_estimated'], P)
                    hull = cntpxls2img(cntpxls, imgsize=(rgb.shape[0], rgb.shape[1]))
                    hull_front = cntpxls2img(cntpxls_front, imgsize=(rgb.shape[0], rgb.shape[1]))
                    hull_front = cv2.dilate(hull_front, np.ones((3, 3), np.uint8), iterations=2)
                else:
                    hull = np.zeros_like(rgb[:, :, 0])
                    hull_front = np.zeros_like(rgb[:, :, 0])
                hull_traj.append(hull)
                # First font is valid, use it as init.
                if np.sum(hull_front / 255) > 10:
                    start_contact = True

            elif len(traj.contact_info['points']) == 0:
                    hull_front = np.zeros_like((rgb.shape[0], rgb.shape[1]))

            # Infer covexhull from sparse contact points (usually mesh vertices)
            else:
                pass

                # cnt_pts, cnt_front = extract_contact_points(traj.contact_info['points'], P)
                # try:
                #     hull = pixels_within_convexhull(cnt_pts.astype(int))
                #     hull_front = pixels_within_convexhull(cnt_front.astype(int), imgsize=rgb.shape)
                # # Skip tiny contact patches
                # except:
                #     print("null contact")
                #     continue

            # if len(hull_traj) > 0 and np.sum(np.abs(hull_hist['hull_front'][-1] / 255 - hull_front / 255)) < 50:
            #     rgb_list = os.listdir(f'{folder}/rgb')
            #     rgb_list.sort()
            #     os.remove(f'{folder}/rgb/{rgb_list[-1]}')
            #     os.remove(f'{folder}/contact_front/{rgb_list[-1]}'.replace('rgb', 'contact'))
            #
            #     # overwrite with the current one.
            #     cv2.imwrite(f'{folder}/rgb/rgb_{i_traj:03d}.png', rgb[:, :, [2, 1, 0]])
            #     cv2.imwrite(f'{folder}/contact_front/contact_{i_traj:03d}.png', hull_front)
            # if len(hull_traj) > 0 and np.sum(np.abs(hull_hist['hull_front'][-1] / 255 - hull_front / 255)) < 50:
            #     rgb_list = os.listdir(f'{folder}/rgb')
            #     rgb_list.sort()
            #     os.remove(f'{folder}/rgb/{rgb_list[-1]}')
            #     os.remove(f'{folder}/contact_front/{rgb_list[-1]}'.replace('rgb', 'contact'))
            #
            #     # overwrite with the current one.
            #     cv2.imwrite(f'{folder}/rgb/rgb_{i_traj:03d}.png', rgb[:, :, [2, 1, 0]])
            #     cv2.imwrite(f'{folder}/contact_front/contact_{i_traj:03d}.png', hull_front)
            try:
                # remove uninformative contact after first contact begins.
                print(np.sum(np.abs(hull_hist['hull_front'][-1] / 255 - hull_front / 255)))
                if len(hull_traj) > 0 and np.sum(np.abs(hull_hist['hull_front'][-1] / 255 - hull_front / 255)) < 200:

                    rgb_list = os.listdir(f'{folder}/rgb')
                    rgb_list.sort()
                    print(f'{folder}/rgb/{rgb_list[-1]}')
                    os.remove(f'{folder}/rgb/{rgb_list[-1]}')
                    os.remove(f'{folder}/contact_front/{rgb_list[-1]}'.replace('rgb', 'contact'))

                    # overwrite with the current one.
                    cv2.imwrite(f'{folder}/rgb/rgb_{i_traj:03d}.png', rgb[:, :, [2, 1, 0]])
                    cv2.imwrite(f'{folder}/contact_front/contact_{i_traj:03d}.png', hull_front)
            except:
                pass

            if start_contact:
                cv2.imwrite(f'{folder}/rgb/rgb_{i_traj:03d}.png', rgb[:, :, [2, 1, 0]])
                cv2.imwrite(f'{folder}/contact_front/contact_{i_traj:03d}.png', hull_front)

            ee_vel = traj.gripper_pose
            ee_history2d.append(np.linalg.norm(ee_vel[:2]))
            ee_historyx.append(ee_vel[0])

            hull_traj.append(hull)
            front_rgb.append(rgb)
            hull_hist['hull_topdown'].append(hull)
            hull_hist['hull_front'].append(hull_front)
            hull_hist['idx'].append(i_traj)
        else:
            # print("null contact")
            continue



## TODO: Consider moving these to config file.
DATA_CONFIG = {"sweep": {"folder":'sweep_dustpan', "taskname":'demo_sweep_dustpan'},
               "wipe": {"folder": 'datacollection/data_h_desk_wipe_0228', "taskname": 'demo_desk_wipe'},

               "scoop": {"folder": 'scoop', "taskname": 'demo_scoop'},
               "press": {"folder": 'button', "taskname": 'demo_push_buttons'}
               }
TASK = 'press' #'scoop'

if __name__ == '__main__':
    cnt_lst = []
    cnt_force = []
    hull_len_hist = []
    for j in range(100, 105):
        # folder = 'datacollection/data_h_desk_wipe__'#'sweep_dustpan'
        folder = DATA_CONFIG[TASK]["folder"]
        taskname = DATA_CONFIG[TASK]["taskname"]
        demo = np.load(f'out/{folder}/{taskname}_{j}.npy', allow_pickle=True)

        # Make directory.
        i = j
        if not os.path.exists(f'out/{folder}/processed/t_{i:03d}'):
            os.makedirs(os.path.join(f'out/{folder}/processed/t_{i:03d}/contact_key'))
            os.makedirs(os.path.join(f'out/{folder}/processed/t_{i:03d}/contact_key_front'))
            os.makedirs(os.path.join(f'out/{folder}/processed/t_{i:03d}/contact_front'))
            os.makedirs( os.path.join(f'out/{folder}/processed/t_{i:03d}/contact'))
            os.makedirs( os.path.join(f'out/{folder}/processed/t_{i:03d}/rgb'))

        cnt_traj = []
        hull_traj = []
        front_rgb = []
        ee_history2d = []
        ee_historyx = []
        hull_hist = {'idx' : [], 'hull_topdown' : [], 'hull_front' : []}

        # Start reading.
        for i_traj, traj in enumerate(demo):
            intrinsics = traj.misc['front_camera_intrinsics']
            extrinsics = traj.misc['front_camera_extrinsics']
            C = np.expand_dims(extrinsics[:3, 3], 0).T

            R = extrinsics[:3, :3]
            R_inv = R.T  # inverse of rot matrix is transpose
            R_inv_C = np.matmul(R_inv, C)
            extrinsics = np.concatenate((R_inv, -R_inv_C), -1)
            P = np.matmul(intrinsics, extrinsics)

            # reset.
            if i_traj == 0:
                hull_front_init = None
                rgb = traj.front_rgb
                cv2.imwrite(f'out/{folder}/processed/t_{i:03d}/rgb/rgb_{i:03d}_{i_traj:03d}.png', rgb[:, :, [2, 1, 0]])
                continue

            cnt_force.append(traj.gripper_touch_forces)
            rgb = traj.front_rgb

            # TODO: Use contact points directly!
            if traj.contact_info != None and len(traj.contact_info['points']) != 0:

                # Directly read contact points as contact patch convex hull
                if 'points_estimated' in traj.contact_info.keys():

                    if len(traj.contact_info['points_estimated']) == 0:
                        cntpxls, cntpxls_front = extract_contact_points(traj.contact_info['points'], P)
                        hull = cntpxls2img(cntpxls, imgsize=(rgb.shape[0], rgb.shape[1]))
                        hull_front = cntpxls2img(cntpxls_front, imgsize=(rgb.shape[0], rgb.shape[1]))

                        hull = cv2.dilate(hull, np.ones((3, 3), np.uint8), iterations=2)
                        hull_front = cv2.dilate(hull_front, np.ones((3, 3), np.uint8), iterations=2)

                        hull_traj.append(hull)
                    else:
                        # print(traj.contact_info['points'], traj.contact_info['points_estimated'])
                        cntpxls, cntpxls_front = extract_contact_points(traj.contact_info['points_estimated'], P)
                        hull = cntpxls2img(cntpxls, imgsize=(rgb.shape[0], rgb.shape[1]))
                        hull_front = cntpxls2img(cntpxls_front, imgsize=(rgb.shape[0], rgb.shape[1]))
                        hull_front = cv2.dilate(hull_front, np.ones((3, 3), np.uint8), iterations=2)



                # Infer covexhull from sparse contact points (usually mesh vertices)
                else:
                    cnt_pts, cnt_front = extract_contact_points(traj.contact_info['points'], P)
                    try:
                        hull = pixels_within_convexhull(cnt_pts.astype(int))
                        hull_front = pixels_within_convexhull(cnt_front.astype(int), imgsize=rgb.shape)
                    # Skip tiny contact patches
                    except:
                        print("null contact")
                        continue

                # If hull looks fine, save it.
                if  np.sum(hull/255) > 10:

                    # # If hull is identical to the previous one, skip it. (Happens when waypoint is for grasping)
                    if len(hull_traj) > 0:
                        print(folder, i_traj, np.sum( np.abs(hull_hist['hull_front'][-1] / 255 - hull_front/255)))
                    try:
                        # for sweeping especially.

                        if len(hull_traj) > 0 and np.sum( np.abs(hull_hist['hull_front'][-1] / 255 - hull_front/255)) < 50:

                            # list dir and remove the last one.
                            rgb_list = os.listdir(f'out/{folder}/processed/t_{i:03d}/rgb')
                            rgb_list.sort()
                            os.remove(f'out/{folder}/processed/t_{i:03d}/rgb/{rgb_list[-1]}')
                            os.remove(f'out/{folder}/processed/t_{i:03d}/contact/{rgb_list[-1]}'.replace('rgb', 'contact'))
                            os.remove(f'out/{folder}/processed/t_{i:03d}/contact_front/{rgb_list[-1]}'.replace('rgb', 'contact'))
                            print("remove",f'out/{folder}/processed/t_{i:03d}/rgb/{rgb_list[-1]}' )


                            # replace with the current one.
                            cv2.imwrite(f'out/{folder}/processed/t_{i:03d}/rgb/rgb_{i:03d}_{i_traj:03d}.png', rgb[:, :, [2, 1, 0]])
                            cv2.imwrite(f'out/{folder}/processed/t_{i:03d}/contact/contact_{i:03d}_{i_traj:03d}.png', hull)
                            cv2.imwrite(f'out/{folder}/processed/t_{i:03d}/contact_front/contact_{i:03d}_{i_traj:03d}.png', hull_front)
                    except:
                        pass
                            # print("save", f'out/{folder}/processed/t_{i:03d}/rgb/rgb_{i:03d}_{i_traj:03d}.png')
                    #
                    # # If initial hull is identical to the current one, skip it. (Happens when waypoint is for grasping)
                    # elif hull_front_init is not None and np.sum( np.abs(hull_front_init/255 - hull_front/255)) < 110:
                    #     print("tool has not moved")
                    #     pass
                    #
                    # # Save Result.
                    # else:
                    #     # Skip initial Contact.
                    #     if hull_front_init is None:
                    #         hull_front_init = copy.copy(hull_front)
                    #         continue
                    #     print( "cnt_hull diff", np.sum(hull_front_init/255 - hull_front/255))
                    cv2.imwrite(f'out/{folder}/processed/t_{i:03d}/rgb/rgb_{i:03d}_{i_traj:03d}.png', rgb[:, :, [2, 1, 0]])
                    cv2.imwrite(f'out/{folder}/processed/t_{i:03d}/contact/contact_{i:03d}_{i_traj:03d}.png', hull)
                    cv2.imwrite(f'out/{folder}/processed/t_{i:03d}/contact_front/contact_{i:03d}_{i_traj:03d}.png', hull_front)

                    ee_vel = traj.gripper_pose
                    ee_history2d.append(np.linalg.norm(ee_vel[:2]))
                    ee_historyx.append(ee_vel[0])

                    hull_traj.append(hull)
                    front_rgb.append(rgb)
                    hull_hist['hull_topdown'].append(hull)
                    hull_hist['hull_front'].append(hull_front)
                    hull_hist['idx'].append(i_traj)
                else:
                    print("null contact")
                    continue
            # l_union = union_img_binary(hull_hist['hull_front'])
            # img_temp = rgb * hull_union[:, :, np.newaxis] / 255.0


        # save a gif with imageio
        hull_len_hist.append(len(hull_traj))
        print("End of traj. length of hull_traj", len(hull_traj))
        try:
            imageio.mimsave(f'out/{folder}/processed/t_{i:03d}/cnt.gif', hull_traj, fps=10)
            imageio.mimsave(f'out/{folder}/processed/t_{i:03d}/front_rgb.gif', front_rgb, fps=10)
        except:
            print('no gif')
            print(hull_traj)
        print("saved gif")

    # import matplotlib.ticker as mticker
    # plt.hist(hull_len_hist, bins=np.amax(hull_len_hist) - np.amin(hull_len_hist))
    # plt.gca().xaxis.set_major_locator(mticker.MultipleLocator(1))
    # plt.gca().yaxis.set_major_locator(mticker.MultipleLocator(5))
    # plt.xlabel('# of contact goal generated')
    # plt.ylabel('# of demos')
    # plt.title('# goals :{:.3f} and std:{:.3f}'.format(np.mean(hull_len_hist), np.std(hull_len_hist)))
    # plt.show()

