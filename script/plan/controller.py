import logging
import time

import numpy as np
import torch
import cv2
import copy
import open3d as o3d
import calamari.inference.utils.utils as utils
from calamari.inference.utils.dynamics import get_next_pose
from pyrep.objects.shape import Shape
from pyrep.const import PrimitiveShape
from calamari.pytorch_mppi.pytorch_mppi import mppi
from calamari.cfg.config import MppiConfig
import matplotlib.pyplot as plt
from calamari.data_processing.utils import pixels_within_convexhull

# task_configs = {'WipeDesk' : {'task': WipeDesk, 'l': "Use the sponge to clean up the dirt.",
#                               'tool_name': 'sponge', 'target_name' : 'diningTable'}}
logger = logging.getLogger(__name__)
cfg = MppiConfig()

def records2obs(records, model_type):
    obs_list = records[- min([4, len(records)]):]
    # if model_type == 'temporal':
    #     obs_list = records[- min([4, len(records)]) :]
    #
    # else:
    #     obs_list = [records[-1]]
    return obs_list


class Controller(object):
    def __init__(self, env,  policy = None, dynamics = None, policy_mode = 'm3'):
        self.device = 'cuda:0'
        self.env = env
        self.task = env._prev_task
        self.policy = policy if policy is not None else policy
        self.dynamics = dynamics
        self.policy_mode = policy_mode
        self.ctrl = mppi.MPPI(self.dynamics.rigid_dynamics, self.running_cost,
                              cfg.nx, cfg.noise_sigma,
                              noise_mu = cfg.noise_mu,
                              num_samples=cfg.num_samples,
                              horizon=cfg.horizon,
                             lambda_=cfg.lambda_,
                              device=cfg.device,
                              u_min=torch.tensor(cfg.action_low, dtype=cfg.dtype, device=cfg.device),
                             u_max=torch.tensor(cfg.action_high, dtype=torch.double, device=cfg.device),
                              u_per_command = cfg.u_per_command)
        self.action_shape = env.action_shape

        # variables
        self.cnt_img_raw = []
        self.cnt_pts = None
        self.pcd_least_occ = None
        self.tool_pcd = utils.get_tool_o3d(self.task)
        self.contact = False
        self.obs_cur = None

    def get_tool_ocd(self):
        # Get next tool coords.
        vertices, indices, normals = self.task.tool.get_mesh_data()
        # make new mesh from scrat
        mesh = o3d.geometry.TriangleMesh()
        mesh.vertices = o3d.utility.Vector3dVector(np.asarray(vertices))
        mesh.triangles = o3d.utility.Vector3iVector(np.asarray(indices))
        mesh.triangle_normals = o3d.utility.Vector3dVector(normals)

        pcd = mesh.sample_points_uniformly(number_of_points=1000)
        return np.array(pcd.points)

    def draw_contact(self, next_pose):
        tool_pcd = utils.transform_mesh(self.tool_pcd, next_pose, self.device, from_euler=True)
        tool_pcd = tool_pcd.detach().cpu()



        front_pxls = self.policy.Config.pcd_to_camera(cnt_pts=tool_pcd)  # tool pixels

        # Pointcloud size = B x N x 3
        if len(tool_pcd.shape) == 2:
            pcd = tool_pcd.unsqueeze(0)  # [np.newaxis, :, :]
        assert len(tool_pcd.shape) == 3

        # detect contact using z offset

        # step 1- extract the bottom 1cm of the tool pcd
        bottom_thres = torch.amin(tool_pcd[..., 2], dim=1) + 0.01
        bottom_thres = bottom_thres[:, np.newaxis]

        # print(tool_pcd.shape, bottom_thres.shape)

        bottom_binary = torch.where(tool_pcd[..., 2] < bottom_thres.repeat((1, tool_pcd.shape[1])),
                                    torch.ones_like(tool_pcd[..., 2]),
                                    torch.zeros_like(tool_pcd[..., 2]))

        bottom_binary = bottom_binary.unsqueeze(2).repeat((1, 1, 2)).numpy()  # [..., np.newaxis] # B x N X 1
        # bottom_binary = bottom_binary.reshape(front_pxls.shape[0], front_pxls.shape[1])

        # step 2- extract Contact / Non-Contact / Penetration masks in 2D
        img_size = (bottom_binary.shape[0], 256, 256)

        _, iidx, _ = np.where(bottom_binary == np.ones_like(bottom_binary))
        # print(bottom_binary.shape)
        # breakpoint()
        self.visualize_next_step(tool_pcd[0, iidx,:])

    def running_cost(self, state, action):
        '''
        Input:
        :param N x 6 tool pose array: x,y,z,r,p,y
        :param N x 6 action array : x,y,z,r,p,y in radian
        :return: N x 7 tool pose array: x,y,z, qx,qy,qz,qw
        '''

        # Get next tool pose.
        # print("start controller")
        # state = state.detach().cpu()  # .numpy()
        # action = action.detach().cpu()
        # print(action)

        # y

        # try:
        #     print(np.amax(action, axis = 0))
        # except:
        #     print(torch.amax(action, dim = 0))
        # action = torch.zeros_like(action)
        # print(action)
        # breakpoint()

        # plot action distribution
        # plt.hist(action[:, -2], bins=100)
        # plt.show()

        # plt.hist(action[:, -1], bins=100)
        # plt.show()
        # breakpoint()


        # next_pose = get_next_pose(cur_pose=state, action=action, return_euler=True)  # x,y,z, w,x,y,z
        next_pose = copy.copy(state)
        B = state.shape[0]


        if self.contact:
            # print("action range", torch.min(action, dim = 0), torch.max(action, dim = 0))
            tool_pcd = utils.transform_mesh(self.tool_pcd, next_pose, self.device, from_euler = True)
            tool_pcd = tool_pcd.detach().cpu()

            # fig = plt.figure()
            # ax = fig.add_subplot(111, projection='3d')
            # ax.scatter3D(tool_pcd[0, :, 0],
            #              tool_pcd[0, :, 1],
            #              tool_pcd[0, :, 2], c='green', s=0.002)
            #
            # X = self.dynamics.init_obs.front_point_cloud[..., 0].reshape(-1)
            # Y = self.dynamics.init_obs.front_point_cloud[..., 1].reshape(-1)
            # Z = self.dynamics.init_obs.front_point_cloud[..., 2].reshape(-1)
            # ax.scatter3D(X, Y, Z, c='black', s=0.002)
            # plt.show()

            front_pxls = self.policy.Config.pcd_to_camera(cnt_pts=tool_pcd) # tool pixels

            # Pointcloud size = B x N x 3
            if len(tool_pcd.shape) == 2:
                pcd = tool_pcd.unsqueeze(0)  # [np.newaxis, :, :]
            assert len(tool_pcd.shape) == 3

            # detect contact using z offset

            # step 1- extract the bottom 1cm of the tool pcd
            bottom_thres = torch.amin(tool_pcd[..., 2], dim = 1) + 0.01
            bottom_thres = bottom_thres[:, np.newaxis]

            # print(tool_pcd.shape, bottom_thres.shape)

            bottom_binary = torch.where(tool_pcd[..., 2] < bottom_thres.repeat((1, tool_pcd.shape[1])),
                                        torch.ones_like(tool_pcd[..., 2]),
                                        torch.zeros_like(tool_pcd[..., 2]))


            bottom_binary = bottom_binary.unsqueeze(2).repeat((1, 1, 2)).numpy()  # [..., np.newaxis] # B x N X 1
            # bottom_binary = bottom_binary.reshape(front_pxls.shape[0], front_pxls.shape[1])


            # step 2- extract Contact / Non-Contact / Penetration masks in 2D
            img_size = (bottom_binary.shape[0], 256, 256)
            bottom_obj_2d_depth = np.zeros(img_size)  # .to(self.device)
            bottom_obj_2d_mask = np.zeros(img_size)  # .to(self.device)
            center_est = []
            front_pxls = front_pxls.cpu().numpy() # tool pcd's pixel indices [ii, jj]

            for b in range(bottom_binary.shape[0]):
                iidx, _ = np.where(bottom_binary[b, :, :] == np.ones_like(bottom_binary[b, :, :]))
                bottom_obj_2d_mask[b, front_pxls[b, iidx, 0], front_pxls[b, iidx, 1]] = np.ones_like(
                    front_pxls[b, iidx, 0])  # .to(self.device).float()
                bottom_obj_2d_depth[b, front_pxls[b, iidx, 0], front_pxls[b, iidx, 1]] = tool_pcd[b, iidx,2]  # .to(self.device).float()
                if b == 0:
                    pass
                    # print("expected pcd", len(iidx))

                # Center of contact patch
                center_est.append(tool_pcd[b, iidx, :].mean(axis=0).tolist())
            center_est = torch.tensor(center_est)

            # print("tool pcd", tool_pcd[b, iidx, 2])
            # print("front depth", self.dynamics.init_obs.front_point_cloud)
            # breakpoint()
            bottom_env_2d_depth = bottom_obj_2d_mask * self.dynamics.init_obs.front_point_cloud[np.newaxis, ..., 2]
            # print(bottom_env_2d_depth.shape, self.dynamics.init_obs.front_point_cloud.shape)


            # fig = plt.figure()
            # ax = fig.add_subplot(111, projection='3d')
            # ax.scatter3D(bottom_env_2d_depth[0][...,0],
            #              self.dynamics.init_obs.front_point_cloud,
            #              bottom_env_2d_depth[0].reshape(-1), c = 'black', s=0.002)
            # plt.show()


            # bottom_env_2d_depth = self.dynamics.init_obs.front_point_cloud[np.newaxis, ..., 2]


            # non_contact_mask = np.where(bottom_obj_2d_depth > bottom_env_2d_depth ,
            #                            np.ones_like(bottom_obj_2d_depth),
            #                            np.zeros_like(bottom_obj_2d_depth))
            # print( np.amin( bottom_obj_2d_depth - bottom_env_2d_depth))
            contact_mask = np.where(bottom_obj_2d_depth < bottom_env_2d_depth ,
                                       np.ones_like(bottom_obj_2d_depth),
                                       np.zeros_like(bottom_obj_2d_depth))
            contact_mask = bottom_obj_2d_mask * contact_mask

            # distance cost.
            cost_dist = torch.norm(torch.tensor(self.dynamics.contact_goal_center).to(self.device) - center_est.to(self.device), dim=-1)  # B x 1 x 1
            cost_dist[torch.isnan(cost_dist)] = 1.
            cost_dist = cost_dist.cpu().numpy()

            # non-penetration cost.
            thres_p = 0.008  # penetration threshold
            # print(np.amax(bottom_env_2d_depth - bottom_obj_2d_depth, axis=(-2,-1)),
            #       np.amin(bottom_env_2d_depth - bottom_obj_2d_depth, axis=(-2,-1) ))


            contact_loss = np.where ( np.amin( bottom_obj_2d_depth - bottom_env_2d_depth , axis=(-2,-1)) < 0.,
                        np.zeros_like(cost_dist), np.ones_like(cost_dist))

            non_pene_loss = np.where ( np.amax(bottom_env_2d_depth - bottom_obj_2d_depth, axis=(-2,-1)) > thres_p,
                        np.ones_like(cost_dist), np.zeros_like(cost_dist))
            # print("pene loss", np.amax(np.amax(bottom_env_2d_depth - bottom_obj_2d_depth, axis=(-2,-1))))


            # # # 3d scatter plot with red color
            # # X,Y = np.meshgrid(range(bottom_obj_2d_depth.shape[1]), range(bottom_obj_2d_depth.shape[2])[::-1])
            # X = self.dynamics.init_obs.front_point_cloud[ ..., 0].reshape(-1)
            # Y = self.dynamics.init_obs.front_point_cloud[ ..., 1].reshape(-1)
            # Z = bottom_env_2d_depth[0].reshape(-1)


            # fig = plt.figure()
            # ax = fig.add_subplot(111, projection='3d')
            # ax.scatter3D(X, Y, Z, c = 'black', s=0.002)
            # # ax.scatter3D(X, Y, bottom_env_2d_depth[0].reshape(-1), c='green', s=0.002)
            # ax.scatter3D(X, Y, bottom_obj_2d_depth[0, ...].reshape(-1), c='red', s=0.003)
            # # ax.scatter3D(X, Y, bottom_obj_2d_depth[0].reshape(-1), c='r', s=0.002)
            # # ax.scatter3D(X, Y, bottom_env_2d_depth[0].reshape(-1), c = 'black', s=0.002)
            # plt.show()



            # print(center_est, self.dynamics.contact_goal_center)
            # print("cost_dist", contact_pcd_padded.shape, center_est.shape, cost_dist.shape) # Size([100, 40, 3]) torch.Size([100, 3]) (100,)
            # print("time: cost_dist", time.time() - st)

            contact_mask_offset = np.zeros_like(contact_mask)
            # (self.cnt_img[:, :, 0]
            cnt_iidx, cnt_jidx = np.where(self.cnt_img[:, :, 0]//255 == 1)
            for bidx in range(contact_mask.shape[0]):
                iidx, jidx = np.where(contact_mask[bidx] == 1)
                # print(cnt_jidx, iidx)
                if len(iidx)> 0:
                    iidx = np.clip(iidx - int(np.mean(iidx)) + int(np.mean(cnt_iidx)), 0, contact_mask.shape[1] -1)
                    jidx = np.clip(jidx - int(np.mean(jidx)) + int(np.mean(cnt_jidx)), 0, contact_mask.shape[1] -1)
                    contact_mask_offset[bidx, iidx, jidx] = np.ones_like(iidx)

            # ## Cost 2: IoU
            st = time.time()

            # Add when distance is sufficiently small
            # if np.amin(cost_dist) > 0.03:
            #     iou = np.zeros_like(cost_dist)
            #     cost_dist = cost_dist*10
            # else:
            iou = utils.IoU(np.tile(self.cnt_img[:, :, 0][np.newaxis, :, :] // 255, (B, 1, 1)), contact_mask_offset,
                            self.device)
            # iou = utils.IoU( np.tile(self.cnt_img[:,:,0][np.newaxis,:,:]//255, (B,1,1)), contact_mask, self.device)
            iou = np.clip(np.ones_like(iou) - iou, 0, 1) * 0.3 #0.05

            # non_pene_loss= np.where( np.sum(non_pene_mask, axis = (1,2)) > 0, np.ones_like(cost_dist), np.zeros_like(cost_dist))

            # Cost
            st = time.time()
            norm = np.linalg.norm(action[:,:3], axis = -1)  #+ np.linalg.norm(action[:,3:], axis = -1) * 0.01
            cost =  cost_dist +  iou+ non_pene_loss + contact_loss + norm  #* 10 #+ pene_loss #+ contact_loss
            cost_idx = np.argsort(cost)[:5]
            # breakpoint()
            # 5 smallest cost values



            # cost =  cost_dist + non_pene_loss + contact_loss + 0.15 #* 10 #+ pene_loss #+ contact_loss

            # cost =  cost_dist + (np.ones_like(iou) - iou) * 0.2 #+ non_pene_loss #+ contact_loss* 10 #+ pene_loss #+ contact_loss
            cost = cost [ np.newaxis, :] # TODO : why the output shape 1x 1x N ?
            idx = np.argmin(cost)
            print("cost", np.min(cost), cost_dist[idx] , iou[idx], non_pene_loss[idx] , contact_loss[idx] , norm[idx] )


            cv2.imwrite(f"iou_test.png", (contact_mask_offset[idx]*0.2 + self.cnt_img[:,:,0][:,:]//255 * 0.8)*255)

            # pt = self.dynamics.contact_goal_center
            # print(pt)
            # spot = Shape.create(type=PrimitiveShape.CUBOID,
            #                     size=[.05, .05, .05],
            #                     mass=0, static=True, respondable=False,
            #                     renderable=False,
            #                     color=[1., 0., 0.])
            # spot.set_parent(self.task.get_base())
            # spot.set_position([pt[0], pt[1], pt[2]])
            #
            # # print("time: cost", time.time() - st)
            # pt = center_est[np.argmin(cost_dist)]
            # spot = Shape.create(type=PrimitiveShape.CUBOID,
            #                     size=[.05, .05, .05],
            #                     mass=0, static=True, respondable=False,
            #                     renderable=False,
            #                     color=[0., 0., 1.])
            # spot.set_parent(self.task.get_base())
            # spot.set_position([pt[0], pt[1], pt[2]])
            #
            # print( "max_pene", np.amax( bottom_env_2d_depth[idx] - bottom_obj_2d_depth[idx] ) )
            # union_ = self.cnt_img[:,:,0] + bottom_obj_2d_mask[idx,:,:]*255
            # plt.imsave('iou_debuc.png', union_/2)

            # plt.imsave('iou_debuc.png',np.concatenate([self.cnt_img[:,:,0] ,dyn_contact_img_final[idx,:,:]*255], axis = 0) )


            # iidx, jidx = np.where(dyn_contact_img[idx, :, :] == 1.)

            # dyn_target = utils.get_cntpts_from_mask( tool_pcd[idx], contact_binary_mask[idx])
            # st = time.time()
            # self.visualize_next_step(dyn_target) # check if tool transform is correct

            # print("dyn contact img", tool_pcd[idx].shape)
            # iidx, _ = np.where(bottom_binary[idx, :, :] == np.ones_like(bottom_binary[idx, :, :]))
            # self.visualize_next_step(tool_pcd[idx, iidx]) # check if tool transform is correct
            # print("time: visualize", time.time() - st)
        else:
            # Free space cost
            cost = torch.norm(
                torch.tensor(self.dynamics.contact_goal_center).to(self.device) - next_pose[:, :3].to(self.device),
                dim=-1) #+ torch.norm(action[:, :3].to(self.device), dim = -1) * 0.05 # B x 1 x 1

            cost = cost.cpu()
        return cost



    def generate_next_goal(self, obss_, l: str,  mode = 'hard', model_type = 'temporal') -> None:
        '''
        If self.cnt_img_raw is empty, draw contact goal via generate_contact_goal.
        if not empty, use use next seq
        '''

        # draw new contact goal
        if len(self.cnt_img_raw) == 0:
            obss = records2obs(obss_, model_type)
            self.generate_new_goal(obss, l,  mode)

        # use next seq
        self.cnt_img = self.cnt_img_raw[0]
        self.cnt_img_raw = self.cnt_img_raw[1:, ...]

        # 2D image -> 3D contact points.
        iidx, jidx = np.where(self.cnt_img[:, :, 0] == 255)
        self.cnt_pts = self.dynamics.init_obs.front_point_cloud[iidx, jidx, :]  # N x 3
        self.set_contact_goal()
        return True


    def generate_new_goal(self, obss, l: str,  mode = 'hard'):
        print("====================================")

        rgbs = []
        for idx, obs in enumerate(obss):
            if idx == len(obss) - 1:
                pcd = obs.front_point_cloud
                mask = np.ones_like(pcd[..., 0])
                mask = np.where(pcd[..., 0] < -0.15, np.zeros_like(mask), mask)
                mask = np.where(pcd[..., 2] > 1.05, np.zeros_like(mask), mask)
                mask = mask[..., np.newaxis]

                obs.front_rgb = obs.front_rgb * mask
                obs.front_rgb = obs.front_rgb.astype(np.uint8)

            # Estimate Contact Goal
            rgb = copy.copy(obs.front_rgb)  # W X H X 3
            rgbs.append(rgb)

        self.cnt_img_raw = self.policy.feedforward(rgbs, l)["contact"]

        # rgb_ = np.clip(rgb + self.cnt_img_raw[0], 0, 255)
        # cv2.imwrite(f"contact_goal_rgb.png", rgb_)



    def set_contact_goal(self):
        self.dynamics._set_contact_goal(self.cnt_img, self.cnt_pts)
        self.visualize_contact_goal()

    def is_valid_contact_goal(self, mode):
        '''
        :param mode: 'soft' or 'hard'
        '''
        if mode == 'soft':
            eps = 1
        elif mode == 'hard':
            eps = 30
        else:
            print("set valid goal mode to 'soft' or 'hard'")

        if len(self.cnt_pts) < eps :
            print("Current contact goal length:", len(self.cnt_pts), "resamplig")
            return False
        else:
            return True

    def visualize_contact_goal(self):
        # Reset Goal.
        if len(self.task.contact_goal) != 0:
                for g in self.task.contact_goal:
                    g.remove()

        # Visualize Goal.
        self.task.contact_goal = []
        for pt in self.cnt_pts:
            spot = Shape.create(type=PrimitiveShape.CUBOID,
                                size=[.01, .01, .01],
                                mass=0, static=True, respondable=False,
                                renderable=False,
                                color=[1., 0., 1.])
                                # color=[0., 0., 1.])

            spot.set_parent(self.task.get_base())
            spot.set_position([pt[0], pt[1], pt[2]])

            self.task.contact_goal.append(spot)

    def visualize_next_step(self, pts):
        # Reset Goal.
        try:
            if len(self.task.next_cnt) != 0:
                    for g in self.task.next_cnt:
                        g.remove()
        except:
            pass

        # Visualize Goal.
        # self.task.next_cnt = []

        # random sample 50 points
        idx = np.random.choice(len(pts), 10)
        pts = pts[idx, :]

        for pt in pts:
            spot = Shape.create(type=PrimitiveShape.CUBOID,
                                size=[.008, .008, .008],
                                mass=0, static=True,
                                respondable=False,
                                # renderable=False,
                                color=[1., 0., 0.])

            spot.set_parent(self.task.get_base())
            spot.set_position([pt[0], pt[1], np.amax([0.753])])

            # self.task.next_cnt.append(spot)