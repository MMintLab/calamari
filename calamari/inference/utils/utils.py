from scipy.spatial.transform import Rotation as R
import copy
import numpy as np
import open3d as o3d
import torch

def get_cntpts_from_mask(pcd: torch.tensor, mask: torch.tensor):
    # pcd: (N, 3)
    # mask: (N, )
    iidx, _ = torch.where(mask[:, :] == torch.ones_like(mask[:, :]))
    return pcd[iidx,:]

def get_handle_o3d(task):
    # Get next tool coords.
    vertices, indices, normals = task.grasp_target.get_mesh_data()
    # make new mesh from scrat
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(np.asarray(vertices))
    mesh.triangles = o3d.utility.Vector3iVector(np.asarray(indices))
    mesh.triangle_normals = o3d.utility.Vector3dVector(normals)

    pcd = mesh.sample_points_uniformly(number_of_points=1000)
    return np.array(pcd.points)

def get_tool_o3d(task, N = 1000):
    # Get next tool coords.
    try:
        vertices, indices, normals = task.tool.get_mesh_data()
    except:
        vertices, indices, normals = task.get_mesh_data()
    # make new mesh from scrat
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(np.asarray(vertices))
    mesh.triangles = o3d.utility.Vector3iVector(np.asarray(indices))
    mesh.triangle_normals = o3d.utility.Vector3dVector(normals)

    pcd = mesh.sample_points_uniformly(number_of_points=N)
    return np.array(pcd.points)


def transform_mesh(v, pose, device, from_euler = False, return_numpy = False):
    '''
    :param mesh: mesh object
    :param pose: tensor [x,y,z,qx,qy,qz,qw] format
    :return: transformed mesh in tensor
    '''

    if from_euler:
        pose_ = pose[:, 3:] # TODO: Check this. Not sure why 3:-1
        a_r = R.from_euler('XYZ', pose_)

        # a_r = R.from_euler('zyx', pose_)
    else:
        pose_ = pose[:, 3:7]
        a_r = R.from_quat(pose_ ) # x,y,z,w format

    a_T = a_r.as_matrix()

    B = a_T.shape[0]
    T = np.zeros((B, 4, 4))  # B x 4 x 4
    T[:, :3, :3] =  a_T
    T[:, :3, 3] = pose[:, :3]
    T[:, 3, 3] = 1
    #
    # # TODO: object transformation is not consistent
    # T_off1 = np.zeros((B, 4, 4))  # B x 4 x 4
    # T_off1[:, :3, :3] = R.from_euler('xyz', [-2.3874e+01, +1.2650e-02, -8.9991e+01]).as_matrix()
    # T_off1[:, 3, 3] = 1
    # T_off1 = torch.tensor( np.linalg.inv(T_off1)).to(device)
    #
    # # TODO: object transformation is not consistent
    # T_off2 = np.zeros((B, 4, 4))  # B x 4 x 4
    # T_off2[:, :3, :3] = R.from_euler('xyz', [-np.pi/2,0 , 0]).as_matrix()
    # T_off2[:, 3, 3] = 1
    # T_off2 = torch.tensor( np.linalg.inv(T_off2)).to(device)

    # Current State.
    V = np.ones((T.shape[0], v.shape[0], 4))  # B x N x 4
    V[:, :, :3] = v[np.newaxis, :, :].repeat(T.shape[0], axis=0)  # B x N x 4

    ## Resultant mesh pointcloud.
    T = torch.tensor(T).to(device)
    V = torch.tensor(V).to(device)
    V_ =  T @ V.transpose(1, 2)  # B x 4 x N
    V_ = V_.transpose(1, 2)  # B x N x 4
    tool_pcd = V_[:, :, :3]

    # visualize with o3d with frame
    mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
        size=0.6, origin=[0, 0, 0])
    pcds = [mesh_frame]

    # if tool_pcd.shape[0] > 1:
    #     for i in range(10):
    #         pcd = o3d.geometry.PointCloud()
    #
    #         pcd.points = o3d.utility.Vector3dVector(tool_pcd[i].cpu().numpy())
    #         pcds.append(pcd)
    #     # print("visualize", pose_)
    #     o3d.visualization.draw_geometries(pcds)


    if return_numpy:
        return tool_pcd.cpu().numpy()
    return tool_pcd
def img_normalize(img):
    if np.amax(img) >254:
        return img /255.
    else:
        return img

def IoU(img1:np.ndarray, img2:np.ndarray, device) -> np.ndarray:
    assert img1.shape ==img2.shape
    img1 = img_normalize(img1)
    img2 = img_normalize(img2)

    img1 = torch.tensor(img1).to(device)
    img2 = torch.tensor(img2).to(device)
    B, H, W = img1.shape

    # import cv2
    # cv2.imwrite("test_goal.png", img1[0] * 255)
    # cv2.imwrite("test.png", img2[0] * 255)

    iou = batch_intersect(img1, img2) / batch_union_torch(img1, img2)
    iou = iou.cpu().numpy()
    # intersection = np.intersect1d(img1, img2).reshape(B, -1)
    # union = np.union1d(img1, img2).reshape(B, -1)
    # iou = np.sum(intersection, axis=1) / np.sum(union, axis=1)
    return iou

def batch_intersect(img1, img2):
    '''
    img1 and img2 are batch of images with 0. and 1. values
    '''
    img1 = img1.reshape(img1.shape[0], -1)
    img2 = img2.reshape(img2.shape[0], -1)

    return torch.sum( img1 * img2, dim = -1)

def batch_union(img1, img2):
    '''
    img1 and img2 are batch of images with 0. and 1. values
    '''

    img1 = img1.reshape(img1.shape[0], -1)
    img2 = img2.reshape(img2.shape[0], -1)
    ones = np.ones_like(img1)

    union = img1 + img2
    union = np.where(union > 1 , np.ones_like(union), union)


    return np.sum( union, axis = -1)
def batch_union_torch(img1, img2):
    '''
    img1 and img2 are batch of images with 0. and 1. values
    '''

    img1 = img1.reshape(img1.shape[0], -1)
    img2 = img2.reshape(img2.shape[0], -1)
    ones = torch.ones_like(img1)

    union = img1 + img2
    union = torch.where(union > 1 , torch.ones_like(union), union)


    return torch.sum( union, dim = -1)
def get_rob_state_from_obs(obs):
    rob_state = obs.gripper_pose
    rob_state_r = R.from_quat(rob_state[3:7]).as_euler('XYZ')
    rob_state_ = [rob_state[0], rob_state[1], rob_state[2], rob_state_r[0], rob_state_r[1], rob_state_r[2]]
    return rob_state_
