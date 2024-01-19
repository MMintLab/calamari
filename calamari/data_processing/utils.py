import cv2
import numpy as np
import torch


def cntpxls2img(pts, imgsize=(150,150)):
    img = np.zeros((imgsize[0], imgsize[1], 3), np.uint8)
    pts = np.clip(pts, 0, imgsize[0]-1)
    img[pts[:, 1], pts[:, 0], :] = 255  # (y, x)
    return img


def pixels_within_convexhull(pts, imgsize=(150, 150)):
    # Set up dummy image
    img = np.zeros((imgsize[0], imgsize[1], 3), np.uint8)
    cv2.fillPoly(img, np.int32([pts]), (0, 0, 255))

    # Threshold image
    _, img_thr = cv2.threshold(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), 10, 255, cv2.THRESH_BINARY)

    # Find contours: OpenCV 4.x
    contours, _ = cv2.findContours(img_thr, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    # Find contours: OpenCV 3.x
    #_, contours, _ = cv2.findContours(img_thr, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    # Convex hulls from contours in list
    hulls = [cv2.convexHull(c) for c in contours]

    # Draw convex hull of first contour
    hull = np.zeros(img_thr.shape, np.uint8)
    if len(hulls[0]) == 1:
        print('No convex hull found', hulls)
        # All 1 will make IoU very low
        return np.ones((imgsize[0], imgsize[1]), np.uint8) * 255

    hull = cv2.drawContours(hull, np.array([np.squeeze(hulls[0])]), -1, 255, cv2.FILLED)

    # Access pixels inside convex hull; here add some green values to original image
    idx = hull == 255
    final = img.copy()
    final[idx, :] = final[idx, :] + (0, 128, 0)

    return hull

## TODO: Consider moving these to utils.py.
def union_img_binary(img_lst):
    img = np.zeros_like(img_lst[0])
    for idx, i in enumerate(img_lst):
        new = np.ones_like(img_lst[0])
        img  = np.where(i > 0.8, new, img)
    return img

def find_nearest(array, value):
        array = np.asarray(array)
        idx = (np.abs(array - value)).argmin()
        return array[idx]

def  extract_contact_points(points : np.ndarray, P: np.ndarray):
    # print(points.shape)
    cnt_ori = np.array(points)[:, :3]
    cnt_pts = np.concatenate([(cnt_ori[:,0].reshape(-1,1) + 0.4) * 110,
                             (cnt_ori[:,1].reshape(-1,1) + 0.6) * 110], axis = 1)

    # Map 3d points to front camera image
    cnt_pts_ = np.ones((4, cnt_ori.shape[0]))
    cnt_pts_[:3, :] = np.array(cnt_ori).T

    ## 3D to 2D
    mapped = P @ cnt_pts_
    cnt_front = mapped[:2, :] / mapped[2, :] # 2 x N
    cnt_front = np.round(cnt_front).astype(int)
    cnt_front  = cnt_front.T

    cnt_pts = np.round(cnt_pts).astype(int)
    return cnt_pts, cnt_front