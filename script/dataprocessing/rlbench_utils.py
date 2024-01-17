import numpy as np


# copy and paste from rlbench utils
DEFAULT_RGB_SCALE_FACTOR = 256000.0
DEFAULT_GRAY_SCALE_FACTOR = {
    np.uint8: 100.0,
    np.uint16: 1000.0,
    np.int32: DEFAULT_RGB_SCALE_FACTOR,
}
DEPTH_SCALE = 2**24 - 1


def image_to_float_array(image, scale_factor=None):
    """Recovers the depth values from an image.

    Reverses the depth to image conversion performed by FloatArrayToRgbImage or
    FloatArrayToGrayImage.

    The image is treated as an array of fixed point depth values.  Each
    value is converted to float and scaled by the inverse of the factor
    that was used to generate the Image object from depth values.  If
    scale_factor is specified, it should be the same value that was
    specified in the original conversion.

    The result of this function should be equal to the original input
    within the precision of the conversion.

    Args:
      image: Depth image output of FloatArrayTo[Format]Image.
      scale_factor: Fixed point scale factor.

    Returns:
      A 2D floating point numpy array representing a depth image.

    """
    image_array = np.array(image)
    image_dtype = image_array.dtype
    image_shape = image_array.shape

    channels = image_shape[2] if len(image_shape) > 2 else 1
    assert 2 <= len(image_shape) <= 3
    if channels == 3:
        # RGB image needs to be converted to 24 bit integer.
        float_array = np.sum(image_array * [65536, 256, 1], axis=2)
        if scale_factor is None:
            scale_factor = DEFAULT_RGB_SCALE_FACTOR
    else:
        if scale_factor is None:
            scale_factor = DEFAULT_GRAY_SCALE_FACTOR[image_dtype.type]
        float_array = image_array.astype(np.float32)
    scaled_array = float_array / scale_factor
    return scaled_array


def _pixel_to_world_coords(pixel_coords, cam_proj_mat_inv):
    h, w = pixel_coords.shape[:2]
    pixel_coords = np.concatenate([pixel_coords, np.ones((h, w, 1))], -1)
    world_coords = _transform(pixel_coords, cam_proj_mat_inv)
    world_coords_homo = np.concatenate([world_coords, np.ones((h, w, 1))], axis=-1)
    return world_coords_homo


def _create_uniform_pixel_coords_image(resolution: np.ndarray):
    pixel_x_coords = np.reshape(
        np.tile(np.arange(resolution[1]), [resolution[0]]),
        (resolution[0], resolution[1], 1),
    ).astype(np.float32)
    pixel_y_coords = np.reshape(
        np.tile(np.arange(resolution[0]), [resolution[1]]),
        (resolution[1], resolution[0], 1),
    ).astype(np.float32)
    pixel_y_coords = np.transpose(pixel_y_coords, (1, 0, 2))
    uniform_pixel_coords = np.concatenate(
        (pixel_x_coords, pixel_y_coords, np.ones_like(pixel_x_coords)), -1
    )
    return uniform_pixel_coords


def pointcloud_from_depth_and_camera_params(
    depth: np.ndarray, extrinsics: np.ndarray, intrinsics: np.ndarray
) -> np.ndarray:
    """Converts depth (in meters) to point cloud in word frame.
    :return: A numpy array of size (width, height, 3)
    """
    upc = _create_uniform_pixel_coords_image(depth.shape)
    pc = upc * np.expand_dims(depth, -1)
    C = np.expand_dims(extrinsics[:3, 3], 0).T
    R = extrinsics[:3, :3]
    R_inv = R.T  # inverse of rot matrix is transpose
    R_inv_C = np.matmul(R_inv, C)
    extrinsics = np.concatenate((R_inv, -R_inv_C), -1)
    cam_proj_mat = np.matmul(intrinsics, extrinsics)
    cam_proj_mat_homo = np.concatenate([cam_proj_mat, [np.array([0, 0, 0, 1])]])
    cam_proj_mat_inv = np.linalg.inv(cam_proj_mat_homo)[0:3]
    world_coords_homo = np.expand_dims(_pixel_to_world_coords(pc, cam_proj_mat_inv), 0)
    world_coords = world_coords_homo[..., :-1][0]
    return world_coords


def _transform(coords, trans):
    h, w = coords.shape[:2]
    coords = np.reshape(coords, (h * w, -1))
    coords = np.transpose(coords, (1, 0))
    transformed_coords_vector = np.matmul(trans, coords)
    transformed_coords_vector = np.transpose(transformed_coords_vector, (1, 0))
    return np.reshape(transformed_coords_vector, (h, w, -1))
