import numpy as np

def read_params(file_path):
    params = {}
    with open(file_path, 'r') as f:
        for line in f:
            key, value = line.strip().split(': ')
            if key in ['Image Height', 'Image Width']:
                params[key] = int(value)
            elif key in ['Camera Height']:
                # Parse the float value, even if it's within brackets
                params[key] = float(value.strip('[]'))
            elif key in ['Camera Matrix', 'Distortion Coefficients', 'Ground Plane Normal']:
                params[key] = np.array(eval(value))
    return params


def calculate_depth_at_pixel(x, y, height, width, normal, camera_height, camera_matrix):
    if y < height // 2 or y >= height or x < 0 or x >= width:
        return 0
    ray = np.linalg.inv(camera_matrix) @ np.array([x, y, 1])
    cos_theta = np.dot(ray, normal) / (np.linalg.norm(ray) * np.linalg.norm(normal))
    depth = camera_height / cos_theta
    return depth

def get_depth_for_pixel(x, y, params_file):
    params = read_params(params_file)
    return calculate_depth_at_pixel(x, y, params['Image Height'], params['Image Width'], params['Ground Plane Normal'], params['Camera Height'], params['Camera Matrix'])


def trans_Pix2Camera_v2( midpoint_x, midpoint_y, depth, params_file):
    # Read parameters from the file
    params = read_params(params_file)

    # Extract intrinsic matrix values
    fx = params['Camera Matrix'][0, 0]
    fy = params['Camera Matrix'][1, 1]
    cx = params['Camera Matrix'][0, 2]
    cy = params['Camera Matrix'][1, 2]

    # Calculate inverse of camera matrix
    K_inv = np.linalg.inv(np.array([[fx, 0, cx],
                                    [0, fy, cy],
                                    [0, 0, 1]]))

    # Pixel coordinates in homogeneous form
    pixel_coords_homogeneous = np.array([midpoint_x, midpoint_y, 1])

    # Camera local coordinates
    camera_coords = depth * np.dot(K_inv, pixel_coords_homogeneous)

    # The distance between camera and center of F1-10
    offset_y = 0.3

    return camera_coords[0], camera_coords[1] + offset_y, camera_coords[2]
