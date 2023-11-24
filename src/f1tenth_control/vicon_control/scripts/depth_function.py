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
        return "Pixel out of range for depth calculation"
    ray = np.linalg.inv(camera_matrix) @ np.array([x, y, 1])
    cos_theta = np.dot(ray, normal) / (np.linalg.norm(ray) * np.linalg.norm(normal))
    depth = camera_height / cos_theta
    return depth

def get_depth_for_pixel(x, y, params_file):
    params = read_params(params_file)
    return calculate_depth_at_pixel(x, y, params['Image Height'], params['Image Width'], params['Ground Plane Normal'], params['Camera Height'], params['Camera Matrix'])
