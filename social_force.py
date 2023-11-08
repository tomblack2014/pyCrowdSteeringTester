# 通过加速度、速度、位移三层积分模型基于社会力实现agent运动，
# 每一个step计算出社会力对应的加速度向量，dt为两个step间隔的时间，
# 给出角速度上限w_top和速度上限v_top
# 当当前self.dir与期望速度方向相同时方可开始运动，否则先将self.dir以不超过w_top的角速度调整至目标速度方向
import numpy as np
import math


def normalize(vec):
    dis = distance([0, 0], vec)
    if dis == 0:
        return vec
    vec = [vec[0] / dis, vec[1] / dis]
    return vec


def distance(point1, point2):
    return math.sqrt((point2[0] - point1[0]) ** 2 + (point2[1] - point1[1]) ** 2)


def rotation_matrix_from_axis_angle(angle):
    c = np.cos(angle)
    s = np.sin(angle)
    return np.array([
        [c, -s],
        [s, c]
    ])


def calculate_angle(vector1, vector2):
    # 计算向量的点积
    dot_product = vector1[0] * vector2[0] + vector1[1] * vector2[1]

    # 计算向量的模
    magnitude1 = math.sqrt(vector1[0]**2 + vector1[1]**2)
    magnitude2 = math.sqrt(vector2[0]**2 + vector2[1]**2)

    # 计算夹角的弧度值
    radians = math.acos(dot_product / (magnitude1 * magnitude2))

    # 判断顺时针还是逆时针方向
    cross_product = vector1[0] * vector2[1] - vector1[1] * vector2[0]
    if cross_product < 0:
        radians = 2 * math.pi - radians

    # 弧度转换为角度
    degrees = math.degrees(radians)

    # 确保结果在0-359之间
    return degrees % 360


def socialforce(pos, dir, perception, perception_samples, target):
    # 目标吸引力
    a_target = [y - x for x, y in zip(pos, target)]
    a_target = normalize(a_target)
    a_target = [x * 1 for x in a_target]

    collision_rate = 200
    a_obs = [0, 0]
    for angle, dis in enumerate(perception):
        if dis > 0:
            t_angle = calculate_angle([1, 0], dir)
            angle = angle * perception_samples + t_angle - int((len(perception) - 1) * perception_samples / 2)
            adjusted_angle = (angle + 360) % 360  # 调整角度
            angle_rad = math.radians(adjusted_angle)  # 将角度转换为弧度
            obs_force = [math.cos(angle_rad), math.sin(angle_rad)]
            a_obs = [x + y * collision_rate / dis / dis for x, y in zip(a_obs, obs_force)]
    ret = [x + y for x, y in zip(a_obs, a_target)]
    ret = normalize(ret)
    ret = [x * 50 for x in ret]
    normalize(ret)
    return ret
