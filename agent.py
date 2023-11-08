import pygame
import numpy as np
import random
import math
import social_force as social_force


def distance(point1, point2):
    return math.sqrt((point2[0] - point1[0]) ** 2 + (point2[1] - point1[1]) ** 2)


def normalize(vec):
    dis = distance([0, 0], vec)
    if dis == 0:
        return vec
    vec = [vec[0] / dis, vec[1] / dis]
    return vec


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


# 构造一个Agent类
class Agent:
    def __init__(self, map_data, cell_size):
        self.map_data = map_data
        self.rows = len(map_data)
        self.cols = len(map_data[0])
        self.dir = [random.uniform(-1,  1),  random.uniform(-1,  1)]
        magnitude = math.sqrt(self.dir[0] ** 2 + self.dir[1] ** 2)
        self.dir = [self.dir[0] / magnitude, self.dir[1] / magnitude]
        self.cell_size = cell_size  # 单元格大小
        self.radius = cell_size // 2
        self.current_position = self.generate_random_position()
        self.target_position = self.generate_random_position()

        self.perception_range = 50  # 感知范围
        self.perception_samples = 10  # 隔多少度采样一个位置
        self.perception_vision_angle = 180  # 视角探测角度，范围（0，360]，以当前朝向顺时针逆时针各转一半均可探测到
        self.perception = [-1] * (self.perception_vision_angle // self.perception_samples + 1) # 例如180度sample是10度，则会有19个距离检测射线，包含0和180

    # 随机生成一个可移动的位置
    def generate_random_position(self):
        while True:
            row = random.randint(0, self.rows - 1)
            col = random.randint(0, self.cols - 1)
            if self.map_data[row][col] == 0:
                return [col * self.cell_size + self.cell_size // 2, row * self.cell_size + self.cell_size // 2]

    def check_pos_valid(self, x, y):
        if 0 <= x < self.rows and 0 <= y < self.cols:
            return True
        return False

    @staticmethod
    def calculate_side_ab(side_ac, side_bc, angle_b):
        if angle_b == 0:
            return side_bc - side_ac
        # 将角度转换为弧度
        angle_b = math.radians(angle_b)

        # 使用正弦定理计算边AB的长度
        angle_a = math.pi - math.asin(side_bc * math.sin(angle_b) / side_ac)
        angle_c = math.pi - angle_b - angle_a
        side_ab = side_bc * math.sin(angle_c) / math.sin(angle_a)

        return side_ab

    def calculate_perception(self, dis, r, angle):
        gap_angle = int(np.arctan(r / dis) / math.pi * 180)
        angle = int(angle)
        t_angle = calculate_angle([1, 0], self.dir)
        collision = False
        if dis < r + self.radius:
            collision = True
        for i in range(max(angle - gap_angle, 0),
                       min(angle + gap_angle, self.perception_vision_angle), 1):
            if i % self.perception_samples != 0:
                continue
            if collision:
                real_dis = 1
            else:
                real_dis = self.calculate_side_ab(r, dis, abs(i - angle))

            index = math.ceil((self.perception_vision_angle - i) / self.perception_samples)
            if self.perception[index] < 0 or real_dis < self.perception[index]:
                self.perception[index] = real_dis

    def perceive_agents(self, agents):
        # 遍历agents
        for agent in agents:
            dis = distance(agent.current_position, self.current_position)
            if dis == 0:
                continue
            real_dis = dis - agent.radius - self.radius
            v = [x - y for x, y in zip(agent.current_position, self.current_position)]
            angle = (int(calculate_angle(self.dir, v)) + int(self.perception_vision_angle / 2)) % 360
            # 如果距离小于感知范围且当前方向上的距离还未更新，则更新perception数组
            if real_dis <= self.perception_range and angle <= self.perception_vision_angle:
                self.calculate_perception(dis, agent.radius, angle)

    def perceive_map(self):
        base_angle = calculate_angle(self.dir, [1, 0])
        for i in range(0, len(self.perception)):
            angle = base_angle + i * self.perception_samples - self.perception_vision_angle / 2
            angle = (angle + 360) % 360
            cur_x = math.floor(self.current_position[0] / self.cell_size)
            cur_y = math.floor(self.current_position[1] / self.cell_size)

            # 沿伸方向
            if angle < 90 or angle > 270:
                x_dir = 1
            elif angle == 90 or angle == 270:
                x_dir = 0
            else:
                x_dir = -1
            if angle > 180:
                y_dir = 1
            elif angle == 180 or angle == 0:
                y_dir = 0
            else:
                y_dir = -1

            # 在x方向上沿伸
            tar_x = cur_x if x_dir == -1 else cur_x + 1
            angle = math.radians(angle)
            while x_dir != 0:
                if tar_x < 0 or tar_x >= self.cols:
                    break
                dis = math.fabs((tar_x * self.cell_size - self.current_position[0]) / math.cos(angle))
                if dis - self.radius > self.perception_range:
                    break
                tar_y = math.floor((dis * math.fabs(math.sin(angle)) * y_dir + self.current_position[1]) / self.cell_size)
                if tar_y < 0 or tar_y >= self.rows:
                    break
                if x_dir < 0:
                    tar_x = tar_x - 1
                if tar_x < 0:
                    break
                if self.map_data[tar_y][tar_x] == 1:
                    if dis < self.perception[i] or math.fabs(self.perception[i] + 1) < 1e-6:
                        self.perception[i] = dis
                    break
                if x_dir > 0:
                    tar_x = tar_x + 1
            # 在y方向上沿伸
            tar_y = cur_y if y_dir < 0 else cur_y + 1
            while y_dir != 0:
                if tar_y < 0 or tar_y >= self.rows:
                    break
                dis = math.fabs((tar_y * self.cell_size - self.current_position[1]) / math.sin(angle))
                if dis - self.radius > self.perception_range:
                    break
                tar_x = math.floor((dis * math.fabs(math.cos(angle)) * x_dir + self.current_position[0]) / self.cell_size)
                if tar_x < 0 or tar_x >= self.cols:
                    break
                if y_dir < 0:
                    tar_y = tar_y - 1
                if tar_y < 0:
                    break
                if self.map_data[tar_y][tar_x] == 1:
                    if dis < self.perception[i] or math.fabs(self.perception[i] + 1) < 1e-6:
                        self.perception[i] = dis
                    break
                if y_dir > 0:
                    tar_y = tar_y + 1
        return

    # 对环境产生感知(感知与移动可以不同步)
    def perceive(self, agents):
        self.perception = [-1] * (self.perception_vision_angle // self.perception_samples + 1)
        self.perceive_agents(agents)
        self.perceive_map()

    # 移动Agent(可以用不同的运动逻辑模块)
    def move(self, dt):
        speed = 80  # 秒速
        self.dir = social_force.socialforce(self.current_position, self.dir, self.perception, self.perception_samples, self.target_position)
        self.dir = normalize(self.dir)
        self.current_position = [x + y * dt * speed for x, y in zip(self.current_position, self.dir)]

        if self.check_goal():
            self.target_position = self.generate_random_position()

    # 检查是否达到目标位置
    def check_goal(self):
        dis = distance(self.current_position, self.target_position)
        if math.fabs(dis) < 1e-2:
            return True
        else:
            return False

    # 绘制Agent
    def draw(self, screen):
        # debug用，绘制perception，从agent按照perception各个角度与距离长度画一条线，如果为-1或0则不绘制
        for angle, dis in enumerate(self.perception):
            if dis != -1:
                t_angle = calculate_angle(self.dir, [1, 0])
                angle = angle * self.perception_samples + t_angle - int(self.perception_vision_angle / 2)
                adjusted_angle = (angle + 360) % 360  # 调整角度
                angle_rad = math.radians(adjusted_angle)  # 将角度转换为弧度
                end_point = [self.current_position[0] + dis * math.cos(angle_rad),
                             self.current_position[1] - dis * math.sin(angle_rad)]
                pygame.draw.line(screen, (0, 255, 0), self.current_position, end_point, 1)

        # debug用，绘制连接target的线条
        pygame.draw.line(screen, (0, 0, 255), self.current_position, self.target_position, 1)

        color = [255, 0, 0]
        pygame.draw.circle(screen, color, self.current_position, self.radius)

        # 绘制表示朝向的三角形
        angle = math.atan2(self.dir[1], self.dir[0])
        length = self.cell_size / 4
        point1 = (int(self.current_position[0] + length * math.cos(angle)), int(self.current_position[1] + length * math.sin(angle)))
        point2 = (int(self.current_position[0] + length * math.cos(angle - math.pi / 2)),
                  int(self.current_position[1] + length * math.sin(angle - math.pi / 2)))
        point3 = (int(self.current_position[0] + length * math.cos(angle + math.pi / 2)),
                  int(self.current_position[1] + length * math.sin(angle + math.pi / 2)))
        pygame.draw.polygon(screen, (255 - color[0], 255 - color[1], 255 - color[2]),
                            [point1, point2, point3])


