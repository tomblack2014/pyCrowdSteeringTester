import pygame
from agent import Agent

class Simulator:
    def __init__(self):
        self.agents = []
        filename = "map.txt"  # 文件名
        self.map = self.read_map_from_file(filename)
        self.rows = len(self.map)
        self.cols = len(self.map[0])
        self.cell_size = 50  # 单元格大小
        self.width = self.cols * self.cell_size
        self.height = self.rows * self.cell_size
        self.create_agents()
        # 初始化Pygame
        pygame.init()
        # 创建画布
        self.screen = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption("人群仿真环境")

    def draw_map(self):
        self.screen.fill((255, 255, 255))
        for i in range(self.rows):
            for j in range(self.cols):
                color = (0, 0, 0) if self.map[i][j] == 1 else (255, 255, 255)
                pygame.draw.rect(self.screen, color, (j * self.cell_size, i * self.cell_size, self.cell_size, self.cell_size))

    # 读取map文件
    @staticmethod
    def read_map_from_file(filename):
        map_data = []
        with open(filename, 'r') as file:
            for line in file:
                row = []
                for char in line.strip():
                    if char.isdigit():
                        row.append(int(char))
                map_data.append(row)
        return map_data

    def create_agents(self, agent_num=5):
        self.agents = []
        for _ in range(agent_num):
            agent = Agent(self.map, self.cell_size)
            self.agents.append(agent)

    def step(self, dt):
        self.draw_map()
        # agents的具体行动
        for agent in self.agents:
            agent.move(dt)
            agent.perceive(self.agents)
        for agent in self.agents:
            agent.draw(self.screen)
        pygame.display.flip()
