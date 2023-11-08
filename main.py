import pygame
import time
from Simulator import Simulator


# 主函数
def main():
    simulator = Simulator()
    running = True
    cur_time = -1
    dt = 0
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        if cur_time < 0:
            cur_time = time.time()
        else:
            dt = cur_time
            cur_time = time.time()
            dt = cur_time - dt
        simulator.step(dt)
        #time.sleep(0.05)

    pygame.quit()


if __name__ == "__main__":
    main()