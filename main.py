from matplotlib import pyplot as plt
from common import Maze

def main():
    env = Maze()
    env.reset()

    frame = env.render(mode='rgb_array')
    plt.axis('off')
    plt.imshow(frame)
    # plt.show()

    print(f"Observation space shape: {env.observation_space.nvec}")
    print(f"Number of actions: {env.action_space.n}")


if __name__ == "__main__":
    main()
