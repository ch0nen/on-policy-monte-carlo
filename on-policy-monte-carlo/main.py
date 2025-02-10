from matplotlib import pyplot as plt
import numpy as np
from utils.common import Maze, plot_action_values, plot_policy

def main():
    env = Maze()
    env.reset()

    frame = env.render(mode='rgb_array')
    plt.axis('off')
    plt.imshow(frame)
    # plt.show()

    print(f"Observation space shape: {env.observation_space.nvec}")
    print(f"Number of actions: {env.action_space.n}")

    action_values = np.zeros((5,5,4))

    plot_action_values(action_values)

    def policy(state, epsilon=0.2):
        if np.random.random() < epsilon:
            return np.random.choice(4)
        else:
            av = action_values[state]
            return np.random.choice(np.flatnonzero(av == av.max()))

    action = policy((0, 0), epsilon=0.5)
    print(f"Selected action: {action}")

    plot_policy(action_values, frame)

if __name__ == "__main__":
    main()