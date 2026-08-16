import random
import numpy as np
import matplotlib.pyplot as plt
from primsmaze import MazeEnv


class QLearningAgent:
    def __init__(
        self, 
        num_states, 
        num_actions=4, 
        alpha=0.2, 
        min_alpha=0.01,
        alpha_decay=0.999,
        gamma=0.999, 
        epsilon=1.0, 
        min_epsilon=0.01, 
        decay_rate=0.995
    ):
        self.num_states = num_states
        self.num_actions = num_actions
        
        self.alpha = alpha            
        self.min_alpha = min_alpha
        self.alpha_decay = alpha_decay

        self.gamma = gamma            
        self.epsilon = epsilon        
        self.min_epsilon = min_epsilon
        self.decay_rate = decay_rate

        self.q_table = np.zeros((num_states, num_actions))

    def choose_action(self, state, train=True):
        if train and random.uniform(0, 1) < self.epsilon:
            return random.randint(0, self.num_actions - 1)
        else:
            q_values = self.q_table[state]
            max_val = np.max(q_values)
            max_actions = np.where(q_values == max_val)[0]
            return random.choice(max_actions)

    def update(self, state, action, reward, next_state, done):
        if done:
            td_target = reward
        else:
            td_target = reward + self.gamma * np.max(self.q_table[next_state])
            
        td_error = td_target - self.q_table[state, action]
        self.q_table[state, action] += self.alpha * td_error

    def decay_hyperparameters(self):
        self.epsilon = max(self.min_epsilon, self.epsilon * self.decay_rate)
        self.alpha = max(self.min_alpha, self.alpha * self.alpha_decay)


def get_training_hyperparameters(grid_size):
    episodes = int(65 * grid_size)
    target_decay_step = int(episodes * 0.8)
    
    decay_rate = (0.01 / 1.0) ** (1 / target_decay_step)
    alpha_decay = (0.02 / 0.2) ** (1 / target_decay_step)
    
    
    return episodes, decay_rate, alpha_decay


def train():
    grid_size = 50
    env = MazeEnv(width=grid_size, height=grid_size)
    
    num_states = env.width * env.height
    episodes, decay_rate, alpha_decay = get_training_hyperparameters(grid_size)
    
    agent = QLearningAgent(
        num_states=num_states, 
        gamma=0.999, 
        decay_rate=decay_rate,
        alpha_decay=alpha_decay
    )
    

    max_steps_per_episode = env.width * env.height * 2
    rewards_history = []

    print(f"Training Q-Learning Agent on {env.height}x{env.width} Maze ({num_states} states) for {episodes} episodes...")

    for episode in range(episodes):
        state = env.reset()
        total_reward = 0

        for step in range(max_steps_per_episode):
            action = agent.choose_action(state, train=True)
            next_state, reward, done = env.step(action)
            
            agent.update(state, action, reward, next_state, done)
            
            state = next_state
            total_reward += reward

            if done:
                break

        agent.decay_hyperparameters()
        rewards_history.append(total_reward)

        if (episode + 1) % 150 == 0:
            print(f"Episode {episode + 1}/{episodes} | Return: {total_reward:.2f} | Epsilon: {agent.epsilon:.3f} | Alpha: {agent.alpha:.3f}")

    np.save("q_table.npy", agent.q_table)
    print("\nTraining complete. Q-table saved to 'q_table.npy'.")

    #final evaluation
    print("\nEvaluating optimal policy...")
    state = env.reset()
    solution_path = [state]
    done = False
    steps = 0
    visited_counts = {state: 1}

    while not done and steps < max_steps_per_episode:
        action = agent.choose_action(state, train=False)
        next_state, _, done = env.step(action)
        
        visited_counts[next_state] = visited_counts.get(next_state, 0) + 1
        
        if visited_counts[next_state] > 10:
            print("Warning: Agent stuck in a loop during evaluation!")
            break

        solution_path.append(next_state)
        state = next_state
        steps += 1

    if done:
        print(f"Goal reached successfully in {len(solution_path) - 1} steps!")
    else:
        print("Agent failed to reach goal within step limit.")

    #ploting reward curve
    plt.figure(figsize=(8, 4))
    plt.plot(rewards_history)
    plt.title("Episode Reward Over Time")
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.grid(True)
    plt.show()

    env.animate_agent_on_grid(solution_path)


if __name__ == "__main__":
    train()