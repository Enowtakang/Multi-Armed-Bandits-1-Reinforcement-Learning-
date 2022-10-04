"""
Multi-Armed Bandits Algorithm
"""
import numpy as np
import matplotlib.pyplot as plt

"""
Function to plot the history of the agent 
interaction with the environment
"""


def plot_history(history):
    rewards = history['rewards']
    cum_rewards = history['cum_rewards']
    chosen_arms = history['arms']

    fig = plt.figure(figsize=[30, 8])

    ax2 = fig.add_subplot(121)
    ax2.plot(cum_rewards, label='avg_rewards')
    ax2.set_title('Cumulative Rewards')

    ax3 = fig.add_subplot(122)
    ax3.bar(
        [i for i in range(len(chosen_arms))],
        chosen_arms, label='chosen_arms')
    ax3.set_title('Chosen Actions')
    plt.show()


"""
- Create an environment
- Set the probability of payout
- Set how much the environment actually pays out
"""


class Environment(object):
    """
    - Define __init__ constructor
    - Set it to take the following variables:
            reward_probabilities (RP)
            rewards (Rs)
    - RP is a list of probabilities with which
        each arm can reward the agent
    - Rs is a list of the actual rewards gotten
        from each arm

    """
    def __init__(
            self,
            reward_probabilities,
            rewards):
        """
        Ensure that the lengths of both
        Reward Probabilities and Rewards are equal
        """
        a1 = len(reward_probabilities)
        a2 = len(rewards)
        if a1 != a2:
            raise Exception(
                f"Size of reward_probabilities: {a1} does not match size of rewards: {a2}")

        """
        If the test above results in no problem,
        """
        self.reward_probabilities = reward_probabilities
        self.rewards = rewards

        """
        Let the Environment know how many arms it
        is dealing with
        """
        self.k_arms = a2  # Or a1. Same length

    """
    Next, create a function called choose_arm
        which takes any arm.
    For example, if k_arms is 10, then the agent
        can choose any arm between 0 and 9
    """
    def choose_arm(self, arm):
        """
        No arm should be less than 0
        No arm should be greater than k_arms
        """
        if arm < 0 or arm > self.k_arms:
            p1 = self.k_arms - 1
            raise Exception(f"Arm ust be a value between 0 and {p1}")

        """Otherwise, """
        return self.rewards[arm] if np.random.random() < self.reward_probabilities[arm] else 0.0


"""
Example: 
In the environment variable below, 3 arms have
    been specified.
    The 1st arm has a 100% chance (0.0) 
        of rewarding you with $0.
    The 2nd arm has a 50% chance (0.50) 
        of rewarding you with $50.
    The 3rd arm has a 90% chance (0.90) 
        of rewarding you with $10.

We shall then proceed to choose the 1st arm 10 times
    (knowing that we won't get 
    anything in return, 10 times)  
"""


def set_example_env():
    enviroment = Environment(
        reward_probabilities=[0.0, 0.50, 0.90],
        rewards=[100, 50, 10])
    print(
        f"Reward Probabilities\t\t: {enviroment.reward_probabilities}")
    print(
        f"Rewards\t\t: {enviroment.rewards}")


# set_example_env()


def pull_arm(arm_number, num_times):
    enviroment = Environment(
        reward_probabilities=[1.0, 0.50, 0.90],
        rewards=[0.0, 50.0, 10.0])

    list_ans = [enviroment.choose_arm(
        arm_number) for _ in range(num_times)]
    print(list_ans)


# pull_arm(0, 10)  # Pulling the first arm 10 times


"""
Next, we build an 'intelligent' agent.
    To do this, we shall actually build MANY agents.
    Then, we shall build a base agent, whose performance
    would be used as the benchmark for comparing the 
    other agents.
    
    - Normaly, while for the first time in, say, 
        a casino, you don't know which machine does best.
        So you need to pull all arms in all machines 
        at random, to get an initial clue.
    - The fist agent shall be built on this basis 
    - This RANDOM Agent shall interact with the 
        environment which we created
"""


class RandomAgent(object):
    """
    In the constructor below, we specify
    how many times the Agent would interact
    with the environment, using the
        'max_iterations' variable
    """
    def __init__(self, env,
                 max_iterations=2000):
        """
        Within these 2000 iterations, we want
        the agent to make the most cumulative
        rewards which it possibly can

        - Pass an instance of the environment to
        the agent
        """
        self.env = env
        self.iterations = max_iterations

    """
    We proceed to create a function which would
    enable the agent to act within the environment
    """
    def act(self):
        """
        For an intelligent AGENT:
            We need to find away to make the agent
            able to make the optimal decision
            (optimize cumulative reward in an
            uncertain environment)

        - Here however, we want the agent to
            behave randomly.
        - So we have to take note of
            1./ the arms the agent is pulling,
            2./ how many times each arm is pulled
            3./ and the rewards it is obtaining
        """
        arm_counts = np.zeros(self.env.k_arms)
        rewards = []
        cum_rewards = []  # it is really the avg rewards
        # We expect the avg rewards to be increasing
        # with time, for an INTELLIGENT Agent

        """Let the agent act"""
        for i in range(self.iterations + 1):
            """
            We add one above so that counting shall
            begin from one, instead of zero.
            
            Now, it is time for the agent to 
            randomly choose an arm
            """
            arm = np.random.choice(self.env.k_arms)
            reward = self.env.choose_arm(arm)

            arm_counts[arm] += 1
            rewards.append(reward)

            avg1 = sum(rewards)/len(rewards)
            cum_rewards.append(avg1)

        return {"arms": arm_counts,
                "rewards": rewards,
                "cum_rewards": cum_rewards}


"""
We now create an instance and evaluate the action 
    of the random agent
"""


def evaluate_random_agent():
    enviroment = Environment(
        reward_probabilities=[1.0, 0.50, 0.90],
        rewards=[0.0, 50.0, 10.0])

    random_agent = RandomAgent(
        env=enviroment, max_iterations=2000)

    r_a_history = random_agent.act()

    """
    Check total agent reward
    """
    print(
        f"TOTAL REWARD: {sum(r_a_history['rewards'])}")

    """
    Visualize results
    """
    plot_history(r_a_history)


# evaluate_random_agent()


"""
We just built the base case.

Now, let us build an intelligent Agent. 
    We begin with the EPSILON-GREEDY Agent
    
    Here EXPLORE and EXPLOIT is its methodology
    For example, if it pulls arms 1, 2 and 3, 
    '''5 times each''', and gets 
    2, 30, 70 rewards (EXPLORE) respectively, then 
    common sense requires that 
    it keeps pulling arm 3 (EXPLOIT).
    
    So, exploration is done a few times (ON ALL ARMS), 
    then exploitation (on the basis of the exploration
    results) is done, TILL THE LAST iteration.
    
    Remember that with the random agent, not all arms
        MUST BE explored. 
        Epsilon-greedy algorithm-based agent solves this
"""


class EpsilonGreedyAgent(object):
    def __init__(self, env,
                 max_iterations=2000,
                 epsilon=0.01):
        self.env = env
        self.iterations = max_iterations
        self.epsilon = epsilon

    def act(self):
        """
        - Initialize q_values.
            Since it is a fresh start, it is assumed
            that the q_value for each arm is
            currently zero
        - Initialize arm_rewards.
            Since it is a fresh start, it is assumed
            that the arm_reward for each arm is
            currently zero
        - Initialize arm_counts.
            Since it is a fresh start, it is assumed
            that the arm_count for each arm is
            currently zero
        """
        q_values = np.zeros(self.env.k_arms)
        arm_rewards = np.zeros(self.env.k_arms)
        arm_counts = np.zeros(self.env.k_arms)

        rewards = []
        cum_rewards = []

        """
        This is how EPSILON WORKS:
            - With epsilon probability, it explores
            - With 1 - epsilon probability, it 
                exploits
        """
        for i in range(1, self.iterations + 1):
            """
            In the code below:
                - we take a random number.
                - if it is less than epsilon, 
                    we want to explore
                - otherwise, we take the action 
                    (exploitation) with the 
                    maximum value 
             """
            arm = np.random.choice(
                self.env.k_arms) if np.random.random(
                ) < self.epsilon else np.argmax(
                q_values)

            reward = self.env.choose_arm(arm)

            arm_rewards[arm] += reward

            arm_counts[arm] += 1

            rp = arm_rewards[arm]/arm_counts[arm]
            q_values[arm] = rp

            rewards.append(reward)
            avg1 = sum(rewards) / len(rewards)
            cum_rewards.append(avg1)

        return {"arms": arm_counts,
                "rewards": rewards,
                "cum_rewards": cum_rewards}


"""
We now create an instance and evaluate the action 
    of the epsilon greedy agent
"""


def evaluate_epsilon_greedy_agent():
    enviroment = Environment(
        reward_probabilities=[1.0, 0.50, 0.90],
        rewards=[0.0, 50.0, 10.0])

    e_greedy_agent = EpsilonGreedyAgent(
        env=enviroment,
        max_iterations=2000,
        epsilon=0.1)  # Epsilon is 0.1, in this case

    e_greedy_history = e_greedy_agent.act()

    """
        Check total agent reward
        """
    print(
        f"TOTAL REWARD: {sum(e_greedy_history['rewards'])}")

    """
    Visualize results
    """
    plot_history(e_greedy_history)


# evaluate_epsilon_greedy_agent()


"""
Epsilon Greedy Agent With Decay
"""


class EpsilonGreedyAgentWithDecay(object):
    def __init__(self, env,
                 max_iterations=2000,
                 epsilon=0.01,
                 decay=0.001,
                 decay_interval=50):
        self.env = env
        self.iterations = max_iterations
        self.epsilon = epsilon
        self.decay = decay
        self.decay_interval = decay_interval

    def act(self):
        """
        - Initialize q_values.
            Since it is a fresh start, it is assumed
            that the q_value for each arm is
            currently zero
        - Initialize arm_rewards.
            Since it is a fresh start, it is assumed
            that the arm_reward for each arm is
            currently zero
        - Initialize arm_counts.
            Since it is a fresh start, it is assumed
            that the arm_count for each arm is
            currently zero
        """
        q_values = np.zeros(self.env.k_arms)
        arm_rewards = np.zeros(self.env.k_arms)
        arm_counts = np.zeros(self.env.k_arms)

        rewards = []
        cum_rewards = []

        """
        This is how EPSILON WORKS:
            - With epsilon probability, it explores
            - With 1 - epsilon probability, it 
                exploits
        """
        for i in range(1, self.iterations + 1):
            """
            In the code below:
                - we take a random number.
                - if it is less than epsilon, 
                    we want to explore
                - otherwise, we take the action 
                    (exploitation) with the 
                    maximum value 
             """
            arm = np.random.choice(
                self.env.k_arms) if np.random.random(
                ) < self.epsilon else np.argmax(
                q_values)

            reward = self.env.choose_arm(arm)

            arm_rewards[arm] += reward

            arm_counts[arm] += 1

            rp = arm_rewards[arm]/arm_counts[arm]
            q_values[arm] = rp

            rewards.append(reward)
            avg1 = sum(rewards) / len(rewards)
            cum_rewards.append(avg1)

            if i % self.decay_interval == 0:
                d = self.epsilon * self.decay
                self.epsilon = d

        return {"arms": arm_counts,
                "rewards": rewards,
                "cum_rewards": cum_rewards}


"""
We now create an instance and evaluate the action 
    of the epsilon greedy agent
"""


def evaluate_epsilon_greedy_agent_with_decay():
    enviroment = Environment(
        reward_probabilities=[1.0, 0.50, 0.90],
        rewards=[0.0, 50.0, 10.0])

    e_greedy_agent_with_decay = EpsilonGreedyAgentWithDecay(
        env=enviroment,
        max_iterations=2000,
        epsilon=0.1,
        decay=1)  # epsilon=0.1, decay=1.0, in THIS case

    e_greedy_history_decay = e_greedy_agent_with_decay.act()

    """
        Check total agent reward
        """
    print(
        f"TOTAL REWARD: {sum(e_greedy_history_decay['rewards'])}")

    """
    Visualize results
    """
    plot_history(e_greedy_history_decay)


# evaluate_epsilon_greedy_agent_with_decay()

