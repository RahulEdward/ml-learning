# Deep Reinforcement Learning: Building a Trading Agent

Reinforcement Learning (RL) hai a computational approach to goal-directed learning performed by an agent that interacts ke saath a typically stochastic environment which the agent has incomplete information about. RL aims to automate how the agent makes decisions to achieve a long-term objective by learning the value ka states aur actions from a reward signal. The ultimate goal hai to derive a policy that encodes behavioral rules aur maps states to actions.

Yeh chapter shows how to formulate an RL problem aur how to apply various solution methods. It covers model-based aur model-free methods, introduces the [OpenAI Gym](https://gym.openai.com/) environment, aur combines deep learning ke saath RL to train an agent that navigates a complex environment. Finally, hum'll show you how to adapt RL to algorithmic trading by modeling an agent that interacts ke saath the financial market while trying to optimize an objective function. 

#### Table ka contents

1. [Key elements ka a reinforcement learning system](#key-elements-ka-a-reinforcement-learning-system)
    * [The policy: translating states into actions](#the-policy-translating-states-into-actions)
    * [Rewards: learning from actions](#rewards-learning-from-actions)
    * [The value function: optimal decisions for the long run](#the-value-function-optimal-decisions-for-the-long-run)
    * [The environment](#the-environment)
    * [Components of an interactive RL system](#components-of-an-interactive-rl-system)
2. [How to solve RL problems](#how-to-solve-rl-problems)
    * [Code example: dynamic programming – value and policy iteration](#code-example-dynamic-programming--value-and-policy-iteration)
    * [Code example: Q-Learning](#code-example-q-learning)
3. [Deep Reinforcement Learning](#deep-reinforcement-learning)
    * [Value function approximation with neural networks](#value-function-approximation-with-neural-networks)
    * [The Deep Q-learning algorithm and extensions](#the-deep-q-learning-algorithm-and-extensions)
    * [The Open AI Gym – the Lunar Lander environment](#the-open-ai-gym--the-lunar-lander-environment)
    * [Code example: Double Deep Q-Learning using Tensorflow](#code-example-double-deep-q-learning-using-tensorflow)
4. [Code example: deep RL ke liye trading ke saath TensorFlow 2 aur OpenAI Gym](#code-example-deep-rl-ke liye-trading-ke saath-tensorflow-2-aur-openai-gym)
    * [How to Design an OpenAI trading environment](#how-to-design-an-openai-trading-environment)
    * [How to build a Deep Q-learning agent for the stock market](#how-to-build-a-deep-q-learning-agent-for-the-stock-market)
5. [Resources](#resources)
    * [RL Algorithms](#rl-algorithms)
    * [Investment Applications](#investment-applications)

## Key elements ka a reinforcement learning system

RL problems feature several elements that set them apart from the ML settings hum have covered so far. The following two sections outline the key features required ke liye defining aur solving an RL problem by learning a policy that automates decisions. 
hum’ll use the notation aur generally follow [Reinforcement Learning: An Introduction](https://web.stanford.edu/class/psych209/Readings/SuttonBartoIPRLBook2ndEd.pdf) (Sutton aur Barto 2018) aur David Silver’s [UCL Courses on RL](https://www.davidsilver.uk/teaching/) that hain recommended ke liye further study beyond the brief summary that the scope ka this chapter permits.

RL problems aim to optimize an agent's decisions based on an objective function vis-a-vis an environment.

### The policy: translating states into actions
At any point mein time, the policy defines the agent’s behavior. It maps any state the agent may encounter to one or several actions. mein an environment ke saath a limited number ka states aur actions, the policy can be a simple lookup table filled mein during training. 

### Rewards: learning from actions

The reward signal hai a single value that the environment sends to the agent at each time step. The agent’s objective hai typically to maximize the total reward received over time. Rewards can also be a stochastic function ka the state aur the actions. They hain typically discounted to facilitate convergence aur reflect the time decay ka value.
 
### The value function: optimal decisions ke liye the long run
The reward provide karta hai immediate feedback on actions. However, solving an RL problem requires decisions that create value mein the long run. Yeh hai where the value function comes mein: it summarizes the utility ka states or ka actions mein a given state mein terms ka their long-term reward. 
 
### The environment
The environment presents information about its state to the agent, assigns rewards ke liye actions, aur transitions the agent to new states subject to probability distributions the agent may or may not know about. 
It may be fully or partially observable, aur may also contain other agents. The design ka the environment typically requires significant up-front design effort to facilitate goal-oriented learning by the agent during training.

RL problems differ by the complexity ka their state aur action spaces that can be either discrete or continuous. The latter requires ML to approximate a functional relationship between states, actions, aur their value. They also require us to generalize from the subset ka states aur actions they hain experienced by the agent during training.

### Components ka an interactive RL system

The components ka an RL system typically include:

- Observations by the agent ka the state ka the environment
- A set ka actions that hain available to the agent
- A policy that governs the agent's decisions

mein addition, the environment emits a reward signal that reflects the new state resulting from the agent's action. At the core, the agent usually learns a value function that shapes its judgment over actions. The agent has an objective function to process the reward signal aur translate the value judgments into an optimal policy.

## How to solve RL problems

RL methods aim to learn from experience on how to take actions that achieve a long-term goal. To this end, the agent aur the environment interact over a sequence ka discrete time steps via the interface ka actions, state observations, aur rewards that hum described mein the previous section.

There hain numerous approaches to solving RL problems which implies finding rules ke liye the agent's optimal behavior:

- **Dynamic programming** (DP) methods make the often unrealistic assumption ka complete knowledge ka the environment, but hain the conceptual foundation ke liye most other approaches.
- **Monte Carlo** (MC) methods learn about the environment aur the costs aur benefits ka different decisions by sampling entire state-action-reward sequences.
- **Temporal difference** (TD) learning significantly improves sample efficiency by learning from shorter sequences. To this end, it relies on bootstrapping, which hai defined as refining its estimates based on its own prior estimates.

Approaches ke liye continuous state aur/or action spaces often leverage ML to approximate a value or policy function. Hence, they integrate supervised learning, aur mein particular, the deep learning methods hum discussed mein the last several chapters. However, these methods face distinct challenges mein the RL context:

- The reward signal does not directly reflect the target concept, such as a labeled sample
- The distribution ka the observations depends on the agent's actions aur the policy which hai itself the subject ka the learning process

### Code example: dynamic programming – value aur policy iteration

Finite MDPs hain a simple yet fundamental framework. Yeh section introduces the trajectories ka rewards that the agent aims to optimize, aur define the policy aur value functions they hain used to formulate the optimization problem aur the Bellman equations that form the basis ke liye the solution methods.

Notebook [gridworld_dynamic_programming](01_gridworld_dynamic_programming.ipynb) applies Value aur Policy Iteration to a toy environment that consists ka a 3 x 4 grid.

### Code example: Q-Learning

Q-learning was an early RL breakthrough when it was developed by Chris Watkins ke liye his [PhD thesis]((http://www.cs.rhul.ac.uk/~chrisw/thesis.html)) mein 1989 . It introduces incremental dynamic programming to control an MDP without knowing or modeling the transition aur reward matrices that hum used ke liye value aur policy iteration mein the previous section. A convergence proof followed three years later by [Watkins aur Dayan](http://www.gatsby.ucl.ac.uk/~dayan/papers/wd92.html).

Q-learning directly optimizes the action-value function, q, to approximate q*. The learning proceeds off-policy, that hai, the algorithm does not need to select actions based on the policy that's implied by the value function alone. However, convergence requires that all state-action pairs continue to be updated throughout the training process. A straightforward way to ensure this hai by use karke an ε-greedy policy.

The Q-learning algorithm keeps improving a state-action value function after random initialization ke liye a given number ka episodes. At each time step, it chooses an action based on an ε-greedy policy, aur use karta hai a learning rate, α, to update the value function based on the reward  aur its current estimate ka the value function ke liye the next state.

Notebook [gridworld_q_learning](02_gridworld_q_learning.ipynb) demonstrate karta hai how to build a Q-learning agent use karke the 3 x 4 grid ka states from the previous section.

## Deep Reinforcement Learning

Yeh section adapts Q-Learning to continuous states aur actions where hum cannot use the tabular solution that simply fills an array ke saath state-action values. Instead, hum will see how to approximate the optimal state-value function use karke a neural network to build a deep Q network ke saath various refinements to accelerate convergence. hum will then see how hum can use the [OpenAI Gym](http://gym.openai.com/docs/) to apply the algorithm to the [Lunar Lander](https://gym.openai.com/envs/LunarLander-v2) environment.

### Value function approximation ke saath neural networks

As mein other fields, deep neural networks have become popular ke liye approximating value functions. However, ML faces distinct challenges mein the RL context where the data hai generated by the interaction ka the model ke saath the environment use karke a (possibly randomized) policy:

- ke saath continuous states, the agent will fail to visit most states aur, thus, needs to generalize.
- Supervised learning aims to generalize from a sample ka independently aur identically distributed samples that hain representative aur correctly labeled. mein the RL context, there hai only one sample per time step so that learning needs to occur online.
- Samples can be highly correlated when sequential states hain similar aur the behavior distribution over states aur actions hai not stationary, but changes as a result ka the agent's learning.

### The Deep Q-learning algorithm aur extensions

Deep Q learning estimates the value ka the available actions ke liye a given state use karke a deep neural network. It was introduced by Deep Mind's [Playing Atari ke saath Deep Reinforcement Learning](https://www.cs.toronto.edu/~vmnih/docs/dqn.pdf) (2013), where RL agents learned to play games solely from pixel input.

The deep Q-learning algorithm approximates the action-value function, q, by learning a set ka weights, θ, ka a multi-layered Deep Q Network (DQN) that maps states to actions.

Several innovations have improved the accuracy aur convergence speed ka deep Q-Learning, namely:
- **Experience replay** stores a history ka state, action, reward, aur next state transitions aur randomly samples mini-batches from this experience to update the network weights at each time step before the agent selects an ε-greedy action. It increases sample efficiency, reduces the autocorrelation ka samples, aur limits the feedback due to the current weights producing training samples that can lead to local minima or divergence.
- **Slowly-changing target network** weakens the feedback loop from the current network parameters on the neural network weight updates. Also invented by by Deep Mind mein [Human-level control through deep reinforcement learning](https://web.stanford.edu/class/psych209/Readings/MnihEtAlHassibis15NatureControlDeepRL.pdf) (2015), it use a slowly-changing target network that has the same architecture as the Q-network, but its weights hain only updated periodically. The target network generates the predictions ka the next state value used to update the Q-Networks estimate ka the current state's value.
- **Double deep Q-learning** addresses the bias ka deep Q-Learning to overestimate action values because it purposely samples the highest action value. Yeh bias can negatively affect the learning process aur the resulting policy if it does not apply uniformly , as shown by Hado van Hasselt mein [Deep Reinforcement Learning ke saath Double Q-learning](https://arxiv.org/abs/1509.06461) (2015). To decouple the estimation ka action values from the selection ka actions, Double Deep Q-Learning (DDQN) use karta hai the weights, ka one network to select the best action given the next state, aur the weights ka another network to provide the corresponding action value estimate.

### The Open AI Gym – the Lunar Lander environment

The [OpenAI Gym](https://gym.openai.com/) hai a RL platform that provide karta hai standardized environments to test aur benchmark RL algorithms use karke Python. It hai also possible to extend the platform aur register custom environments.

The [Lunar Lander](https://gym.openai.com/envs/LunarLander-v2) (LL) environment requires the agent to control its motion mein two dimensions, based on a discrete action space aur low-dimensional state observations that include position, orientation, aur velocity. At each time step, the environment provide karta hai an observation ka the new state aur a positive or negative reward. Each episode consists ka up to 1,000 time steps.

### Code example: Double Deep Q-Learning use karke Tensorflow

The [lunar_lander_deep_q_learning](03_lunar_lander_deep_q_learning.ipynb) notebook implements a DDQN agent that use karta hai TensorFlow aur Open AI Gym's Lunar Lander environment.

## Code example: deep RL ke liye trading ke saath TensorFlow 2 aur OpenAI Gym

To train a trading agent, hum need to create a market environment that provide karta hai price aur other information, offers trading-related actions, aur keeps track ka the portfolio to reward the agent accordingly.

### How to Design an OpenAI trading environment

The OpenAI Gym allows ke liye the design, registration, aur utilization ka environments that adhere to its architecture, as described mein its [documentation](https://github.com/openai/gym/tree/master/gym/envs#how-to-create-new-environments-ke liye-gym). 
- The [trading_env.py](trading_env.py) file implements an example that illustrates how to create a class that implements the requisite `step()` aur `reset()` methods.

The trading environment consists ka three classes that interact to facilitate the agent's activities:
 1. The `DataSource` class loads a time series, generates a few features, aur provide karta hai the latest observation to the agent at each time step. 
 2. `TradingSimulator` tracks the positions, trades aur cost, aur the performance. It also implements aur records the results ka a buy-aur-hold benchmark strategy. 
 3. `TradingEnvironment` itself orchestrates the process. 
 
### How to build a Deep Q-learning agent ke liye the stock market
 
Notebook [q_learning_for_trading](04_q_learning_for_trading.ipynb) demonstrate karta hai how to set up a simple game ke saath a limited set ka options, a relatively low-dimensional state, aur other parameters that can be easily modified aur extended to train the Deep Q-Learning agent used mein [lunar_lander_deep_q_learning](03_lunar_lander_deep_q_learning.ipynb).
 
<p align="center">
<img src="https://i.imgur.com/lg0ofbZ.png" width="60%">
</p>


## Sansadhan (Resources)

- [Reinforcement Learning: An Introduction, 2nd eition](http://incompleteideas.net/book/RLbook2018.pdf), Richard S. Sutton aur Andrew G. Barto, 2018
- [University College ka London Course on Reinforcement Learning](http://www0.cs.ucl.ac.uk/staff/d.silver/web/Teaching.html), David Silver, 2015
- [Implementation ka Reinforcement Learning Algorithms](https://github.com/dennybritz/reinforcement-learning), Denny Britz
    - This repository provides code, exercises and solutions for popular Reinforcement Learning algorithms. These are meant to serve as a learning tool to complement the theoretical materials from Sutton/Baron and Silver (see above).

### RL Algorithms

- Q Learning
    - [Learning from Delayed Rewards](http://www.cs.rhul.ac.uk/~chrisw/thesis.html), PhD Thesis, Chris Watkins, 1989
    - [Q-Learning](http://www.gatsby.ucl.ac.uk/~dayan/papers/wd92.html), Machine Learning, 1992
- Deep Q Networks
    - [Playing Atari with Deep Reinforcement Learning](https://www.cs.toronto.edu/~vmnih/docs/dqn.pdf), Mnih et al, 2013
    - We present the first deep learning model to successfully learn control policies directly from high-dimensional sensory input using reinforcement learning. The model is a convolutional neural network, trained with a variant of Q-learning, whose input is raw pixels and whose output is a value function estimating future rewards. We apply our method to seven Atari 2600 games from the Arcade Learning Environment, with no adjustment of the architecture or learning algorithm. We find that it outperforms all previous approaches on six of the games and surpasses a human expert on three of them.
- Asynchronous Advantage Actor-Critic (A2C/A3C)
    - [Asynchronous Methods for Deep Reinforcement Learning](https://arxiv.org/abs/1602.01783), Mnih, V. et al. 2016
    - We propose a conceptually simple and lightweight framework for deep reinforcement learning that uses asynchronous gradient descent for optimization of deep neural network controllers. We present asynchronous variants of four standard reinforcement learning algorithms and show that parallel actor-learners have a stabilizing effect on training allowing all four methods to successfully train neural network controllers. The best performing method, an asynchronous variant of actor-critic, surpasses the current state-of-the-art on the Atari domain while training for half the time on a single multi-core CPU instead of a GPU. Furthermore, we show that asynchronous actor-critic succeeds on a wide variety of continuous motor control problems as well as on a new task of navigating random 3D mazes using a visual input.
- Proximal Policy Optimization (PPO)
    - [Proximal Policy Optimization Algorithms](https://arxiv.org/abs/1707.06347), Schulman et al, 2017
    - We propose a new family of policy gradient methods for reinforcement learning, which alternate between sampling data through interaction with the environment, and optimizing a "surrogate" objective function using stochastic gradient ascent. Whereas standard policy gradient methods perform one gradient update per data sample, we propose a novel objective function that enables multiple epochs of minibatch updates. The new methods, which we call proximal policy optimization (PPO), have some of the benefits of trust region policy optimization (TRPO), but they are much simpler to implement, more general, and have better sample complexity (empirically). Our experiments test PPO on a collection of benchmark tasks, including simulated robotic locomotion and Atari game playing, and we show that PPO outperforms other online policy gradient methods, and overall strikes a favorable balance between sample complexity, simplicity, and wall-time.

- Trust Region Policy Optimization (TRPO)
    - [Trust Region Policy Optimization](https://arxiv.org/abs/1502.05477), Schulman et al, 2015
    - We describe an iterative procedure for optimizing policies, with guaranteed monotonic improvement. By making several approximations to the theoretically-justified procedure, we develop a practical algorithm, called Trust Region Policy Optimization (TRPO). This algorithm is similar to natural policy gradient methods and is effective for optimizing large nonlinear policies such as neural networks. Our experiments demonstrate its robust performance on a wide variety of tasks: learning simulated robotic swimming, hopping, and walking gaits; and playing Atari games using images of the screen as input. Despite its approximations that deviate from the theory, TRPO tends to give monotonic improvement, with little tuning of hyperparameters.
    
- Deep Deterministic Policy Gradient (DDPG)
    - [Continuous control with deep reinforcement learning](https://arxiv.org/abs/1509.02971), Lillicrap et al, 2015
    - We adapt the ideas underlying the success of Deep Q-Learning to the continuous action domain. We present an actor-critic, model-free algorithm based on the deterministic policy gradient that can operate over continuous action spaces. Using the same learning algorithm, network architecture and hyper-parameters, our algorithm robustly solves more than 20 simulated physics tasks, including classic problems such as cartpole swing-up, dexterous manipulation, legged locomotion and car driving. Our algorithm is able to find policies whose performance is competitive with those found by a planning algorithm with full access to the dynamics of the domain and its derivatives. We further demonstrate that for many of the tasks the algorithm can learn policies end-to-end: directly from raw pixel inputs.
- Twin Delayed DDPG (TD3)
    - [Addressing Function Approximation Error in Actor-Critic Methods](https://arxiv.org/abs/1802.09477), Fujimoto et al, 2018
    - In value-based reinforcement learning methods such as deep Q-learning, function approximation errors are known to lead to overestimated value estimates and suboptimal policies. We show that this problem persists in an actor-critic setting and propose novel mechanisms to minimize its effects on both the actor and the critic. Our algorithm builds on Double Q-learning, by taking the minimum value between a pair of critics to limit overestimation. We draw the connection between target networks and overestimation bias, and suggest delaying policy updates to reduce per-update error and further improve performance. We evaluate our method on the suite of OpenAI gym tasks, outperforming the state of the art in every environment tested.
- Soft Actor-Critic (SAC)
    - [Soft Actor-Critic: Off-Policy Maximum Entropy Deep Reinforcement Learning with a Stochastic Actor](https://arxiv.org/abs/1801.01290), Haarnoja et al, 2018
    - Model-free deep reinforcement learning (RL) algorithms have been demonstrated on a range of challenging decision making and control tasks. However, these methods typically suffer from two major challenges: very high sample complexity and brittle convergence properties, which necessitate meticulous hyperparameter tuning. Both of these challenges severely limit the applicability of such methods to complex, real-world domains. In this paper, we propose soft actor-critic, an off-policy actor-critic deep RL algorithm based on the maximum entropy reinforcement learning framework. In this framework, the actor aims to maximize expected reward while also maximizing entropy. That is, to succeed at the task while acting as randomly as possible. Prior deep RL methods based on this framework have been formulated as Q-learning methods. By combining off-policy updates with a stable stochastic actor-critic formulation, our method achieves state-of-the-art performance on a range of continuous control benchmark tasks, outperforming prior on-policy and off-policy methods. Furthermore, we demonstrate that, in contrast to other off-policy algorithms, our approach is very stable, achieving very similar performance across different random seeds.
- Categorical 51-Atom DQN (C51)
    - [A Distributional Perspective on Reinforcement Learning](https://arxiv.org/abs/1707.06887), Bellemare, et al 2017
    - In this paper we argue for the fundamental importance of the value distribution: the distribution of the random return received by a reinforcement learning agent. This is in contrast to the common approach to reinforcement learning which models the expectation of this return, or value. Although there is an established body of literature studying the value distribution, thus far it has always been used for a specific purpose such as implementing risk-aware behaviour. We begin with theoretical results in both the policy evaluation and control settings, exposing a significant distributional instability in the latter. We then use the distributional perspective to design a new algorithm which applies Bellman's equation to the learning of approximate value distributions. We evaluate our algorithm using the suite of games from the Arcade Learning Environment. We obtain both state-of-the-art results and anecdotal evidence demonstrating the importance of the value distribution in approximate reinforcement learning. Finally, we combine theoretical and empirical evidence to highlight the ways in which the value distribution impacts learning in the approximate setting.
    
### Investment Applications
- [A Deep Reinforcement Learning Framework ke liye the Financial Portfolio Management Problem](https://arxiv.org/abs/1706.10059), Zhengyao Jiang, Dixing Xu, Jinjun Liang 2017
    - Financial portfolio management is the process of constant redistribution of a fund into different financial products. This paper presents a financial-model-free Reinforcement Learning framework to provide a deep machine learning solution to the portfolio management problem. The framework consists of the Ensemble of Identical Independent Evaluators (EIIE) topology, a Portfolio-Vector Memory (PVM), an Online Stochastic Batch Learning (OSBL) scheme, and a fully exploiting and explicit reward function. This framework is realized in three instants in this work with a Convolutional Neural Network (CNN), a basic Recurrent Neural Network (RNN), and a Long Short-Term Memory (LSTM). They are, along with a number of recently reviewed or published portfolio-selection strategies, examined in three back-test experiments with a trading period of 30 minutes in a cryptocurrency market. Cryptocurrencies are electronic and decentralized alternatives to government-issued money, with Bitcoin as the best-known example of a cryptocurrency. All three instances of the framework monopolize the top three positions in all experiments, outdistancing other compared trading algorithms. Although with a high commission rate of 0.25% in the backtests, the framework is able to achieve at least 4-fold returns in 50 days.
    - [PGPortfolio](https://github.com/ZhengyaoJiang/PGPortfolio); corresponding GitHub repo
- [Financial Trading as a Game: A Deep Reinforcement Learning Approach](https://arxiv.org/pdf/1807.02787.pdf), Huang, Chien-Yi, 2018
- [Order placement ke saath Reinforcement Learning](https://github.com/mjuchli/ctc-executioner)
    - CTC-Executioner is a tool that provides an on-demand execution/placement strategy for limit orders on crypto currency markets using Reinforcement Learning techniques. The underlying framework provides functionalities which allow to analyse order book data and derive features thereof. Those findings can then be used in order to dynamically update the decision making process of the execution strategy.
    - The methods being used are based on a research project (master thesis) currently proceeding at TU Delft.
    
- [Q-Trader](https://github.com/edwardhdlu/q-trader)
    - An implementation of Q-learning applied to (short-term) stock trading. The model uses n-day windows of closing prices to determine if the best action to take at a given time is to buy, sell or sit. As a result of the short-term state representation, the model is not very good at making decisions over long-term trends, but is quite good at predicting peaks and troughs.
    