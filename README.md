## Deep-Q-Learning-on-Stock-Data

The best of deep learning and reinforcement learning to predict Stock price.


Reinforcement learning is the branch of machine learning that deals with this type of sequential decision making in a stochastically changing environment. At the time of each decision, there are a number of states and a number of possible actions. The decision maker takes an action, $A_0$, at time zero when the state $S_0$ is known. The results in a reward, $R_1$, at time 1 and a new state, $S_1$, is then encountered. The decision maker then takes another action, $A_1$ which results in a reward, $R_2$ at time 2 and  new state, $S_2$; and so on. The algorithm receives a rewards when outcomes are good and incurs costs (negative rewards) when they are bad. The objective of the algorithm is to maximize expected future rewards possibly with discounting.

The exploitation/exploration trade-off plays a central role to reinforcement learning. We have been considering a decay factor $\beta=0.995$ that reduces the probability of exploration on trial.

# Link

A *Link* is an object that holds parameters (i.e. optimization targets). It is an object that combines parameters and optimizes the parameters. The fundamental form of a link can be represented by a function whose arguments are parameters. On the most frequently used links is *Linear* link (a.k.a. *fully-connected layer* or *affine transformation*). It represents a mathematical function 
```math
 f(x) = Wx+b 
 ```
where the matrix $W$ and the vector $b$ are parameters. The parameters of a link are stored as attributes as instance of *Variable*. $W$ is initialized randomly, whereas $b$ is initialized with zeros. Gradients of parameters are computed by the *backward()* method. Note that gradients are accumulated by the method rather than overwritten. So run first *cleargrads()* method.

# Chain

A *Chain*
