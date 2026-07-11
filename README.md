# Path-dependent-volatility models

The code in this branch is designed to implement a simple linear model for path-dependent volatility described in the paper by Guyon and Lekeufack "Volatility is (Mostly) Path-Dependent". The principle behind path-dependent models is that we can use information concerning the entire path of data points up to a given point in time $t_0$. The model we build takes as input the vector of returns at all time increments $t \leq t_0$. We now describe specifically how this model is built.

Let $S_t$ denote the asset price at time $t$, and define $r_t = (S_t - S_{t-1})/ S_{t-1}$ to be the simple return at time increment $t$. We will build out a linear model of the form:
$$
\text{Volatility}_{t_0} = \beta_0 + \beta_1 R_{t_0} + \beta_2 \Sigma_{t_0}
$$
where $R_{t_0}, \Sigma_{t_0}$ are both functions of the entire sequence of returns $\{ r_t\}_{t \leq t_0}$. In reality, $R_t$ depends directly on the simple returns $r_t$, while $\Sigma_t$ depends on the sequence of squared simple returns $r_t^2$.

$R_t$ is designed to learn the so-called "leverage effect", namely that volatility tends to rise when assets fall. On the other hand, $\Sigma_t$ is meant to learn "volatility clustering", which describes the way in which periods of low/high volatility tend to be alongside other periods of low/high volatility. Naturally, both of these terms depend more heavily on the recent past as compared to the further past.

In particular, we express $R_t = \sum_{t \leq t_0} K_1(t_0 - t) \cdot r_t$ and $\Sigma_t = \sqrt{ \sum_{t \leq t_0} K_2(t_0 - t) \cdot r_t^2 }$, where the coefficient sequences determined by the functions $K_1, K_2$ decay to zero as the inputs grow. There are many possible choices for such functions, but Guyon and Lekeufack advocate for the use of time-shifted power laws, which are functions of the form $K(t) = \frac{1}{C} (t+\delta)^\alpha$ (where $\delta > 0$ and $\alpha > 1$ are constants, and $C$ is a normalization constant depending only on $\delta, \alpha$.

The code we implement here jointly fits the seven parameters $\beta_0, \beta_1, \beta_2, \alpha_1, \delta_1, \alpha_2, \delta_2$ all at once, and relies on reasonable initialization constants (since the optimization problem is non-convex). For fixed kernels $K_1, K_2$, fitting the $\beta_i$'s is an ordinary least squares regression problem. Once the relevant constants are in hand, volatility predictions can be made given the sequence of relevant returns $(r_t)_{t \leq t_0}$.
