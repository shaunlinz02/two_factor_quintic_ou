The two-factor Quintic OU stochastic volatility model is described as

The two-factor Quintic OU model has the following model parameters
$$
(\lambda_x, \lambda_y, \theta, \rho, \alpha_0, \alpha_1, \alpha_2, \alpha_3, \alpha_4, \alpha_5),
$$
where $\lambda_x, \lambda_y >0$, $\theta \geq 0$, $\alpha_k \in \mathbb{R}$, together with the deterministic input curve $g_0$ that can be used to match the market term structure of volatility.

For example, setting 
$$
    g_0(t):=\sqrt{\frac{\xi_0(t)}{\mathbb{E} \left[p(Z_t)^2\right]}}
$$
allows the model to match the term structure of the forward variance swap observable on the market, since
$$
    \mathbb{E}\left[ \int_0^T \sigma_t^2 dt \right] = \int_0^T \xi_0(t) dt, \quad T > 0.
$$
