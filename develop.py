from time import perf_counter as pc
import numpy as np
from scipy.integrate import odeint
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
import pandas as pd


class DataAssimilation:
    def __init__(self, df: pd.DataFrame,  q0, alpha0):
        self.C_e = df['ConfirmedCases'].tolist()
        self.D_e = df['DeadCases'].tolist()

        nt = df.shape[0]
        self.t0 = 0
        self.tf = self.t0 + nt - 1
        self.time = np.linspace(self.t0, self.tf, nt)
        self.C = np.array([[0, 1, 1, 1], [0, 0, 0, 1]])
        self.q0 = q0
        self.beta = [alpha0[0]] * nt
        self.gamma = [alpha0[1]] * nt
        self.delta = [alpha0[2]] * nt
        self.q = self.S, self.I, self.R, self.D = [None, None, None, None]
        self.qs = [0, 0, 0, 0]
        self.djdbeta = None
        self.djdgamma = None
        self.djddelta = None

    def e_f(self, x):
        return np.array([
            interp1d(self.time, self.C_e, fill_value="extrapolate")(x),
            interp1d(self.time, self.D_e, fill_value="extrapolate")(x)
        ])

    def beta_f(self, x):
        return interp1d(self.time, self.beta, fill_value="extrapolate")(x)

    def gamma_f(self, x):
        return interp1d(self.time, self.gamma, fill_value="extrapolate")(x)

    def delta_f(self, x):
        return interp1d(self.time, self.delta, fill_value="extrapolate")(x)

    def S_f(self, x):
        return interp1d(self.time, self.S, fill_value="extrapolate")(x)

    def I_f(self, x):
        return interp1d(self.time, self.I, fill_value="extrapolate")(x)

    def R_f(self, x):
        return interp1d(self.time, self.R, fill_value="extrapolate")(x)

    def D_f(self, x):
        return interp1d(self.time, self.D, fill_value="extrapolate")(x)

    def g_f(self, x):
        return np.matmul(
            self.C.T,
            np.matmul(self.C, np.array([self.S_f(x), self.I_f(x), self.R_f(x), self.D_f(x)])) - self.e_f(x)
        )

    def get_dfdq(self, x):
        return np.array([
            [
                - self.beta_f(x) * self.I_f(x) * (self.I_f(x) + self.R_f(x) + self.D_f(x)) / ((self.S_f(x) + self.I_f(x) + self.R_f(x) + self.D_f(x)) ** 2),
                - self.beta_f(x) * self.S_f(x) * (self.S_f(x) + self.R_f(x) + self.D_f(x)) / ((self.S_f(x) + self.I_f(x) + self.R_f(x) + self.D_f(x)) ** 2),
                self.beta_f(x) * self.I_f(x) * self.S_f(x) / ((self.S_f(x) + self.I_f(x) + self.R_f(x) + self.D_f(x)) ** 2),
                self.beta_f(x) * self.I_f(x) * self.S_f(x) / ((self.S_f(x) + self.I_f(x) + self.R_f(x) + self.D_f(x)) ** 2)
            ],
            [
                self.beta_f(x) * self.I_f(x) * (self.I_f(x) + self.R_f(x) + self.D_f(x)) / ((self.S_f(x) + self.I_f(x) + self.R_f(x) + self.D_f(x)) ** 2),
                self.beta_f(x) * self.S_f(x) * (self.S_f(x) + self.R_f(x) + self.D_f(x)) / ((self.S_f(x) + self.I_f(x) + self.R_f(x) + self.D_f(x)) ** 2) - (
                            self.gamma_f(x) + self.delta_f(x)),
                - self.beta_f(x) * self.I_f(x) * self.S_f(x) / ((self.S_f(x) + self.I_f(x) + self.R_f(x) + self.D_f(x)) ** 2),
                - self.beta_f(x) * self.I_f(x) * self.S_f(x) / ((self.S_f(x) + self.I_f(x) + self.R_f(x) + self.D_f(x)) ** 2)
            ],
            [
                0,
                self.gamma_f(x),
                0,
                0
            ],
            [
                0,
                self.delta_f(x),
                0,
                0
            ]
        ])

    def get_norm(self):
        n1 = np.dot(self.C, self.q) - np.array([self.C_e, self.D_e])
        d1 = np.array([self.C_e, self.D_e])
        return np.linalg.norm(n1) / np.linalg.norm(d1)

    def solve_state_equation(self):
        def state_equation(t, q, beta_f, gamma_f, delta_f):
            S, I, R, D = q
            dydt = [
                - beta_f(t) * I * S / (S + I + R + D),
                beta_f(t) * I * S / (S + I + R + D) - (gamma_f(t) + delta_f(t)) * I,
                gamma_f(t) * I,
                delta_f(t) * I
            ]
            return dydt

        self.q = self.S, self.I, self.R, self.D = solve_ivp(
            fun=state_equation,
            t_span=[self.t0, self.tf],
            y0=self.q0,
            t_eval=self.time,
            vectorized=True,
            args=(self.beta_f, self.gamma_f, self.delta_f)
        ).y

    def solve_adjoint_equation(self):
        def adjoint_equation(t, qs, get_dfdq, g_f):
            dfdq = get_dfdq(t)
            g0, g1, g2, g3 = g_f(t)
            dydt = [
                - (dfdq[0, 0] * qs[0] + dfdq[1, 0] * qs[1] + dfdq[2, 0] * qs[2] + dfdq[3, 0] * qs[3]) - g0,
                - (dfdq[0, 1] * qs[0] + dfdq[1, 1] * qs[1] + dfdq[2, 1] * qs[2] + dfdq[3, 1] * qs[3]) - g1,
                - (dfdq[0, 2] * qs[0] + dfdq[1, 2] * qs[1] + dfdq[2, 2] * qs[2] + dfdq[3, 2] * qs[3]) - g2,
                - (dfdq[0, 3] * qs[0] + dfdq[1, 3] * qs[1] + dfdq[2, 3] * qs[2] + dfdq[3, 3] * qs[3]) - g3,
            ]

            return dydt

        qs = solve_ivp(
            fun=adjoint_equation,
            t_span=[self.tf, self.t0],
            y0=[0, 0, 0, 0],
            t_eval=self.time[-1::-1],
            vectorized=True,
            args=(self.get_dfdq, self.g_f)
        ).y
        self.qs = [v[-1::-1] for v in qs]

    def get_derivatives(self):
        dfda = [
            [- self.I * self.S / (self.S + self.I + self.R + self.D), 0, 0],
            [self.I * self.S / (self.S + self.I + self.R + self.D), - self.I, - self.I],
            [0, self.I, 0],
            [0, 0, self.I]
        ]

        self.djdbeta = self.qs[0] * dfda[0][0] + self.qs[1] * dfda[1][0] + self.qs[2] * dfda[2][0] + self.qs[3] * dfda[3][0]
        self.djdgamma = self.qs[0] * dfda[0][1] + self.qs[1] * dfda[1][1] + self.qs[2] * dfda[2][1] + self.qs[3] * dfda[3][1]
        self.djddelta = self.qs[0] * dfda[0][2] + self.qs[1] * dfda[1][2] + self.qs[2] * dfda[2][2] + self.qs[3] * dfda[3][2]

    def get_derivative_newton(self, eps, idx):
        dx = eps
        self.beta[idx] -= eps
        self.solve_state_equation()
        self.beta[idx] += eps
        j_prev = self.get_j()
        self.beta[idx] += eps
        self.solve_state_equation()
        self.beta[idx] -= eps
        j_post = self.get_j()
        print('dj/dbeta = {:.5e}'.format((j_post - j_prev) / (2 * dx)))

    def get_j(self):
        e = np.matmul(self.C, self.q) - np.array([self.C_e, self.D_e])
        E = [np.matmul(et.T, et) for et in e.T]
        j = E[0] / 2 + E[-1] / 2 + sum(E[1: -1])
        return j / 2

    def update_alpha(self, epsilon):
        self.get_derivatives()
        self.beta -= self.djdbeta * epsilon
        self.gamma -= self.djdgamma * epsilon
        self.delta -= self.djddelta * epsilon

        self.beta = [v if v > 0 else 0 for v in self.beta]
        self.gamma = [v if v > 0 else 0 for v in self.gamma]
        self.delta = [v if v > 0 else 0 for v in self.delta]


if __name__ == '__main__':
    # get data and initial settings
    china_data = pd.read_csv('input/ChinaExample.csv').iloc[:65]
    q_init = [1.4e9, 501, 30, 17]
    alpha_init = [0.18, 0.11, 0.015]

    # run the optimization by simply using gradient descent
    da = DataAssimilation(china_data, q_init, alpha_init)
    da.solve_state_equation()
    print('Initial value:', da.get_norm())

    history_ = []
    max_iter = 2000
    iter_ = 0
    eps = 0.8 * 1e-13
    ct_start = pc()
    while iter_ < max_iter:
        da.solve_adjoint_equation()
        da.update_alpha(eps)
        da.solve_state_equation()
        history_.append(da.get_norm())
        if iter_ % 100 == 0:
            print(iter_, history_[-1])
        iter_ += 1

    print('\nTotal minutes of the computation: {:.0f}'.format((pc() - ct_start) / 60))
    _, ax = plt.subplots(1, 1)
    ax.plot(da.time, da.beta, color='r')
    ax.plot(da.time, da.gamma, color='g')
    ax.plot(da.time, da.delta, color='b')

    _, ax = plt.subplots(1, 1, figsize=(15, 7))
    ax.plot(da.time, da.I, color='b')    # infected
    ax.plot(da.time, da.R, color='g')    # Recovery
    ax.plot(da.time, da.D, color='r')    # Dead
    ax.plot(da.time, da.C_e, color='y')
    ax.plot(da.time, da.D_e, color='orange')
    ax.plot(da.time, da.I + da.R + da.D, color='k')

    pd.DataFrame(
        np.array([china_data['Date'].tolist(), da.S, da.I, da.R, da.D, da.beta, da.gamma, da.delta, da.C_e, da.D_e]).T,
        columns=['Date', 'S', 'I', 'R', 'D', 'beta', 'gamma', 'delta', 'C_e', 'D_e']
    ).to_csv('output/SIRD.csv', index=False)
    pd.DataFrame(history_, columns=['J']).to_csv('output/history.csv', index=False)
