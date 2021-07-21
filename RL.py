# Group 27: Joao Galinho (87667) & Filipe Henriques (87653)
import numpy as np

from tempfile import TemporaryFile
outfile = TemporaryFile()


class finiteMDP:
    def __init__(self, nS, nA, gamma, P=None, R=None, absorv=None, alpha=0.8):
        self.nS = nS
        self.nA = nA
        self.gamma = gamma
        self.Q = np.zeros((self.nS, self.nA))
        self.P = [] if P is None else P
        self.R = [] if R is None else R
        self.absorv = [] if absorv is None else absorv
        self.alpha = alpha

    def runPolicy(self, n, x0, poltype='exploitation', polpar=None):
        traj = np.zeros((n, 4))
        x = x0
        J = 0
        for ii in range(0, n):
            a = self.policy(x, poltype, polpar)
            r = self.R[x, a]
            y = np.nonzero(np.random.multinomial(1, self.P[x, a, :]))[0][0]
            traj[ii, :] = np.array([x, a, y, r])
            J = J + r * self.gamma**ii
            if self.absorv[x]:
                y = x0
            x = y
        return J, traj

    def VI(self):
        nQ = np.zeros((self.nS, self.nA))
        while True:
            self.V = np.max(self.Q, axis=1)
            for a in range(0, self.nA):
                nQ[:, a] = self.R[:, a] + self.gamma * np.dot(self.P[:, a, :], self.V)
            err = np.linalg.norm(self.Q - nQ)
            self.Q = np.copy(nQ)
            if err<1e-7:
                break
        self.V = np.max(self.Q, axis=1)
        self.Pol = np.argmax(self.Q, axis=1)
        return self.Q, self.Q2pol(self.Q)

    def traces2Q(self, trace):
        while True:
            oQ = np.copy(self.Q)
            for traj in trace:
                initState, action, finalState, reward = int(traj[0]), int(traj[1]), int(traj[2]), int(traj[3])
                self.Q[initState, action] += \
                    self.alpha * (reward + self.gamma * np.amax(self.Q[finalState]) - self.Q[initState, action])
            err = np.linalg.norm(self.Q - oQ)
            if err<1e-7:
                break
        return self.Q

    def policy(self, x, poltype='exploration', polpar=None):
        softmax = self.VI()[1]
        if poltype == 'exploration':
            return np.argmax(softmax[x])
        if poltype == 'exploitation':
            return np.argmax(polpar[x]) if polpar is not None else self.Pol[x]
        return

    def Q2pol(self, Q, eta=5):
        return np.exp(eta * Q) / np.dot(np.exp(eta * Q), np.ones((len(Q[0]), len(Q[0]))))
