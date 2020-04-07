# coding: utf-8
"""
@author: nelson zhao
@date:   2020.04.05
"""
import numpy as np


class AliasSampling(object):
    def __init__(self):
        self.distribution = None
        self.events = None
        self.n_events = None
        self._idx_to_event = None
        self._event_to_idx = None
        self.normalized = None
        self.accept_prob = None
        self.alias = None

    def _make_events(self):
        self._idx_to_event = {idx: event for idx, event in enumerate(self.events)}
        self._event_to_idx = {event: idx for idx, event in self._idx_to_event.items()}

    def fit(self, x, events=None, normalized=True):
        """
        Fit given distribution
        :param x: array or list of shape [n_samples]
        :param events: array or list of shape [n_samples]
        :param normalized: if the sum of elements in x is equal to 1
        :return: The object of AliasSampling
        """
        x = np.array(x)
        if not normalized:
            x = x / np.sum(x)

        if events:
            assert len(x) == len(events), IndentationError("Error: The dimensions of x and events is not consistent.")
            self.events = events
            self._make_events()

        self.n_events = len(self.events)
        self.distribution = x
        self.normalized = normalized
        self.accept_prob, self.alias = self._build_alias_table()

        return self

    def sample(self):
        """
        Sample from given distribution
        :return: event name
        """
        assert self.distribution is not None, ValueError("Value Error: distribution is None.")
        idx = np.random.randint(0, len(self.distribution))
        rd = np.random.random()

        if rd < self.accept_prob[idx]:
            return self._idx_to_event[idx]
        else:
            return self._idx_to_event[self.alias[idx]]

    def _build_alias_table(self):
        assert self.n_events > 0, ValueError("The events should be larger than 0.")

        distribution_norm = np.array(self.distribution) * self.n_events

        q_h = []
        q_l = []
        for idx, prob in enumerate(distribution_norm):
            if prob >= 1:
                q_h.append(idx)
            else:
                q_l.append(idx)

        accept_prob = [0] * self.n_events
        alias = [0] * self.n_events

        while q_h and q_l:
            h_idx = q_h.pop()
            l_idx = q_l.pop()

            accept_prob[l_idx] = distribution_norm[l_idx]
            alias[l_idx] = h_idx

            distribution_norm[h_idx] = distribution_norm[h_idx] - (1-distribution_norm[l_idx])

            if distribution_norm[h_idx] < 1:
                q_l.append(h_idx)
            else:
                q_h.append(h_idx)

        while q_h:
            h_idx = q_h.pop()
            accept_prob[h_idx] = 1

        while q_l:
            l_idx = q_l.pop()
            accept_prob[l_idx] = 1

        return accept_prob, alias

    def alias_table(self):
        return self.accept_prob, self.alias

    def distribution(self):
        return self.distribution


if __name__ == "__main__":
    events = ["A"]
    distribution = [5]

    N = 10000
    alias_sampling = AliasSampling()
    o = alias_sampling.fit(distribution, events, False)

    ret = {k: 0 for k in events}
    for i in range(N):
        e = alias_sampling.sample()
        ret[e] += 1

    print(ret)
