# q_learning.py
import math
import random
def get_best_action(Q, state, actions):
    """
    Return the action with the lowest Q-value (cost minimization).
    If unseen state, return a random or default action.
    """
    best_a, best_val = None, float("inf")
    for a in actions:
        val = Q.get((state, a), float("inf"))
        if val < best_val:
            best_val, best_a = val, a
    return best_a if best_a is not None else actions[0]


def update(Q, state, action, cost, next_state, alpha, gamma, actions):
    """
    Q-learning update for cost minimization (inf/nan safe).
    Q(s,a) ← Q(s,a) + α * [(cost + γ * min_a' Q(s',a')) − Q(s,a)]
    """
    if not math.isfinite(cost):
        return  # skip invalid updates

    old_value = Q.get((state, action), 0.0)
    next_min = min(Q.get((next_state, a), float('inf')) for a in actions)
    if not math.isfinite(next_min):
        next_min = 0.0

    target = cost + gamma * next_min
    if not math.isfinite(target):
        return

    Q[(state, action)] = old_value + alpha * (target - old_value)


def decay_epsilon(eps, eps_min, decay):
    """Exponential epsilon decay (never below eps_min)."""
    new_eps = eps * decay
    return eps_min if new_eps < eps_min else new_eps
