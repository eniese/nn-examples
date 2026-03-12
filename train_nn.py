#!/usr/bin/env python3
import math
import random
from typing import List, Tuple

# ------------------------------
# Activation functions
# ------------------------------
def relu(x: float) -> float:
    return max(0.0, x)

def relu_deriv(x: float) -> float:
    return 1.0 if x > 0 else 0.0

def softmax(z: List[float]) -> List[float]:
    max_val = max(z)
    exp_vals = [math.exp(zi - max_val) for zi in z]
    total = sum(exp_vals)
    return [e / total for e in exp_vals] if total > 0 else [0.5, 0.5]

# ------------------------------
# Data – simple toy dataset (you can expand this)
# ------------------------------
# Format: (number_of_legs, target_class)  0 = dog, 1 = spider
training_data: List[Tuple[float, int]] = [
    (0, 0), (1, 0), (2, 0), (2.5, 0), (3, 0), (4, 0), (4.2, 0),
    (4.8, 0), (5, 1), (5.2, 1), (6, 1), (7, 1), (8, 1), (8.5, 1),
    (9, 1), (10, 1), (11, 1), (12, 1)
]

# Convert to one-hot
def to_onehot(label: int) -> List[float]:
    return [1.0, 0.0] if label == 0 else [0.0, 1.0]

# ------------------------------
# Model parameters
# ------------------------------
W_ih = [random.gauss(0, 0.3) for _ in range(3)]          # 3 weights
b_h  = [random.gauss(0, 0.1) for _ in range(3)]          # 3 biases

W_ho = [[random.gauss(0, 0.2) for _ in range(3)] for _ in range(2)]  # 2×3
b_o  = [random.gauss(0, 0.1) for _ in range(2)]                      # 2 biases


# ------------------------------
# Forward pass
# ------------------------------
def forward(x: float) -> Tuple[List[float], List[float], List[float]]:
    # Hidden layer
    z_h = [W_ih[i] * x + b_h[i] for i in range(3)]
    a_h = [relu(z) for z in z_h]

    # Output layer
    z_o = [
        sum(W_ho[0][j] * a_h[j] for j in range(3)) + b_o[0],
        sum(W_ho[1][j] * a_h[j] for j in range(3)) + b_o[1]
    ]

    probs = softmax(z_o)
    return probs, a_h, z_h


# ------------------------------
# Cross-entropy loss
# ------------------------------
def cross_entropy_loss(probs: List[float], target: List[float]) -> float:
    return -sum(t * math.log(p + 1e-12) for t, p in zip(target, probs))


# ------------------------------
# Train one step (full-batch gradient descent)
# ------------------------------
def train_step(learning_rate: float = 0.02):
    global W_ih, b_h, W_ho, b_o

    dW_ih = [0.0] * 3
    db_h  = [0.0] * 3
    dW_ho = [[0.0] * 3 for _ in range(2)]
    db_o  = [0.0] * 2

    total_loss = 0.0

    for x, label in training_data:
        target = to_onehot(label)
        probs, a_h, z_h = forward(x)
        loss = cross_entropy_loss(probs, target)
        total_loss += loss

        # Output error (δ_o = p - y)
        delta_o = [probs[i] - target[i] for i in range(2)]

        # Output layer gradients
        for i in range(2):
            db_o[i] += delta_o[i]
            for j in range(3):
                dW_ho[i][j] += delta_o[i] * a_h[j]

        # Hidden layer error
        delta_h = [0.0] * 3
        for j in range(3):
            delta_h[j] = sum(W_ho[i][j] * delta_o[i] for i in range(2))
            delta_h[j] *= relu_deriv(z_h[j])

        # Hidden layer gradients
        for j in range(3):
            db_h[j] += delta_h[j]
            dW_ih[j] += delta_h[j] * x

    # Average gradients
    n = len(training_data)
    for j in range(3):
        dW_ih[j] /= n
        db_h[j] /= n
    for i in range(2):
        db_o[i] /= n
        for j in range(3):
            dW_ho[i][j] /= n

    # Update
    for j in range(3):
        W_ih[j] -= learning_rate * dW_ih[j]
        b_h[j]  -= learning_rate * db_h[j]

    for i in range(2):
        b_o[i] -= learning_rate * db_o[i]
        for j in range(3):
            W_ho[i][j] -= learning_rate * dW_ho[i][j]

    return total_loss / n


# ------------------------------
# Evaluation / printing
# ------------------------------
def print_predictions(epoch: int):
    print(f"\nEpoch {epoch:4d}   weights & biases:")
    print(f"  W_ih = {[round(w,4) for w in W_ih]}")
    print(f"  b_h  = {[round(b,4) for b in b_h]}")
    
    # Nicely formatted W_ho
    formatted_rows = [f"[{', '.join(f'{round(w,4)}' for w in row)}]" for row in W_ho]
    print(f"  W_ho = [{', '.join(formatted_rows)}]")
    print(f"  b_o  = {[round(b,4) for b in b_o]}")
    print("\n legs │   P(dog)    │  P(spider)  │ winner")

    for legs in range(13):
        probs, _, _ = forward(float(legs))
        winner = "DOG" if probs[0] > probs[1] else "SPIDER"
        print(f"{legs:5} │ {probs[0]:11.6f} │ {probs[1]:11.6f} │ {winner}")


# ------------------------------
# Training loop
# ------------------------------
if __name__ == "__main__":
    print("Initial predictions:")
    print_predictions(0)

    LEARNING_RATE = 0.03
    EPOCHS = 1200

    for epoch in range(1, EPOCHS + 1):
        loss = train_step(LEARNING_RATE)

        if epoch % 200 == 0 or epoch == 1 or epoch == 10 or epoch == 50:
            print(f"\nEpoch {epoch:4d}   loss = {loss:.6f}")
            print_predictions(epoch)

    print("\nFinal model after training:")
    print_predictions(EPOCHS)
