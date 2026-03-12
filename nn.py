#!/usr/bin/env python3

import math

def relu(x):
    return max(0.0, x)

def softmax(x):
    max_val = max(x)
    exp_x = [math.exp(xi - max_val) for xi in x]
    total = sum(exp_x)
    return [e / total for e in exp_x] if total != 0 else [0.5, 0.5]


# Weights & biases
W_input_hidden = [-3.0, 2.2, -1.1]
b_hidden       = [13.0, -9.0, 4.5]

W_hidden_output = [
    [ 4.5, -3.8,  2.0],   # → DOG
    [-5.0,  4.0, -2.5]    # → SPIDER
]
b_output = [2.0, -3.0]


def predict_relu_softmax(legs: float):
    z_hidden = [
        W_input_hidden[0] * legs + b_hidden[0],
        W_input_hidden[1] * legs + b_hidden[1],
        W_input_hidden[2] * legs + b_hidden[2],
    ]
    a_hidden = [relu(z) for z in z_hidden]

    z_output = [
        sum(w * a for w, a in zip(W_hidden_output[0], a_hidden)) + b_output[0],
        sum(w * a for w, a in zip(W_hidden_output[1], a_hidden)) + b_output[1],
    ]

    probs = softmax(z_output)
    return probs[0], probs[1], a_hidden


print(" legs │    H1      H2      H3   │ P(dog)     │ P(spider)  │ winner ")
print("──────┼─────────────────────────┼────────────┼────────────┼────────")

for legs in range(11):
    p_dog, p_spider, h = predict_relu_softmax(legs)
    winner = "DOG" if p_dog > p_spider else "SPIDER" if p_spider > p_dog else "TIE"

    print(f"{legs:>5} │ "
          f"{h[0]:7.3f} {h[1]:7.3f} {h[2]:7.3f} │ "
          f"{p_dog:10.8f} │ "
          f"{p_spider:10.8f} │ "
          f"{winner}")
