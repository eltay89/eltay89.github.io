---
title: 'A Foundational Guide to Supervised Learning: Regression and Classification'
date: 2025-08-05
permalink: /posts/2024/05/foundational-guide-supervised-learning/
tags:
  - machine-learning
  - supervised-learning
  - regression
  - classification
  - mathematics
  - theory
  - python
---

## Introduction: Teaching a Computer to Learn

Imagine teaching a child to recognize different animals. You don't write a long list of rules like "if it has whiskers and pointy ears, it's a cat." Instead, you show them pictures. You point to a picture of a cat and say "cat." You point to a picture of a dog and say "dog." After seeing enough examples, the child learns the underlying patterns on their own.

**Supervised learning** operates on this exact principle. We provide a computer algorithm with a dataset of "examples" where we already know the correct answer. The algorithm's job is to learn the relationship between the inputs and the outputs so it can make accurate predictions when it sees new, unseen data.

This field is dominated by two fundamental types of problems, which we will explore in parallel:

1.  **Regression:** The goal is to predict a continuous numerical value.
    *   *Question:* "Based on the size of this house, how much will it cost?"
    *   *Answer:* A specific number, like `$350,000`.

2.  **Classification:** The goal is to predict a discrete category or class.
    *   *Question:* "Based on the contents of this email, is it spam or not spam?"
    *   *Answer:* A specific label, like `Spam`.

This guide will walk you through both of these worlds in detail. We will build our understanding from the ground up, starting with the core problem, defining a model, measuring its error, and finally, teaching it to improve.

---

## Part 1: Regression - The Art of Predicting Numbers

**Our Core Problem:** We are a real estate agent who wants to build a model to predict house prices. To start simply, we will use just one piece of information: the size of the house in square feet.

Our dataset looks like this:

| Size (sq. ft.), `x` | Price ($1000s), `y` |
|---------------------|---------------------|
| 2104                | 400                 |
| 1600                | 330                 |
| 2400                | 369                 |

Our goal is to create a function that takes any new size `x` and predicts a price `y`.

### Step 1: The Model's Blueprint (The Hypothesis Function)

The most basic assumption we can make is that the relationship between size and price is linear. As the size goes up, the price goes up in a predictable, straight-line fashion. In algebra, the equation for a line is `y = mx + c`. In machine learning, we use a slightly different notation which we call our **hypothesis function**, denoted as `h(x)`.

$$
h_\theta(x) = \theta_0 + \theta_1 x
$$

Let's patiently unpack every piece of this equation. It is the foundation for everything that follows.

-   `h(x)`: This is our model's prediction. You give it a house size `x`, and it gives you back a predicted price. The `h` stands for hypothesis because, at the start, this is just our "best guess" about how the world works.
-   `x`: This is our input **feature**. In this case, it's the size of the house.
-   `θ₁` (pronounced "theta-one"): This is the **slope** of the line. It's a **parameter** or **weight** that our model needs to learn. It represents the "price per square foot." For every one-unit increase in `x`, the price `h(x)` will increase by `θ₁`.
-   `θ₀` (pronounced "theta-zero"): This is the **y-intercept**. It's another **parameter** the model must learn. You can think of it as the base price of a house, representing the value of the land and basic construction before any size is added.

Our entire goal in training this model is to discover the perfect numerical values for `θ₀` and `θ₁` that result in a line that best fits our data.

```plotly
{
  "data": [
    {
      "x": [2104, 1600, 2400, 1416],
      "y": [400, 330, 369, 232],
      "mode": "markers",
      "type": "scatter",
      "name": "Our Housing Data"
    }
  ],
  "layout": {
    "title": "Our Goal: Find the Best Line Through This Data",
    "xaxis": {"title": "Size (sq. ft.)"},
    "yaxis": {"title": "Price ($1000s)"}
  }
}
```

### Step 2: Measuring How Wrong We Are (The Cost Function)

Before we can find the "best" line, we need a way to measure how "bad" a particular line is. For example, is the line `Price = 100 + 0.1 * Size` better or worse than `Price = 50 + 0.2 * Size`?

To do this, we use a **cost function**, often written as `J(θ)`. This function takes our current parameters (`θ₀` and `θ₁`) and computes a single number that represents the total error of our model. A high cost means a bad model; a low cost means a good model.

For regression, the standard cost function is the **Mean Squared Error (MSE)**. Let's build it conceptually.

1.  **Find the error for a single house.** For any one house in our dataset, the error is the vertical distance between the actual price (`y`) and the price our line predicted (`h(x)`).

    ```plotly
    {
      "data": [
        { "x": [1600, 2104, 2400], "y": [330, 400, 369], "mode": "markers", "type": "scatter", "name": "Data Points" },
        { "x": [1000, 3000], "y": [150, 550], "mode": "lines", "type": "scatter", "name": "A Possible Model (h(x))" },
        { "x": [2104, 2104], "y": [400, 410.8], "mode": "lines", "type": "scatter", "name": "Error for one point", "line": {"color": "red", "width": 3, "dash": "dash"} }
      ],
      "layout": { "title": "The Error is the Vertical Distance", "xaxis": {"title": "Size"}, "yaxis": {"title": "Price"} }
    }
    ```

2.  **Square the error.** The difference `h(x) - y` can be positive or negative. If we just add them up, they might cancel each other out. To solve this, we square each error: `(h(x) - y)²`. This has two benefits: all errors become positive, and it heavily penalizes larger errors. A mistake of 10 is squared to 100, while a mistake of 2 is only squared to 4.

3.  **Sum all the squared errors and take the average.** We do this for all `m` houses in our dataset. This gives us the "Mean" part of Mean Squared Error.

This logic gives us our final equation for the cost function:

$$
J(\theta_0, \theta_1) = \frac{1}{2m} \sum_{i=1}^{m} (h_\theta(x^{(i)}) - y^{(i)})^2
$$

Let's re-examine every symbol here:
-   `J(θ₀, θ₁)`: The cost, which is a function of our chosen parameters.
-   `m`: The total number of examples (houses) in our dataset.
-   `Σ`: The summation symbol, which means "add everything up."
-   `i=1` to `m`: We are summing from the first house (`i=1`) to the last house (`m`).
-   `x⁽ⁱ⁾`: The size of the `i`-th house in our dataset.
-   `y⁽ⁱ⁾`: The actual price of the `i`-th house.
-   `h(x⁽ⁱ⁾)`: Our model's predicted price for the `i`-th house.
-   The `1/2` is included for mathematical convenience, as it simplifies the calculus in the next step. It doesn't change the location of the minimum cost.

Our goal is now crystal clear: **find the values of `θ₀` and `θ₁` that make `J(θ₀, θ₁)` as small as possible.**

### Step 3: The Process of Learning (Optimization via Gradient Descent)

We have a cost function that we can imagine as a 3D bowl. The two horizontal axes are `θ₀` and `θ₁`, and the vertical axis is the cost `J`. Our goal is to find the coordinates of the absolute bottom of this bowl.

The algorithm to do this is called **Gradient Descent**.

Imagine you are standing on the side of that foggy bowl. You want to get to the bottom. You can't see it directly, but you can feel the slope of the ground beneath your feet. The most logical thing to do is to identify the steepest downhill direction and take a small step. Then, from your new position, you repeat the process. If you keep taking small steps in the steepest downhill direction, you will eventually arrive at the bottom.

That is exactly what Gradient Descent does. It's an iterative algorithm that adjusts `θ₀` and `θ₁` step-by-step to lower the cost `J`.

The "direction of the slope" is given by the **partial derivative** of the cost function, written as `∂/∂θⱼ J(...)`. This is a concept from calculus that tells us how the cost `J` changes if we make a tiny nudge to a single parameter `θⱼ`.

The update rule for Gradient Descent is:
$$
\theta_j := \theta_j - \alpha \frac{\partial}{\partial \theta_j} J(\theta_0, \theta_1)
$$
-   `:=` means we are updating the value of `θⱼ`.
-   `α` (alpha) is the **learning rate**. It's a small number you choose (e.g., 0.01) that controls how big of a step you take. Too big, and you might overshoot the bottom. Too small, and it will take forever to get there.
-   The `-` sign is crucial. The derivative gives us the *uphill* direction. By subtracting it, we ensure we are always moving *downhill*.

When we perform the calculus (which you don't need to do by hand), we find the specific derivatives for our MSE cost function. This gives us the final update rules that the computer will execute thousands of times:

Repeat {
<br>
&nbsp;&nbsp;&nbsp;&nbsp;\\(\theta_0 := \theta_0 - \alpha \frac{1}{m} \sum_{i=1}^{m} (h_\theta(x^{(i)}) - y^{(i)})\\)
<br>
&nbsp;&nbsp;&nbsp;&nbsp;\\(\theta_1 := \theta_1 - \alpha \frac{1}{m} \sum_{i=1}^{m} (h_\theta(x^{(i)}) - y^{(i)})x^{(i)}\\)
<br>
}

With each repetition, the values of `θ₀` and `θ₁` get closer to the optimal values that define the best-fit line.

### Step 4: Connecting Theory to Code and Evaluating the Result

The beautiful part is that libraries like `scikit-learn` handle all of this complex math for us. But now we understand exactly what is happening inside.

```python
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
import numpy as np

# Our data: [Size], [Price in $1000s]
X = np.array([[1500], [2200], [1200], [1800], [2500], [1300]])
y = np.array([300, 450, 210, 370, 520, 240])

# We split the data so we can train on one part and test on another, unseen part.
# This gives us a fair evaluation of how well our model generalizes.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# --- This is where the theory comes to life ---

# 1. The Hypothesis: The LinearRegression() object is our h(x) = θ₀ + θ₁x.
# It's a blueprint waiting for its parameters.
model = LinearRegression()

# 2 & 3. Cost Function and Optimization: The .fit() method is the engine.
# It runs an advanced version of Gradient Descent to minimize the MSE cost function (J(θ))
# and finds the optimal values for θ₀ and θ₁.
model.fit(X_train, y_train)

# After fitting, the model has learned the parameters.
# model.intercept_ is our learned θ₀
# model.coef_ is our learned θ₁
print(f"Learned Parameters: θ₀ = {model.intercept_:.2f}, θ₁ = {model.coef_[0]:.2f}")

# --- Now, we evaluate our trained model ---
predictions = model.predict(X_test)

# Evaluation Metric 1: Mean Absolute Error (MAE)
# This tells us, on average, how far off our predictions are in the original units.
mae = mean_absolute_error(y_test, predictions)
print(f"\nMean Absolute Error: Our model's predictions are off by an average of ${mae:.2f}k.")

# Evaluation Metric 2: R-squared (R²)
# This tells us what percentage of the variation in house prices our model is able to explain.
# A value of 1.0 is a perfect explanation.
r2 = r2_score(y_test, predictions)
print(f"R-squared: Our model explains {r2*100:.2f}% of the variance in house prices.")
```

---

## Part 2: Classification - The Art of Predicting Categories

**Our Core Problem:** We are medical researchers building a model to predict if a tumor is malignant (cancerous) or benign (non-cancerous) based on its size.

Our dataset has two classes, which we label as `1` (Malignant) and `0` (Benign).

| Tumor Size, `x` | Class, `y` |
|-----------------|------------|
| 2               | 0 (Benign) |
| 3               | 0 (Benign) |
| 7               | 1 (Malignant)|
| 8               | 1 (Malignant)|

Our goal is to create a function that takes a new tumor size `x` and predicts its class (`0` or `1`).

### Step 1: The Model's Blueprint (The Logistic Hypothesis)

We face an immediate problem. Our linear regression model `h(x) = θ₀ + θ₁x` outputs numbers like `350.7` or `-50`. How can we map that to a class like `0` or `1`?

We need a function that takes any number and "squashes" it into a value between 0 and 1, so we can interpret it as a probability. The perfect function for this job is the **Sigmoid Function** (also called the Logistic Function).

$$
g(z) = \frac{1}{1 + e^{-z}}
$$

Let's look at its graph. No matter what `z` is (positive or negative), the output `g(z)` is always between 0 and 1.

```plotly
{
  "data": [
    { "x": [-10, -5, 0, 5, 10], "y": [0.00004, 0.0067, 0.5, 0.9933, 0.9999], "mode": "lines", "type": "scatter", "name": "Sigmoid Function g(z)" }
  ],
  "layout": { "title": "The Sigmoid Function: A 'Squashing' Machine", "xaxis": {"title": "Input z"}, "yaxis": {"title": "Output g(z) (Probability)"} }
}
```

Our new hypothesis for **Logistic Regression** is simple: we take our old linear model and plug it into the sigmoid function.

$$
h_\theta(x) = g(\theta_0 + \theta_1 x)
$$

The output of this new `h(x)` is now a probability. For example, if `h(x) = 0.9` for a given tumor size, our model is saying there is a 90% probability that the tumor is malignant (`y=1`). We can then set a threshold (like 0.5) to make a final decision: if `h(x) >= 0.5`, predict `1`, otherwise predict `0`.

### Step 2: A New Way to Measure Error (The Cost Function)

We can't use Mean Squared Error here. If we did, our cost function "bowl" would become wavy with many local minima, and Gradient Descent might get stuck in the wrong one.

We need a cost function designed for probabilities. This is called **Log Loss** (or Binary Cross-Entropy). Its design is incredibly clever.

Let's think about the cost for a single example:
-   **If the actual class is `y=1` (Malignant):** We want our model's prediction `h(x)` to be as close to 1 as possible. The cost for this case is defined as `−log(h(x))`. Look at the graph of `−log(z)`: if `h(x)` is 1, `log(1)` is 0, so the cost is 0. If `h(x)` is 0 (a terrible prediction), `log(0)` approaches infinity, so the cost is huge. It perfectly penalizes being wrong.
-   **If the actual class is `y=0` (Benign):** We want `h(x)` to be close to 0. The cost is defined as `−log(1−h(x))`. If `h(x)` is 0, the cost is `−log(1)`, which is 0. If `h(x)` is 1 (a terrible prediction), the cost is `−log(0)`, which is huge.

These two ideas are combined into one single, elegant equation for the cost function `J(θ)`:
$$
J(\theta) = -\frac{1}{m} \sum_{i=1}^{m} [y^{(i)}\log(h_\theta(x^{(i)})) + (1 - y^{(i)})\log(1 - h_\theta(x^{(i)}))]
$$
This equation looks intimidating, but it's just a trick to implement both cases at once. When `y=1`, the second part of the sum becomes zero. When `y=0`, the first part becomes zero.

### Step 3: The Learning Process (Optimization)

The good news is that our learning process is exactly the same. We use **Gradient Descent** to find the `θ` values that minimize this new Log Loss cost function. The algorithm `θⱼ := θⱼ - α * (derivative)` is identical in principle; only the derivative calculation changes because it's based on a different `J(θ)`.

### Step 4: Connecting Theory to Code and Evaluating the Result

For classification, we need different metrics to evaluate our model's performance. Accuracy alone can be misleading, especially if one class is much rarer than the other.

```python
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Our data: [Tumor Size], [Is_Malignant (1=Yes, 0=No)]
X = np.array([[2], [3], [4], [5], [6], [7], [8], [9]])
y = np.array([0, 0, 0, 0, 1, 1, 1, 1])

# --- This is where the theory comes to life ---

# 1. The Hypothesis: The LogisticRegression() object is our h(x) = g(θ₀ + θ₁x).
model = LogisticRegression()

# 2 & 3. Cost Function and Optimization: The .fit() method minimizes the Log Loss cost function
# using an optimizer to find the best θ₀ and θ₁.
model.fit(X, y)

# --- Now, we evaluate our trained model ---
predictions = model.predict(X)

# Evaluation Metric 1: Accuracy
# What percentage of our predictions were correct?
accuracy = accuracy_score(y, predictions)
print(f"Accuracy: {accuracy*100:.2f}% of our predictions were correct.")

# Evaluation Metric 2: Precision
# Of all the times we predicted 'Malignant', how often were we right?
# This is important for avoiding false alarms and unnecessary treatments.
precision = precision_score(y, predictions)
print(f"Precision: {precision*100:.2f}%.")

# Evaluation Metric 3: Recall
# Of all the tumors that were *actually* Malignant, how many did our model successfully identify?
# This is important for not missing dangerous cases.
recall = recall_score(y, predictions)
print(f"Recall: {recall*100:.2f}%.")

# The Confusion Matrix gives us the full picture
cm = confusion_matrix(y, predictions)
print("\nConfusion Matrix:")
print("TN | FP")
print("FN | TP")
print(cm)
```

## Conclusion: Two Problems, One Unified Approach

You have now walked through the complete foundational process for the two pillars of supervised learning. While the specifics of the models and their error measurements differ, the core philosophy is the same:

1.  **Start with a problem** and choose a model structure (**Hypothesis**) that fits it.
2.  Define a mathematical way to measure the model's total error (**Cost Function**).
3.  Use an **Optimization Algorithm** like Gradient Descent to systematically tune the model's parameters to minimize that error.
4.  **Evaluate** the final model using metrics that are meaningful for the specific problem you are trying to solve.

This fundamental loop of Hypothesis -> Cost -> Optimization is the intellectual engine that drives a vast portion of modern artificial intelligence. Understanding it deeply provides you with the foundation to explore any other supervised learning algorithm you may encounter.
