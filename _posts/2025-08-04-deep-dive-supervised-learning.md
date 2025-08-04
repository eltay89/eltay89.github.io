---
title: 'Supervised Learning: A Step-by-Step Mathematical Guide'
date: 2025-08-04
permalink: /posts/2024/05/supervised-learning-math-guide/
tags:
  - machine-learning
  - supervised-learning
  - regression
  - mathematics
  - theory
  - gradient-descent
---

Supervised learning is the process of creating a predictive model by learning from a dataset of labeled examples. Our goal is to learn a function, let's call it `f`, that can take a new set of input features, `X`, and accurately predict the corresponding output label, `Y`.

This guide will walk you through the "how" and "why" of this process, focusing on the fundamental algorithm: **Linear Regression**. We will build it step-by-step, from the initial idea to the final optimization equations.

### The Scenario: Predicting House Prices

To make this concrete, let's use a simple goal: **predicting the price of a house based on its size (square footage).**

We have a dataset of houses where we know both the size and the price it sold for.

| Size (sq. ft.), `x` | Price ($1000s), `y` |
|---------------------|---------------------|
| 2104                | 400                 |
| 1600                | 330                 |
| 2400                | 369                 |
| 1416                | 232                 |
| ...                 | ...                 |

Our task is to create a model that, given a new house size (e.g., 1800 sq. ft.), can predict its price.

---

### Step 1: Define the Model's Form (The Hypothesis)

The simplest assumption we can make is that the relationship between size and price is linear. In other words, we want to draw a straight line that best fits our data points.

The mathematical equation for a straight line is `y = mx + c`. In machine learning, we write this slightly differently, which we call our **hypothesis function**, `h(x)`:

$$
h_\theta(x) = \theta_0 + \theta_1 x
$$

Let's dissect every part of this:
-   `h(x)`: The hypothesis, which is our model's prediction for a given input `x`.
-   `x`: The input feature (the size of the house).
-   `θ₀` (theta-zero): This is the y-intercept (`c` in the original equation). It represents the base price of a house, even with zero square footage (a theoretical starting point).
-   `θ₁` (theta-one): This is the slope (`m` in the original equation). It represents the weight of our feature—how much the price increases for each additional square foot.

Our goal is no longer abstract. It is now concrete: **We must find the values of `θ₀` and `θ₁` that define the "best" possible line.**

```plotly
{
  "data": [
    {
      "x": [2104, 1600, 2400, 1416, 3000],
      "y": [400, 330, 369, 232, 540],
      "mode": "markers",
      "type": "scatter",
      "name": "Actual Data"
    },
    {
      "x": [1000, 3200],
      "y": [150, 550],
      "mode": "lines",
      "type": "scatter",
      "name": "A Possible Model (Line)"
    }
  ],
  "layout": {
    "title": "Finding the Best Line Through the Data",
    "xaxis": {"title": "Size (sq. ft.)"},
    "yaxis": {"title": "Price ($1000s)"}
  }
}
```

### Step 2: Define "Best" (The Cost Function)

How do we measure if one line is better than another? We measure its **error**. For any single data point, the error is the vertical distance between the actual price (`y`) and the price our line predicted (`h(x)`).

We need to aggregate this error across all our data points into a single number. This number is our **cost**. A low cost means a good model; a high cost means a bad model.

Here's how we build the cost function, called **Mean Squared Error (MSE)**, piece by piece:

1.  **Error for one point:** For the `i`-th house in our dataset, the error is `(h(x⁽ⁱ⁾) - y⁽ⁱ⁾)`.
2.  **Make errors positive:** Some errors will be positive (prediction too high) and some negative (prediction too low). If we just add them, they could cancel out. To fix this, we square the error: `(h(x⁽ⁱ⁾) - y⁽ⁱ⁾)²`. This also has the nice property of penalizing larger errors much more than smaller ones.
3.  **Sum all errors:** We do this for every one of our `m` data points and add them up:
    $$ \sum_{i=1}^{m} (h_\theta(x^{(i)}) - y^{(i)})^2 $$
4.  **Take the average:** To make the cost independent of the dataset size, we take the average by dividing by `m`. We also divide by `2` as a mathematical convenience that will simplify our calculus in the next step.

This gives us our final cost function, `J(θ₀, θ₁)`:

$$
J(\theta_0, \theta_1) = \frac{1}{2m} \sum_{i=1}^{m} (h_\theta(x^{(i)}) - y^{(i)})^2
$$

Our goal is now even more specific: **Find the values of `θ₀` and `θ₁` that minimize `J(θ₀, θ₁)`.**

### Step 3: Find the Minimum Cost (The Optimization Algorithm)

We need an algorithm that can find the bottom of our cost function "bowl". The most common method is **Gradient Descent**.

**The Intuition:** Imagine you are standing on a foggy hill (the cost function) and you want to get to the very bottom (the minimum cost). You can't see the bottom, but you can feel the slope of the ground right where you are. The best strategy is to take a small step in the steepest downhill direction, and repeat until you can't go any lower.

**The Math:**
"The steepest downhill direction" is the negative of the **gradient**. The gradient is just a collection of all the partial derivatives of the cost function. We need to figure out how the cost `J` changes as we slightly change `θ₀` and `θ₁`.

The general update rule for Gradient Descent is:

Repeat until convergence {
<br>
&nbsp;&nbsp;&nbsp;&nbsp;\\(\theta_j := \theta_j - \alpha \frac{\partial}{\partial \theta_j} J(\theta_0, \theta_1)\\)
<br>
}

-   `:=` is the assignment operator.
-   `α` (alpha) is the **learning rate**. It's a small number (e.g., 0.01) that controls how big of a step we take.
-   `∂/∂θⱼ J(...)` is the partial derivative—the slope of the cost function with respect to a single parameter `θⱼ`.

Now, let's calculate those derivatives for our specific cost function. This requires the chain rule from calculus.

**Derivative with respect to `θ₀`:**
$$
\frac{\partial}{\partial \theta_0} J = \frac{\partial}{\partial \theta_0} \frac{1}{2m} \sum (h_\theta(x) - y)^2 = \frac{1}{2m} \sum \frac{\partial}{\partial \theta_0} (\theta_0 + \theta_1 x - y)^2
$$
$$
= \frac{1}{2m} \sum 2(\theta_0 + \theta_1 x - y) \cdot \frac{\partial}{\partial \theta_0}(\theta_0 + \theta_1 x - y)
$$
$$
= \frac{1}{m} \sum (\theta_0 + \theta_1 x - y) \cdot 1 = \frac{1}{m} \sum (h_\theta(x) - y)
$$

**Derivative with respect to `θ₁`:**
$$
\frac{\partial}{\partial \theta_1} J = \frac{\partial}{\partial \theta_1} \frac{1}{2m} \sum (h_\theta(x) - y)^2 = \frac{1}{2m} \sum \frac{\partial}{\partial \theta_1} (\theta_0 + \theta_1 x - y)^2
$$
$$
= \frac{1}{2m} \sum 2(\theta_0 + \theta_1 x - y) \cdot \frac{\partial}{\partial \theta_1}(\theta_0 + \theta_1 x - y)
$$
$$
= \frac{1}{m} \sum (\theta_0 + \theta_1 x - y) \cdot x = \frac{1}{m} \sum (h_\theta(x) - y)x
$$

### Step 4: Putting It All Together

Now we have our final, specific update rules for Linear Regression with Gradient Descent. We initialize `θ₀` and `θ₁` to some values (e.g., 0) and then repeatedly run these updates:

Repeat {
<br>
&nbsp;&nbsp;&nbsp;&nbsp;\\(\text{temp0} := \theta_0 - \alpha \frac{1}{m} \sum_{i=1}^{m} (h_\theta(x^{(i)}) - y^{(i)})\\)
<br>
&nbsp;&nbsp;&nbsp;&nbsp;\\(\text{temp1} := \theta_1 - \alpha \frac{1}{m} \sum_{i=1}^{m} (h_\theta(x^{(i)}) - y^{(i)})x^{(i)}\\)
<br>
&nbsp;&nbsp;&nbsp;&nbsp;\\(\theta_0 := \text{temp0}\\)
<br>
&nbsp;&nbsp;&nbsp;&nbsp;\\(\theta_1 := \text{temp1}\\)
<br>
}

(We use temporary variables to ensure we update `θ₀` and `θ₁` simultaneously based on the *old* values).

This loop continues, and with each iteration, our line `h(x)` gets closer and closer to the best fit for the data. The cost `J(θ₀, θ₁)` decreases until it converges at the minimum. At that point, we have found our optimal parameters and our model is trained.
