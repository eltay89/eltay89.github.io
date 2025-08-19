---
title: 'Understanding Linear Regression'
date: 2025-08-19
permalink: /posts/2025/08/understanding-linear-regression/
tags:
  - ai
  - algorithms
  - computer-science
  - beginners
---
# **Understanding Linear Regression: How to Predict $b_0$ and $b_1$**

Linear regression is one of the simplest and most important techniques in machine learning. It’s a type of **supervised learning** used to predict **continuous values** based on input features.

---

### **1. The Linear Regression Model**

The model assumes a linear relationship between input $x$ and output $y$:

$$
y = b_0 + b_1 x
$$

Where:

* $y$ = predicted value
* $x$ = input feature
* $b_0$ = intercept (where the line crosses the y-axis)
* $b_1$ = slope (how much y changes when x changes by 1)

Our goal is to find the **best values** of $b_0$ and $b_1$ to fit the data.

---

### **2. The Idea of Best Fit**

We define “best fit” as the line that **minimizes the difference between predicted and actual values**.

The **error** for each data point is:

$$
e_i = y_i - \hat{y}_i = y_i - (b_0 + b_1 x_i)
$$

The total error is measured using **sum of squared errors (SSE):**

$$
SSE = \sum_{i=1}^n (y_i - (b_0 + b_1 x_i))^2
$$

We want the line that **minimizes SSE**.

---

### **3. Finding $b_0$ and $b_1$ Mathematically**

We use **calculus** to minimize SSE:

1. Take derivatives of SSE with respect to $b_0$ and $b_1$.
2. Set derivatives to zero → gives **normal equations**.

#### **Derivative with respect to $b_0$:**

$$
\frac{\partial SSE}{\partial b_0} = -2 \sum (y_i - b_0 - b_1 x_i) = 0 \implies \sum (y_i - b_0 - b_1 x_i) = 0
$$

#### **Derivative with respect to $b_1$:**

$$
\frac{\partial SSE}{\partial b_1} = -2 \sum x_i (y_i - b_0 - b_1 x_i) = 0 \implies \sum x_i (y_i - b_0 - b_1 x_i) = 0
$$

Solve these two equations together to get:

$$
b_1 = \frac{\sum (x_i - \bar{x})(y_i - \bar{y})}{\sum (x_i - \bar{x})^2}, \quad b_0 = \bar{y} - b_1 \bar{x}
$$

Where $\bar{x}$ and $\bar{y}$ are the averages of $x_i$ and $y_i$.

**Intuition:**

* $b_1$ = slope → measures how y changes with x (based on covariance/variance)
* $b_0$ = intercept → ensures the line goes through the center of the data

---

### **4. Implementing in Python**

```python
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# Example data
X = [[1], [2], [3], [4], [5]]
y = [2, 4, 5, 4, 5]

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Train model
model = LinearRegression()
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Evaluate
print("MSE:", mean_squared_error(y_test, y_pred))
print("R²:", r2_score(y_test, y_pred))
```

Python’s `LinearRegression` automatically calculates $b_0$ and $b_1$ using the formulas we derived.

---

### **5. Summary**

1. Linear regression predicts a continuous output using a straight line.
2. The line is defined as $y = b_0 + b_1 x$.
3. Best fit is found by minimizing sum of squared errors (SSE).
4. Using calculus, we derive formulas for $b_0$ and $b_1$.
5. In practice, Python libraries like scikit-learn handle the calculation automatically.

Linear regression is the foundation of many ML models, so understanding **how $b_0$ and $b_1$ are derived** is key before moving to more complex algorithms like neural networks.

---

If you want, I can **also create a small diagram showing the line fitting points visually**, which makes the math much easier to grasp.

Do you want me to make that diagram?
