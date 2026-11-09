import type { Phase } from '../types'

const md = String.raw

export const phase0: Phase = {
  id: 'p0',
  title: 'Math foundations',
  goal: 'Only the math machine learning actually uses: vectors, gradients, probability and the idea of minimizing an error.',
  weeks: '2–4 weeks, alongside Phase 1',
  topics: [
    {
      id: 'm-linalg',
      title: 'Linear algebra for ML',
      summary: 'Vectors, dot products and matrices: how a model turns a table of numbers into predictions.',
      guide: { label: '3Blue1Brown: Essence of Linear Algebra', url: 'https://www.3blue1brown.com/topics/linear-algebra' },
      intuition: md`A dataset is a table. Each **row** is one example (a house, a patient, an email) and each **column** is one feature (size, age, number of words). In math, one row is a **vector** and the whole table is a **matrix**.

Most models start by combining features with weights: "price = 2 × size + 0.5 × rooms". That weighted sum is a **dot product**. Doing it for every row at once is a **matrix–vector multiplication**.

So when you see «X @ w» in code, read it as "make a prediction for every row at the same time".`,
      math: md`A data matrix with $n$ rows (samples) and $p$ columns (features):

$$X \in \mathbb{R}^{n \times p}, \qquad w \in \mathbb{R}^{p}$$

The **dot product** of one row $x_i$ with the weights:

$$x_i \cdot w = \sum_{j=1}^{p} x_{ij}\, w_j$$

All predictions at once:

$$\hat{y} = Xw$$

- $x_{ij}$: value of feature $j$ for sample $i$
- $w_j$: how much feature $j$ matters
- $\hat{y}$: the vector of predictions, one per row

Also useful later: the **transpose** $X^\top$ swaps rows and columns, and $X^\top X$ is a $p \times p$ matrix that appears in linear regression.`,
      paper: {
        task: md`Compute $\hat{y} = Xw$ by hand:

$$X = \begin{bmatrix} 1 & 2 \\ 1 & 3 \\ 1 & 5 \end{bmatrix}, \qquad w = \begin{bmatrix} 0.5 \\ 2 \end{bmatrix}$$

Then: what shape does $X^\top X$ have, and what is its top-left entry?`,
        solution: md`Row by row:

- $1 \cdot 0.5 + 2 \cdot 2 = 4.5$
- $1 \cdot 0.5 + 3 \cdot 2 = 6.5$
- $1 \cdot 0.5 + 5 \cdot 2 = 10.5$

So $\hat{y} = [4.5,\ 6.5,\ 10.5]$. The first column of ones acts as an **intercept**: $w_1 = 0.5$ is added to every prediction.

$X^\top X$ is $2 \times 2$. Its top-left entry is $1^2 + 1^2 + 1^2 = 3$, which is simply the number of rows.`,
      },
      code: {
        task: md`Write the matrix–vector product with plain Python loops, then check it against NumPy's «@» operator. Seeing the loop once makes «X @ w» feel obvious forever.`,
        starter: md`import numpy as np

X = np.array([[1, 2],
              [1, 3],
              [1, 5]], dtype=float)
w = np.array([0.5, 2.0])

def matvec(X, w):
    n, p = X.shape
    y_hat = np.zeros(n)
    for i in range(n):
        total = 0.0
        for j in range(p):
            total += X[i, j] * w[j]   # dot product of row i with w
        y_hat[i] = total
    return y_hat

print("loops :", matvec(X, w))
print("numpy :", X @ w)
print("X^T X =\n", X.T @ X)
`,
      },
      real: {
        task: md`Load the diabetes dataset (442 patients, 10 features). Look at its shape, compute the mean of each column, and make a first "prediction" with a weight vector of your choice. Question to answer: why are all column means almost exactly zero?`,
        starter: md`import numpy as np
from sklearn.datasets import load_diabetes

data = load_diabetes()
X, y = data.data, data.target

print("X shape:", X.shape, " y shape:", y.shape)
print("features:", data.feature_names)
print("column means:", np.round(X.mean(axis=0), 4))

# A naive model: only use BMI (column 2) with weight 900, plus the average target
w = np.zeros(X.shape[1])
w[2] = 900
y_hat = X @ w + y.mean()
print("first 5 predictions:", np.round(y_hat[:5], 1))
print("first 5 true values:", y[:5])
`,
      },
    },
    {
      id: 'm-calc',
      title: 'Calculus and gradients',
      summary: 'Derivatives tell you which way is downhill. That is all training needs.',
      guide: { label: '3Blue1Brown: Essence of Calculus', url: 'https://www.3blue1brown.com/topics/calculus' },
      intuition: md`Imagine standing on a hill in fog. You can't see the valley, but you can feel which way the ground slopes under your feet. The **derivative** is that slope.

With many parameters, you need the slope in every direction at once. That list of slopes is the **gradient**. It points uphill, so training a model means repeatedly stepping in the **opposite** direction of the gradient.

The **chain rule** lets you compute slopes of functions nested inside each other, which is exactly what a model's error is: error of (prediction of (input)).`,
      math: md`Derivative of a one-variable function:

$$f'(w) = \lim_{h \to 0} \frac{f(w+h) - f(w)}{h}$$

Rules you need: $\frac{d}{dw} w^2 = 2w$, and the chain rule $\frac{d}{dw} g(h(w)) = g'(h(w))\, h'(w)$.

For a function of several variables, the **gradient** collects all partial derivatives:

$$\nabla f(a, b) = \left[ \frac{\partial f}{\partial a},\ \frac{\partial f}{\partial b} \right]$$

Example: $f(w) = (w-3)^2$ gives $f'(w) = 2(w-3)$, which is zero at the minimum $w = 3$.`,
      paper: {
        task: md`1. For $f(w) = (w-3)^2$, compute $f'(1)$ and $f'(5)$. Which direction should you step from each point to go downhill?
2. For $f(a, b) = a^2 + 3ab$, compute the gradient at $(a, b) = (1, 2)$.`,
        solution: md`1. $f'(1) = 2(1-3) = -4$: the slope is negative, so step **right** (increase $w$). $f'(5) = 2(5-3) = 4$: step **left**. Both point toward $w = 3$.
2. $\frac{\partial f}{\partial a} = 2a + 3b = 2 + 6 = 8$ and $\frac{\partial f}{\partial b} = 3a = 3$. So $\nabla f(1,2) = [8,\ 3]$.`,
      },
      code: {
        task: md`Approximate derivatives numerically with a tiny step $h$ (the "finite difference"), and compare them with the exact formulas. This trick is how people check their gradient code.`,
        starter: md`import numpy as np

def f(w):
    return (w - 3) ** 2

def f_prime_exact(w):
    return 2 * (w - 3)

def numeric_derivative(func, w, h=1e-5):
    return (func(w + h) - func(w - h)) / (2 * h)

for w in [1.0, 3.0, 5.0]:
    print(f"w={w}: numeric={numeric_derivative(f, w):.5f}  exact={f_prime_exact(w):.5f}")

# Gradient of f(a, b) = a^2 + 3ab
def g(v):
    a, b = v
    return a**2 + 3*a*b

def numeric_gradient(func, v, h=1e-5):
    grad = np.zeros_like(v)
    for k in range(len(v)):
        step = np.zeros_like(v); step[k] = h
        grad[k] = (func(v + step) - func(v - step)) / (2 * h)
    return grad

print("gradient at (1,2):", numeric_gradient(g, np.array([1.0, 2.0])))
`,
      },
      real: {
        task: md`Use the BMI feature of the diabetes data. Define the error of the model "prediction = w × bmi + mean(y)" as a function of $w$. Find the $w$ where the numeric derivative of the error is (almost) zero. That $w$ is the best weight.`,
        starter: md`import numpy as np
from sklearn.datasets import load_diabetes

X, y = load_diabetes(return_X_y=True)
bmi = X[:, 2]

def error(w):
    pred = w * bmi + y.mean()
    return np.mean((y - pred) ** 2)

def slope(w, h=1e-3):
    return (error(w + h) - error(w - h)) / (2 * h)

for w in [0, 500, 900, 1000, 1500]:
    print(f"w={w:5}: error={error(w):8.1f}  slope={slope(w):9.3f}")

# Where is the slope zero? Search a fine grid.
ws = np.linspace(0, 2000, 2001)
slopes = np.array([abs(slope(w)) for w in ws])
print("best w is about", ws[np.argmin(slopes)])
`,
      },
    },
    {
      id: 'm-prob',
      title: 'Probability and statistics',
      summary: 'Mean, spread, the normal curve and Bayes’ theorem: the language of uncertainty.',
      guide: { label: 'Mathematics for Machine Learning (free book)', url: 'https://mml-book.github.io/' },
      intuition: md`Data is noisy, so ML talks in averages and probabilities. The **mean** is the center of your data, the **variance** says how spread out it is, and the **standard deviation** is the spread in the original units.

**Conditional probability** answers "how likely is A, given that I know B?". **Bayes' theorem** flips it around: from "how often does a sick person test positive" to "how likely am I sick, given a positive test". This is the heart of Naive Bayes and of thinking clearly about classifiers.`,
      math: md`Mean and variance of $n$ values:

$$\mu = \frac{1}{n}\sum_{i=1}^n x_i, \qquad \sigma^2 = \frac{1}{n}\sum_{i=1}^n (x_i - \mu)^2, \qquad \sigma = \sqrt{\sigma^2}$$

The **z-score** says how many standard deviations a value is from the mean:

$$z = \frac{x - \mu}{\sigma}$$

**Bayes' theorem:**

$$P(A \mid B) = \frac{P(B \mid A)\, P(A)}{P(B)}$$

where $P(B) = P(B \mid A)P(A) + P(B \mid \neg A)P(\neg A)$.`,
      paper: {
        task: md`1. For the values $[2, 4, 4, 4, 5, 5, 7, 9]$, compute the mean, variance and standard deviation. What is the z-score of 9?
2. A disease affects 1% of people. A test detects it 90% of the time, but also gives a positive result for 5% of healthy people. You test positive. What is the probability you have the disease?`,
        solution: md`1. Mean $= 40/8 = 5$. Squared deviations: $9,1,1,1,0,0,4,16$, sum $= 32$. Variance $= 32/8 = 4$, standard deviation $= 2$. z-score of 9 is $(9-5)/2 = 2$.
2. $P(+) = 0.9 \cdot 0.01 + 0.05 \cdot 0.99 = 0.009 + 0.0495 = 0.0585$. Then $P(D \mid +) = 0.009 / 0.0585 \approx 0.154$. **Only about 15%**, because healthy people vastly outnumber sick ones. The same effect makes "99% accuracy" meaningless on rare events like fraud.`,
      },
      code: {
        task: md`Implement mean, variance, standard deviation and z-scores yourself. Then check the Bayes answer by **simulating** one million people.`,
        starter: md`import numpy as np

x = np.array([2, 4, 4, 4, 5, 5, 7, 9], dtype=float)

def mean(v): return sum(v) / len(v)
def var(v):
    m = mean(v)
    return sum((vi - m) ** 2 for vi in v) / len(v)

print("mean:", mean(x), " numpy:", x.mean())
print("var :", var(x), " numpy:", x.var())
print("std :", var(x) ** 0.5, " numpy:", x.std())
print("z   :", (x - mean(x)) / var(x) ** 0.5)

# Simulate the disease test
rng = np.random.default_rng(0)
n = 1_000_000
sick = rng.random(n) < 0.01
positive = np.where(sick, rng.random(n) < 0.90, rng.random(n) < 0.05)
print("P(sick | positive) ~", round(sick[positive].mean(), 4))
`,
      },
      real: {
        task: md`In the breast cancer dataset, compare the mean and spread of «mean radius» for benign and malignant tumors. Plot both histograms. Ask yourself: how well could one feature alone separate the two classes?`,
        starter: md`import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer

data = load_breast_cancer()
X, y = data.data, data.target          # 0 = malignant, 1 = benign
radius = X[:, 0]

print("class counts:", np.bincount(y), "->", data.target_names)
for label in [0, 1]:
    r = radius[y == label]
    print(f"{data.target_names[label]:9}: mean={r.mean():.2f}  std={r.std():.2f}")

plt.hist(radius[y == 0], bins=30, alpha=0.6, label="malignant")
plt.hist(radius[y == 1], bins=30, alpha=0.6, label="benign")
plt.xlabel("mean radius"); plt.ylabel("count"); plt.legend()
plt.show()
`,
      },
    },
    {
      id: 'm-opt',
      title: 'The optimization idea',
      summary: 'A model is a function with knobs. Training means turning the knobs until the error is smallest.',
      guide: { label: 'Mathematics for Machine Learning, ch. 7', url: 'https://mml-book.github.io/' },
      intuition: md`Almost every algorithm in the scikit-learn guide follows the same recipe:

- pick a **model**: a formula with adjustable parameters (the knobs)
- pick a **loss**: a number that says how wrong the model is on the training data
- find the parameters that make the loss as small as possible

Linear regression, logistic regression, SVMs and neural networks only differ in the formula and the loss. Once you see this pattern, the guide becomes much less intimidating.`,
      math: md`For a model $f_\theta$ with parameters $\theta$, training solves:

$$\theta^* = \arg\min_{\theta} \; L(\theta)$$

The most common loss for predicting numbers is the **mean squared error**:

$$L(\theta) = \frac{1}{n} \sum_{i=1}^{n} \big(y_i - f_\theta(x_i)\big)^2$$

For the simple model $f(x) = wx$, setting the derivative to zero gives a direct answer:

$$w^* = \frac{\sum_i x_i y_i}{\sum_i x_i^2}$$`,
      paper: {
        task: md`Data points: $(1, 2),\ (2, 4),\ (3, 5)$. Model: $\hat{y} = wx$.

1. Compute the MSE for $w = 1.5$ and for $w = 2$.
2. Compute the best $w^*$ with the formula and its MSE.`,
        solution: md`1. $w = 1.5$: predictions $1.5, 3, 4.5$; errors $0.5, 1, 0.5$; MSE $= (0.25 + 1 + 0.25)/3 = 0.5$.
   $w = 2$: predictions $2, 4, 6$; errors $0, 0, -1$; MSE $= 1/3 \approx 0.333$.
2. $w^* = (2 + 8 + 15)/(1 + 4 + 9) = 25/14 \approx 1.786$. Predictions $1.786, 3.571, 5.357$; MSE $\approx 0.119$. Lower than both guesses, as it must be.`,
      },
      code: {
        task: md`Find the best $w$ by brute force: try many values, compute the loss for each, and pick the smallest. Plot the loss curve. Compare with the formula.`,
        starter: md`import numpy as np
import matplotlib.pyplot as plt

x = np.array([1, 2, 3], dtype=float)
y = np.array([2, 4, 5], dtype=float)

def mse(w):
    return np.mean((y - w * x) ** 2)

ws = np.linspace(0, 4, 401)
losses = [mse(w) for w in ws]
w_grid = ws[np.argmin(losses)]
w_formula = (x * y).sum() / (x ** 2).sum()

print("best w (grid)   :", w_grid)
print("best w (formula):", round(w_formula, 4))

plt.plot(ws, losses)
plt.axvline(w_formula, linestyle="--")
plt.xlabel("w"); plt.ylabel("MSE"); plt.title("The loss is a valley")
plt.show()
`,
      },
      real: {
        task: md`Fit "disease progression = w × bmi + b" on the diabetes data by searching over a grid of both $w$ and $b$. Then check your result with scikit-learn's «LinearRegression». Your grid answer should be close.`,
        starter: md`import numpy as np
from sklearn.datasets import load_diabetes
from sklearn.linear_model import LinearRegression

X, y = load_diabetes(return_X_y=True)
bmi = X[:, 2]

best = (None, None, np.inf)
for w in np.linspace(0, 1500, 151):
    for b in np.linspace(100, 200, 101):
        loss = np.mean((y - (w * bmi + b)) ** 2)
        if loss < best[2]:
            best = (w, b, loss)

print("grid search: w=%.1f  b=%.1f  MSE=%.1f" % best)

model = LinearRegression().fit(bmi.reshape(-1, 1), y)
print("sklearn    : w=%.1f  b=%.1f" % (model.coef_[0], model.intercept_))
`,
      },
    },
  ],
}
