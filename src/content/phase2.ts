import type { Phase } from '../types'

const md = String.raw
const SK = 'https://scikit-learn.org/stable/'

export const phase2: Phase = {
  id: 'p2',
  title: 'Supervised learning',
  goal: 'The core models, from a straight line to boosted trees. Each one is a model, a loss, and a way to minimize it.',
  weeks: '8–10 weeks',
  topics: [
    {
      id: 's-ols',
      title: 'Linear regression (least squares)',
      summary: 'Fit the straight line that makes the squared errors as small as possible.',
      guide: { label: 'User Guide 1.1.1: Ordinary Least Squares', url: SK + 'modules/linear_model.html#ordinary-least-squares' },
      widget: 'ols',
      intuition: md`You have points on a scatter plot: house size against price. Linear regression draws the one straight line that is "closest" to all of them.

"Closest" means: for each point, measure the vertical gap (the **residual**) between the point and the line, square it, and add them up. The best line has the smallest total. Squaring punishes big misses much more than small ones.

Try it in the playground below: click to add points and drag them around. Turn on the squares to literally see the quantity being minimized.`,
      math: md`Model with one feature:

$$\hat{y} = \beta_0 + \beta_1 x$$

Minimize the sum of squared residuals:

$$\min_{\beta_0, \beta_1} \sum_{i=1}^{n} (y_i - \beta_0 - \beta_1 x_i)^2$$

Setting both derivatives to zero gives:

$$\beta_1 = \frac{\sum_i (x_i - \bar{x})(y_i - \bar{y})}{\sum_i (x_i - \bar{x})^2}, \qquad \beta_0 = \bar{y} - \beta_1 \bar{x}$$

With many features (and a column of ones in $X$ for the intercept), the **normal equation**:

$$w = (X^\top X)^{-1} X^\top y$$`,
      paper: {
        task: md`Five apartments:

| size (m²) | 50 | 70 | 80 | 100 | 120 |
|---|---|---|---|---|---|
| price (k€) | 150 | 200 | 230 | 280 | 330 |

Compute $\beta_1$ and $\beta_0$ by hand. What does $\beta_1$ mean in plain words? Predict the price of a 90 m² apartment.`,
        solution: md`$\bar{x} = 84$, $\bar{y} = 238$.

Deviations $x - \bar{x}$: $-34, -14, -4, 16, 36$. Deviations $y - \bar{y}$: $-88, -38, -8, 42, 92$.

$\sum (x-\bar{x})(y-\bar{y}) = 2992 + 532 + 32 + 672 + 3312 = 7540$

$\sum (x - \bar{x})^2 = 1156 + 196 + 16 + 256 + 1296 = 2920$

$\beta_1 = 7540 / 2920 \approx 2.582$, and $\beta_0 = 238 - 2.582 \cdot 84 \approx 21.1$.

Meaning: **each extra square meter adds about 2,580 €**. For 90 m²: $21.1 + 2.582 \cdot 90 \approx 253$ k€.`,
      },
      code: {
        task: md`Implement linear regression with the normal equation. Add a column of ones for the intercept. Compare your coefficients with «LinearRegression» on the apartment data.`,
        starter: md`import numpy as np
from sklearn.linear_model import LinearRegression

size = np.array([50, 70, 80, 100, 120], dtype=float)
price = np.array([150, 200, 230, 280, 330], dtype=float)

X = np.c_[np.ones_like(size), size]          # column of ones = intercept
w = np.linalg.inv(X.T @ X) @ X.T @ price      # normal equation
print("mine   : b0=%.3f  b1=%.4f" % (w[0], w[1]))

ref = LinearRegression().fit(size.reshape(-1, 1), price)
print("sklearn: b0=%.3f  b1=%.4f" % (ref.intercept_, ref.coef_[0]))
print("prediction for 90 m2:", round(w[0] + w[1] * 90, 1))
`,
      },
      real: {
        task: md`Predict house prices with the California Housing dataset (it downloads on your computer; in the browser the code falls back to the diabetes data). Look at the coefficients: which features push the price up, which push it down? Report R² on the test set.`,
        starter: md`import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

try:
    from sklearn.datasets import fetch_california_housing
    data = fetch_california_housing()
    print("Using California Housing (target: median house value in 100k$)")
except Exception:
    from sklearn.datasets import load_diabetes
    data = load_diabetes()
    print("Download not available here, using the diabetes data instead")

X_tr, X_te, y_tr, y_te = train_test_split(data.data, data.target, random_state=0)
model = LinearRegression().fit(X_tr, y_tr)

for name, coef in sorted(zip(data.feature_names, model.coef_), key=lambda t: -abs(t[1])):
    print(f"{name:12} {coef:+.4f}")
print("test R2:", round(model.score(X_te, y_te), 3))
`,
      },
    },
    {
      id: 's-gd',
      title: 'Gradient descent',
      summary: 'Finding the minimum by taking small steps downhill, again and again.',
      guide: { label: 'User Guide 1.5: Stochastic Gradient Descent', url: SK + 'modules/sgd.html' },
      widget: 'gd',
      intuition: md`The normal equation solves linear regression in one shot, but it becomes slow with huge data and doesn't exist for most other models. **Gradient descent** works for almost everything: start somewhere, compute the slope of the loss, take a small step downhill, repeat.

The **learning rate** is the step size. Too small and training takes forever. Too large and you overshoot the valley and bounce around or even explode. Play with it in the visual below.

**Stochastic** gradient descent (SGD) uses one sample or a small batch per step instead of the whole dataset, which makes each step much cheaper.`,
      math: md`Update rule, with learning rate $\eta$:

$$w \leftarrow w - \eta \, \nabla L(w)$$

For linear regression with the MSE loss $L(w) = \frac{1}{n} \lVert y - Xw \rVert^2$:

$$\nabla L(w) = \frac{2}{n} X^\top (Xw - y)$$

For the one-weight model $\hat{y} = wx$:

$$\frac{dL}{dw} = \frac{2}{n} \sum_i (w x_i - y_i)\, x_i$$`,
      paper: {
        task: md`Data: $(1, 2)$ and $(2, 4)$. Model $\hat{y} = wx$, start at $w = 0$, learning rate $\eta = 0.1$.

Do two gradient descent steps by hand. Where is it heading?`,
        solution: md`Step 1: errors $wx - y$ are $-2$ and $-4$. Gradient $= \frac{2}{2}\big((-2)(1) + (-4)(2)\big) = -10$. New $w = 0 - 0.1 \cdot (-10) = 1$.

Step 2: predictions $1, 2$; errors $-1, -2$. Gradient $= (-1)(1) + (-2)(2) = -5$. New $w = 1 + 0.5 = 1.5$.

It is heading to $w = 2$ (the perfect fit), halving the distance each step.`,
      },
      code: {
        task: md`Implement batch gradient descent for linear regression with an intercept. Record the loss at every step and plot it. Then try learning rates 0.001, 0.01 and 0.5 and describe what happens.`,
        starter: md`import numpy as np
import matplotlib.pyplot as plt

rng = np.random.default_rng(0)
x = rng.uniform(0, 2, 100)
y = 4 + 3 * x + rng.normal(0, 0.5, 100)    # true line: 4 + 3x
X = np.c_[np.ones_like(x), x]

def gradient_descent(lr, steps=200):
    w = np.zeros(2)
    losses = []
    for _ in range(steps):
        error = X @ w - y
        losses.append(np.mean(error ** 2))
        grad = 2 / len(y) * X.T @ error
        w = w - lr * grad
    return w, losses

for lr in [0.01, 0.1, 0.5]:
    w, losses = gradient_descent(lr)
    print(f"lr={lr:<5} w={np.round(w, 3)}  final loss={losses[-1]:.4f}")
    plt.plot(losses, label=f"lr={lr}")

plt.yscale("log"); plt.xlabel("step"); plt.ylabel("MSE"); plt.legend()
plt.show()
`,
      },
      real: {
        task: md`Fit «SGDRegressor» on the (scaled) diabetes data and compare its coefficients with «LinearRegression». Then remove the scaler and see what happens. Why does gradient descent care so much about scaling?`,
        starter: md`import numpy as np
from sklearn.datasets import load_diabetes
from sklearn.linear_model import LinearRegression, SGDRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

X, y = load_diabetes(return_X_y=True)
X = X * 50 + 3        # undo the built-in scaling to make the data "raw"

exact = make_pipeline(StandardScaler(), LinearRegression()).fit(X, y)
sgd = make_pipeline(StandardScaler(), SGDRegressor(max_iter=2000, random_state=0)).fit(X, y)
print("exact coefs:", np.round(exact[-1].coef_, 1))
print("sgd coefs  :", np.round(sgd[-1].coef_, 1))

raw_sgd = SGDRegressor(max_iter=2000, random_state=0).fit(X, y)
print("R2 with scaling   :", round(sgd.score(X, y), 3))
print("R2 without scaling:", round(raw_sgd.score(X, y), 3))
`,
      },
    },
    {
      id: 's-ridge-lasso',
      title: 'Ridge and Lasso',
      summary: 'Regularization: penalizing big weights to fight overfitting and pick useful features.',
      guide: { label: 'User Guide 1.1.2–1.1.3: Ridge and Lasso', url: SK + 'modules/linear_model.html#ridge-regression-and-classification' },
      intuition: md`With many features and little data, linear regression can find huge, wild weights that fit the noise. **Regularization** adds a penalty for large weights, a kind of "keep it simple" rule.

**Ridge** (L2) shrinks all weights smoothly toward zero. **Lasso** (L1) can push some weights to exactly zero, so it also selects features for you.

The strength of the penalty is $\alpha$. With $\alpha = 0$ you get plain linear regression; with huge $\alpha$ all weights go to zero. This is the **bias–variance trade-off** in one knob.`,
      math: md`**Ridge** loss:

$$L(w) = \lVert y - Xw \rVert^2 + \alpha \lVert w \rVert_2^2, \qquad \lVert w \rVert_2^2 = \sum_j w_j^2$$

Closed-form solution:

$$w = (X^\top X + \alpha I)^{-1} X^\top y$$

**Lasso** loss (scikit-learn's scaling):

$$L(w) = \frac{1}{2n} \lVert y - Xw \rVert^2 + \alpha \lVert w \rVert_1, \qquad \lVert w \rVert_1 = \sum_j |w_j|$$

The L1 penalty has a "corner" at zero, which is why Lasso produces exact zeros.`,
      paper: {
        task: md`One feature, no intercept: $x = [1, 2, 3]$, $y = [2, 4, 5]$. For a single weight the ridge solution is

$$w = \frac{\sum x_i y_i}{\sum x_i^2 + \alpha}$$

Compute $w$ for $\alpha = 0$, $\alpha = 1$ and $\alpha = 14$. What happens as $\alpha$ grows?`,
        solution: md`$\sum x_i y_i = 25$ and $\sum x_i^2 = 14$.

- $\alpha = 0$: $w = 25/14 \approx 1.786$ (ordinary least squares)
- $\alpha = 1$: $w = 25/15 \approx 1.667$
- $\alpha = 14$: $w = 25/28 \approx 0.893$

The weight **shrinks toward zero** as $\alpha$ grows. With $\alpha$ equal to $\sum x_i^2$, it is cut in half.`,
      },
      code: {
        task: md`Implement ridge regression with the closed-form formula and compare it to «Ridge(fit_intercept=False)». Then plot how each coefficient shrinks as $\alpha$ grows.`,
        starter: md`import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_diabetes
from sklearn.linear_model import Ridge

X, y = load_diabetes(return_X_y=True)
y = y - y.mean()          # center y so we can skip the intercept

def ridge(X, y, alpha):
    p = X.shape[1]
    return np.linalg.solve(X.T @ X + alpha * np.eye(p), X.T @ y)

print("mine   :", np.round(ridge(X, y, 1.0)[:4], 2))
print("sklearn:", np.round(Ridge(alpha=1.0, fit_intercept=False).fit(X, y).coef_[:4], 2))

alphas = np.logspace(-3, 2, 50)
coefs = np.array([ridge(X, y, a) for a in alphas])
plt.semilogx(alphas, coefs)
plt.xlabel("alpha"); plt.ylabel("coefficient"); plt.title("Ridge shrinks every weight")
plt.show()
`,
      },
      real: {
        task: md`Fit Lasso on the diabetes data for increasing $\alpha$ and list which features survive (non-zero weight). Which three features are most important for predicting disease progression?`,
        starter: md`import numpy as np
from sklearn.datasets import load_diabetes
from sklearn.linear_model import Lasso, LassoCV

data = load_diabetes()
X, y = data.data, data.target

for alpha in [0.01, 0.1, 0.5, 1.0, 2.0]:
    coef = Lasso(alpha=alpha).fit(X, y).coef_
    kept = [n for n, c in zip(data.feature_names, coef) if c != 0]
    print(f"alpha={alpha:<4} kept {len(kept):2}: {kept}")

best = LassoCV(cv=5).fit(X, y)
print("\nbest alpha by CV:", round(best.alpha_, 4))
for n, c in sorted(zip(data.feature_names, best.coef_), key=lambda t: -abs(t[1])):
    print(f"  {n:4} {c:+8.1f}")
`,
      },
    },
    {
      id: 's-logreg',
      title: 'Logistic regression',
      summary: 'A linear model that outputs probabilities for yes/no questions.',
      guide: { label: 'User Guide 1.1.11: Logistic regression', url: SK + 'modules/linear_model.html#logistic-regression' },
      intuition: md`Despite its name, logistic regression is a **classifier**. It computes the same weighted sum as linear regression, but then squeezes the result through the **sigmoid** function, an S-shaped curve that turns any number into a probability between 0 and 1.

Large positive score → probability near 1. Large negative score → near 0. Score of 0 → exactly 0.5, which is the decision boundary.

Training uses the **log-loss**: confident wrong answers are punished very heavily, confident right answers barely at all.`,
      math: md`Score and probability:

$$z = w^\top x + b, \qquad p = \sigma(z) = \frac{1}{1 + e^{-z}}$$

Log-loss for one sample with label $y \in \{0, 1\}$:

$$\ell = -\big[\, y \log p + (1 - y) \log(1 - p) \,\big]$$

Its gradient has the same simple form as linear regression:

$$\frac{\partial \ell}{\partial w} = (p - y)\, x$$

Interpretation: $e^{w_j}$ is the **odds ratio**, how much the odds multiply when feature $j$ increases by one.`,
      paper: {
        task: md`A model has $w = 2$, $b = -4$. A sample has $x = 3$.

1. Compute $z$ and $p$.
2. Compute the log-loss if the true label is $y = 1$, and if it is $y = 0$.`,
        solution: md`1. $z = 2 \cdot 3 - 4 = 2$. $p = 1/(1 + e^{-2}) = 1/(1 + 0.135) \approx 0.881$.
2. If $y = 1$: $\ell = -\ln 0.881 \approx 0.127$ (small, the model was right).
   If $y = 0$: $\ell = -\ln(1 - 0.881) = -\ln 0.119 \approx 2.13$ (big, it was confidently wrong).`,
      },
      code: {
        task: md`Implement logistic regression trained by gradient descent on the log-loss. Compare its accuracy with scikit-learn's «LogisticRegression» on scaled breast cancer data.`,
        starter: md`import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

X, y = load_breast_cancer(return_X_y=True)
X = StandardScaler().fit_transform(X)

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

w, b, lr = np.zeros(X.shape[1]), 0.0, 0.1
for step in range(2000):
    p = sigmoid(X @ w + b)
    w -= lr * X.T @ (p - y) / len(y)
    b -= lr * np.mean(p - y)
    if step % 500 == 0:
        loss = -np.mean(y * np.log(p + 1e-12) + (1 - y) * np.log(1 - p + 1e-12))
        print(f"step {step:4}: log-loss {loss:.4f}")

mine = np.mean((sigmoid(X @ w + b) > 0.5) == y)
ref = LogisticRegression(max_iter=1000).fit(X, y).score(X, y)
print("accuracy mine: %.3f  sklearn: %.3f" % (mine, ref))
`,
      },
      real: {
        task: md`Train logistic regression for breast cancer diagnosis and print the five features with the largest positive and negative weights. Translate one weight into an odds ratio in plain language.`,
        starter: md`import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

data = load_breast_cancer()     # target 1 = benign
pipe = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000)).fit(data.data, data.target)
coef = pipe[-1].coef_[0]

order = np.argsort(coef)
print("pushes toward MALIGNANT:")
for i in order[:5]:
    print(f"  {data.feature_names[i]:24} w={coef[i]:+.2f}  odds x{np.exp(coef[i]):.2f} per std")
print("pushes toward BENIGN:")
for i in order[-5:]:
    print(f"  {data.feature_names[i]:24} w={coef[i]:+.2f}  odds x{np.exp(coef[i]):.2f} per std")

print("\nprobabilities for 3 patients:", pipe.predict_proba(data.data[:3])[:, 1].round(3))
`,
      },
    },
    {
      id: 's-knn',
      title: 'k-nearest neighbors',
      summary: 'Predict by asking the most similar examples you have already seen.',
      guide: { label: 'User Guide 1.6: Nearest Neighbors', url: SK + 'modules/neighbors.html' },
      widget: 'knn',
      intuition: md`k-NN has no training at all. To classify a new point, find the $k$ closest points in the training data and let them vote. To predict a number, average their values.

It is the "ask your neighbors" algorithm: if the 5 houses most similar to yours sold for around 300k, yours probably will too.

Small $k$ follows every wiggle in the data (overfitting); large $k$ smooths everything out (underfitting). Because it's all about distance, **scaling matters a lot**. Move the query point and change $k$ in the playground to see the vote.`,
      math: md`Euclidean distance between two points:

$$d(a, b) = \sqrt{\sum_{j=1}^{p} (a_j - b_j)^2}$$

Classification: let $N_k(x)$ be the $k$ nearest training points. Predict the majority class

$$\hat{y} = \arg\max_c \sum_{i \in N_k(x)} \mathbb{1}[y_i = c]$$

Regression: $\hat{y} = \frac{1}{k}\sum_{i \in N_k(x)} y_i$.

Manhattan distance is an alternative: $d(a,b) = \sum_j |a_j - b_j|$.`,
      paper: {
        task: md`Training points: A(1, 1) red, B(2, 1) red, C(4, 4) blue, D(5, 4) blue. Query point Q(2, 2).

Compute all four distances. What does k-NN predict for $k = 1$ and $k = 3$?`,
        solution: md`- $d(Q, A) = \sqrt{1 + 1} \approx 1.41$
- $d(Q, B) = \sqrt{0 + 1} = 1$
- $d(Q, C) = \sqrt{4 + 4} \approx 2.83$
- $d(Q, D) = \sqrt{9 + 4} \approx 3.61$

$k = 1$: nearest is B → **red**. $k = 3$: B, A, C → two red, one blue → **red**.`,
      },
      code: {
        task: md`Implement a k-NN classifier with NumPy: compute all distances, take the $k$ smallest with «argsort», and vote. Compare with «KNeighborsClassifier» on Iris.`,
        starter: md`import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

def knn_predict(X_train, y_train, X_new, k=5):
    preds = []
    for x in X_new:
        dist = np.sqrt(((X_train - x) ** 2).sum(axis=1))
        nearest = np.argsort(dist)[:k]
        preds.append(np.bincount(y_train[nearest]).argmax())
    return np.array(preds)

X, y = load_iris(return_X_y=True)
X_tr, X_te, y_tr, y_te = train_test_split(X, y, random_state=0)

mine = knn_predict(X_tr, y_tr, X_te, k=5)
ref = KNeighborsClassifier(n_neighbors=5).fit(X_tr, y_tr).predict(X_te)
print("accuracy mine   :", np.mean(mine == y_te).round(3))
print("accuracy sklearn:", np.mean(ref == y_te).round(3))
`,
      },
      real: {
        task: md`Recognize handwritten digits with k-NN. Plot accuracy against $k$, then show a few digits the model got wrong. Would a human also struggle with them?`,
        starter: md`import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.neighbors import KNeighborsClassifier

X, y = load_digits(return_X_y=True)
ks = [1, 3, 5, 9, 15, 25, 50]
scores = [cross_val_score(KNeighborsClassifier(k), X, y, cv=5).mean() for k in ks]
print(dict(zip(ks, np.round(scores, 3))))

X_tr, X_te, y_tr, y_te = train_test_split(X, y, random_state=0)
pred = KNeighborsClassifier(3).fit(X_tr, y_tr).predict(X_te)
wrong = np.where(pred != y_te)[0][:6]

fig, axes = plt.subplots(1, 7, figsize=(12, 2))
axes[0].plot(ks, scores, "o-"); axes[0].set_title("accuracy vs k")
for ax, i in zip(axes[1:], wrong):
    ax.imshow(X_te[i].reshape(8, 8), cmap="gray_r"); ax.axis("off")
    ax.set_title(f"true {y_te[i]}, said {pred[i]}")
plt.tight_layout(); plt.show()
`,
      },
    },
    {
      id: 's-nb',
      title: 'Naive Bayes',
      summary: 'Bayes’ theorem plus one bold simplification: features are independent.',
      guide: { label: 'User Guide 1.9: Naive Bayes', url: SK + 'modules/naive_bayes.html' },
      intuition: md`A spam filter sees the words "free" and "winner". How likely is spam? Naive Bayes multiplies together how typical each word is for spam, and compares that with how typical the words are for normal mail.

It is **naive** because it assumes words appear independently of each other, which is clearly false ("free" and "shipping" love to appear together). Surprisingly, it still works very well for text, is extremely fast, and needs little data.`,
      math: md`For class $c$ and features $x_1, \dots, x_p$, assuming independence:

$$P(c \mid x) \propto P(c) \prod_{j=1}^{p} P(x_j \mid c)$$

Predict the class with the largest value. In practice we add logarithms to avoid tiny numbers:

$$\hat{y} = \arg\max_c \Big[ \log P(c) + \sum_j \log P(x_j \mid c) \Big]$$

**Gaussian** NB models each $P(x_j \mid c)$ as a normal distribution with a per-class mean and variance. **Multinomial** NB uses word counts.`,
      paper: {
        task: md`40% of mail is spam. In spam, "free" appears in 50% of emails and "meeting" in 10%. In normal mail, "free" appears in 5% and "meeting" in 40%.

An email contains both "free" and "meeting". What is the probability it is spam, according to Naive Bayes?`,
        solution: md`Spam: $0.4 \cdot 0.5 \cdot 0.1 = 0.020$

Normal: $0.6 \cdot 0.05 \cdot 0.4 = 0.012$

Normalize: $P(\text{spam} \mid \text{email}) = 0.020 / (0.020 + 0.012) = 0.625$.

So about **62.5% spam**: "free" points strongly to spam, "meeting" pulls back toward normal.`,
      },
      code: {
        task: md`Implement Gaussian Naive Bayes: per class, store the prior, the means and variances of each feature, and predict using the sum of log-probabilities. Compare with «GaussianNB» on Iris.`,
        starter: md`import numpy as np
from sklearn.datasets import load_iris
from sklearn.naive_bayes import GaussianNB

X, y = load_iris(return_X_y=True)
classes = np.unique(y)
prior = np.array([np.mean(y == c) for c in classes])
mu = np.array([X[y == c].mean(axis=0) for c in classes])
var = np.array([X[y == c].var(axis=0) + 1e-9 for c in classes])

def log_gauss(x, m, v):
    return -0.5 * np.log(2 * np.pi * v) - (x - m) ** 2 / (2 * v)

def predict(X):
    scores = np.array([np.log(prior[c]) + log_gauss(X, mu[c], var[c]).sum(axis=1) for c in classes])
    return scores.argmax(axis=0)

print("accuracy mine   :", np.mean(predict(X) == y).round(3))
print("accuracy sklearn:", round(GaussianNB().fit(X, y).score(X, y), 3))
`,
      },
      real: {
        task: md`Classify news articles by topic with the 20 Newsgroups dataset (downloads about 14 MB, so run this on your computer). Turn text into word counts, train Multinomial Naive Bayes, and print the most telling words per topic.`,
        local: true,
        starter: md`import numpy as np
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline

cats = ["sci.space", "rec.autos", "comp.graphics", "talk.politics.misc"]
train = fetch_20newsgroups(subset="train", categories=cats, remove=("headers", "footers", "quotes"))
test = fetch_20newsgroups(subset="test", categories=cats, remove=("headers", "footers", "quotes"))

model = make_pipeline(TfidfVectorizer(stop_words="english"), MultinomialNB(alpha=0.1))
model.fit(train.data, train.target)
print("test accuracy:", round(model.score(test.data, test.target), 3))

vec, nb = model[0], model[1]
words = np.array(vec.get_feature_names_out())
for i, cat in enumerate(train.target_names):
    print(f"{cat:20}", ", ".join(words[np.argsort(nb.feature_log_prob_[i])[-8:]]))

print(model.predict(["The rocket launch to orbit was delayed"]))
`,
      },
    },
    {
      id: 's-tree',
      title: 'Decision trees',
      summary: 'A flowchart of yes/no questions, learned from data.',
      guide: { label: 'User Guide 1.10: Decision Trees', url: SK + 'modules/tree.html' },
      intuition: md`A decision tree is a game of 20 questions. "Is the passenger female? Is the ticket first class? Younger than 10?" Each question splits the data into two groups, and the tree chooses the question that makes the groups as **pure** as possible (mostly one class).

Trees are easy to read and need no scaling. Their weakness: grown deep enough, they memorize every training example. Limiting depth or pruning keeps them honest.`,
      math: md`**Gini impurity** of a node with class proportions $p_k$:

$$G = 1 - \sum_k p_k^2$$

**Entropy** (alternative): $H = -\sum_k p_k \log_2 p_k$.

A split into left (L) and right (R) children is scored by the weighted impurity:

$$G_{\text{split}} = \frac{n_L}{n} G_L + \frac{n_R}{n} G_R$$

The tree picks the split with the largest **impurity decrease** $G_{\text{parent}} - G_{\text{split}}$.`,
      paper: {
        task: md`A node has 10 passengers: 6 survived, 4 did not. A split creates:

- left: 4 passengers, all survived
- right: 6 passengers, 2 survived, 4 did not

Compute the Gini of the parent, both children, the weighted split Gini, and the impurity decrease.`,
        solution: md`Parent: $1 - 0.6^2 - 0.4^2 = 1 - 0.36 - 0.16 = 0.48$.

Left: pure, $G = 0$. Right: $1 - (1/3)^2 - (2/3)^2 = 1 - 0.111 - 0.444 \approx 0.444$.

Weighted: $0.4 \cdot 0 + 0.6 \cdot 0.444 \approx 0.267$.

Decrease: $0.48 - 0.267 \approx 0.213$. A good split.`,
      },
      code: {
        task: md`Write a function that finds the best threshold for one feature by trying every possible split and computing the weighted Gini. Check that a depth-1 «DecisionTreeClassifier» picks the same threshold.`,
        starter: md`import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.tree import DecisionTreeClassifier

def gini(y):
    if len(y) == 0:
        return 0.0
    p = np.bincount(y) / len(y)
    return 1 - np.sum(p ** 2)

def best_split(x, y):
    values = np.unique(x)
    best = (None, np.inf)
    for t in (values[:-1] + values[1:]) / 2:       # midpoints
        left, right = y[x <= t], y[x > t]
        g = len(left) / len(y) * gini(left) + len(right) / len(y) * gini(right)
        if g < best[1]:
            best = (t, g)
    return best

data = load_breast_cancer()
x = data.data[:, 22]       # "worst perimeter"
t, g = best_split(x, data.target)
print("mine   : threshold %.3f  weighted gini %.4f" % (t, g))

stump = DecisionTreeClassifier(max_depth=1).fit(x.reshape(-1, 1), data.target)
print("sklearn: threshold %.3f" % stump.tree_.threshold[0])
`,
      },
      real: {
        task: md`Who survived the Titanic? Load the data from OpenML (run this locally, it downloads), train a shallow tree, and print its rules. Describe the survival story the tree tells.`,
        local: true,
        starter: md`from sklearn.datasets import fetch_openml
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeClassifier, export_text

titanic = fetch_openml("titanic", version=1, as_frame=True)
df = titanic.frame
X = df[["pclass", "sex", "age", "sibsp", "parch", "fare"]].copy()
X["sex"] = (X["sex"] == "female").astype(int)
X = X.fillna(X.median(numeric_only=True))
y = (df["survived"] == "1").astype(int)

tree = DecisionTreeClassifier(max_depth=3, random_state=0)
print("CV accuracy:", cross_val_score(tree, X, y, cv=5).mean().round(3))

tree.fit(X, y)
print(export_text(tree, feature_names=list(X.columns)))
`,
      },
    },
    {
      id: 's-ensemble',
      title: 'Ensembles: forests and boosting',
      summary: 'Many weak trees together beat one strong tree.',
      guide: { label: 'User Guide 1.11: Ensembles', url: SK + 'modules/ensemble.html' },
      intuition: md`Ask one expert and you get their mistakes. Ask 500 independent experts and average, and the mistakes cancel out. That is a **random forest**: many deep trees, each trained on a random sample of rows and features, voting together.

**Boosting** works differently, like a team that learns from failure. The first small tree makes predictions; the next tree is trained to fix the first one's errors (the residuals); the next fixes what's left, and so on. Gradient boosting is often the best model for tabular data in practice.`,
      math: md`**Bagging**: average $B$ models trained on bootstrap samples:

$$\hat{f}(x) = \frac{1}{B} \sum_{b=1}^{B} f_b(x)$$

If the models had variance $\sigma^2$ and were independent, the average would have variance $\sigma^2 / B$. Random feature choice makes trees less correlated, so this works better.

**Gradient boosting** (squared error), with learning rate $\nu$:

$$F_0(x) = \bar{y}, \qquad r_i = y_i - F_{m-1}(x_i), \qquad F_m(x) = F_{m-1}(x) + \nu\, h_m(x)$$

where $h_m$ is a small tree fitted to the residuals $r_i$.`,
      paper: {
        task: md`Three samples with targets $y = [3, 5, 9]$. Do one round of gradient boosting with $\nu = 0.5$.

1. Compute $F_0$ and the residuals.
2. A stump groups samples 1–2 on the left and sample 3 on the right, predicting the mean residual of each group. Compute $F_1$ for each sample.`,
        solution: md`1. $F_0 = 17/3 \approx 5.667$. Residuals: $-2.667,\ -0.667,\ 3.333$.
2. Left mean residual $= -1.667$, right $= 3.333$.
   $F_1 = 5.667 + 0.5 \cdot (-1.667) \approx 4.833$ for samples 1 and 2, and $5.667 + 0.5 \cdot 3.333 \approx 7.333$ for sample 3.

All predictions moved toward their targets. More rounds keep shrinking the residuals.`,
      },
      code: {
        task: md`Build bagging by hand: train 50 decision trees on bootstrap samples and average their votes. Compare one tree, your bagged ensemble, and «RandomForestClassifier».`,
        starter: md`import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

X, y = load_breast_cancer(return_X_y=True)
X_tr, X_te, y_tr, y_te = train_test_split(X, y, random_state=0)
rng = np.random.default_rng(0)

votes = np.zeros(len(X_te))
for b in range(50):
    idx = rng.integers(0, len(X_tr), len(X_tr))        # bootstrap sample
    tree = DecisionTreeClassifier(random_state=b).fit(X_tr[idx], y_tr[idx])
    votes += tree.predict(X_te)
bagged = (votes / 50 > 0.5).astype(int)

single = DecisionTreeClassifier(random_state=0).fit(X_tr, y_tr).score(X_te, y_te)
forest = RandomForestClassifier(n_estimators=200, random_state=0).fit(X_tr, y_tr).score(X_te, y_te)
print("single tree :", round(single, 3))
print("my bagging  :", round(np.mean(bagged == y_te), 3))
print("random forest:", round(forest, 3))
`,
      },
      real: {
        task: md`Compare a single tree, a random forest and histogram gradient boosting with 5-fold cross-validation. Which wins, and by how much? Also time them.`,
        starter: md`import time
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeClassifier

X, y = load_breast_cancer(return_X_y=True)
models = {
    "decision tree": DecisionTreeClassifier(random_state=0),
    "random forest": RandomForestClassifier(n_estimators=200, random_state=0),
    "grad. boosting": HistGradientBoostingClassifier(random_state=0),
}
for name, model in models.items():
    start = time.time()
    s = cross_val_score(model, X, y, cv=5)
    print(f"{name:15} {s.mean():.3f} +/- {s.std():.3f}   ({time.time() - start:.1f}s)")
`,
      },
    },
    {
      id: 's-svm',
      title: 'Support vector machines',
      summary: 'Draw the boundary with the widest possible street between classes.',
      guide: { label: 'User Guide 1.4: Support Vector Machines', url: SK + 'modules/svm.html' },
      intuition: md`Many lines can separate two groups of points. An SVM picks the one with the **widest margin**: the widest empty street between the classes. Only the points on the edge of the street matter; they are the **support vectors**.

When the classes can't be separated by a straight line, the **kernel trick** compares points with a similarity function instead of a dot product. It acts as if the data lived in a much higher dimension where a straight separator exists, without ever computing that space.

The parameter $C$ trades margin width against training mistakes.`,
      math: md`Linear decision function: $f(x) = w^\top x + b$, predict the sign.

Hard-margin SVM (labels $y_i \in \{-1, +1\}$):

$$\min_{w, b} \ \tfrac{1}{2}\lVert w \rVert^2 \quad \text{subject to} \quad y_i (w^\top x_i + b) \geq 1$$

The margin width is $2 / \lVert w \rVert$.

Soft margin with **hinge loss**:

$$\min_{w,b} \ \tfrac{1}{2}\lVert w \rVert^2 + C \sum_i \max\big(0,\ 1 - y_i (w^\top x_i + b)\big)$$

RBF kernel: $K(x, z) = \exp\!\big(-\gamma \lVert x - z \rVert^2\big)$.`,
      paper: {
        task: md`Points: positive $(2, 2)$ and $(3, 3)$; negative $(0, 0)$ and $(1, 1)$. Candidate boundary $w = (1, 1)$, $b = -3$.

1. Compute $y_i (w^\top x_i + b)$ for each point. Is every constraint satisfied?
2. Which points are support vectors? How wide is the margin?
3. Compute the RBF kernel between $(0,0)$ and $(1,1)$ with $\gamma = 0.5$.`,
        solution: md`1. $(2,2)$: $+1 \cdot (4 - 3) = 1$. $(3,3)$: $+1 \cdot 3 = 3$. $(0,0)$: $-1 \cdot (-3) = 3$. $(1,1)$: $-1 \cdot (-1) = 1$. All are $\geq 1$, so yes.
2. The points with value exactly 1: $(2,2)$ and $(1,1)$. Margin $= 2 / \sqrt{2} \approx 1.414$.
3. $\lVert x - z \rVert^2 = 2$, so $K = e^{-1} \approx 0.368$.`,
      },
      code: {
        task: md`Train a linear SVM yourself by subgradient descent on the hinge loss. Compare the accuracy with «LinearSVC» on scaled breast cancer data.`,
        starter: md`import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

X, y01 = load_breast_cancer(return_X_y=True)
X = StandardScaler().fit_transform(X)
y = np.where(y01 == 1, 1, -1)          # SVM uses labels -1 / +1

w, b, C, lr = np.zeros(X.shape[1]), 0.0, 1.0, 0.001
for epoch in range(300):
    margins = y * (X @ w + b)
    violated = margins < 1
    grad_w = w - C * (y[violated, None] * X[violated]).sum(axis=0)
    grad_b = -C * y[violated].sum()
    w -= lr * grad_w
    b -= lr * grad_b

print("support-ish points (margin <= 1):", np.sum(y * (X @ w + b) <= 1))
print("accuracy mine   :", np.mean(np.sign(X @ w + b) == y).round(3))
print("accuracy sklearn:", round(LinearSVC(C=1.0).fit(X, y).score(X, y), 3))
`,
      },
      real: {
        task: md`Classify handwritten digits with an RBF-kernel SVM. Try a few values of $C$ and $\gamma$ and see how accuracy and the number of support vectors change.`,
        starter: md`from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

X, y = load_digits(return_X_y=True)
X_tr, X_te, y_tr, y_te = train_test_split(X, y, random_state=0)

for C in [0.1, 1, 10]:
    for gamma in [0.001, 0.01, 0.1]:
        model = make_pipeline(StandardScaler(), SVC(C=C, gamma=gamma)).fit(X_tr, y_tr)
        n_sv = model[-1].n_support_.sum()
        print(f"C={C:<4} gamma={gamma:<6} test acc={model.score(X_te, y_te):.3f}  support vectors={n_sv}")
`,
      },
    },
    {
      id: 's-tuning',
      title: 'Hyperparameter tuning',
      summary: 'Choosing settings like k, alpha or C systematically instead of guessing.',
      guide: { label: 'User Guide 3.2: Tuning hyper-parameters', url: SK + 'modules/grid_search.html' },
      intuition: md`Parameters (weights) are learned from data. **Hyperparameters** (like $k$ in k-NN, $\alpha$ in Ridge, $C$ in SVM) are chosen by you before training.

**Grid search** tries every combination from lists you give it and keeps the one with the best cross-validation score. **Randomized search** tries a fixed number of random combinations, which is often just as good and much faster when there are many hyperparameters.

Golden rule: tune with cross-validation on the training data, and touch the final test set only once at the end.`,
      math: md`Pick the hyperparameters $\lambda$ that maximize the cross-validated score on the training data:

$$\lambda^* = \arg\max_{\lambda \in \Lambda} \ \frac{1}{k} \sum_{m=1}^{k} s_m(\lambda)$$

A grid over hyperparameters with $n_1, n_2, \dots$ values has $\prod_i n_i$ combinations, and each needs $k$ fits, plus one final refit:

$$\text{fits} = k \cdot \prod_i n_i + 1$$`,
      paper: {
        task: md`You grid-search an SVM with $C \in \{0.1, 1, 10\}$ and $\gamma \in \{0.01, 0.1, 1, 10\}$ using 5-fold CV.

1. How many models are trained in total?
2. If one fit takes 2 seconds, how long does it take? How long if you add a third hyperparameter with 5 values?`,
        solution: md`1. $3 \cdot 4 = 12$ combinations, $12 \cdot 5 = 60$ fits, plus 1 refit = **61**.
2. About 2 minutes. With a third hyperparameter: $60 \cdot 5 + 1 = 301$ fits, about 10 minutes. Grids grow multiplicatively, which is why randomized search exists.`,
      },
      code: {
        task: md`Write grid search by hand with two nested loops and «cross_val_score». Then check that «GridSearchCV» finds the same best combination.`,
        starter: md`import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

X, y = load_breast_cancer(return_X_y=True)
Cs, gammas = [0.1, 1, 10], [0.001, 0.01, 0.1]

best = (None, None, -np.inf)
for C in Cs:
    for g in gammas:
        pipe = make_pipeline(StandardScaler(), SVC(C=C, gamma=g))
        score = cross_val_score(pipe, X, y, cv=5).mean()
        print(f"C={C:<4} gamma={g:<6} {score:.4f}")
        if score > best[2]:
            best = (C, g, score)
print("mine   : best C=%s gamma=%s score=%.4f" % best)

grid = GridSearchCV(make_pipeline(StandardScaler(), SVC()),
                    {"svc__C": Cs, "svc__gamma": gammas}, cv=5).fit(X, y)
print("sklearn:", grid.best_params_, round(grid.best_score_, 4))
`,
      },
      real: {
        task: md`Do it properly: split off a test set, tune a pipeline on the training part with «RandomizedSearchCV», then report the test score once. Compare the CV score and the test score.`,
        starter: md`from scipy.stats import loguniform
from sklearn.datasets import load_digits
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

X, y = load_digits(return_X_y=True)
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, stratify=y, random_state=0)

search = RandomizedSearchCV(
    make_pipeline(StandardScaler(), SVC()),
    {"svc__C": loguniform(1e-2, 1e2), "svc__gamma": loguniform(1e-4, 1e-1)},
    n_iter=15, cv=5, random_state=0,
).fit(X_tr, y_tr)

print("best params:", search.best_params_)
print("CV score   :", round(search.best_score_, 4))
print("test score :", round(search.score(X_te, y_te), 4), "(look at this only once!)")
`,
      },
    },
  ],
}
