import type { Phase } from '../types'

const md = String.raw
const SK = 'https://scikit-learn.org/stable/'

export const phase4: Phase = {
  id: 'p4',
  title: 'Understanding and trusting models',
  goal: 'Diagnosing over- and underfitting, explaining what a model learned, and saving it for later.',
  weeks: '2 weeks',
  topics: [
    {
      id: 'i-curves',
      title: 'Validation and learning curves',
      summary: 'Two plots that tell you whether your model is too simple, too complex, or short of data.',
      guide: { label: 'User Guide 3.5: Validation curves', url: SK + 'modules/learning_curve.html' },
      intuition: md`A **validation curve** varies one hyperparameter (say, polynomial degree) and plots training and validation scores. Too simple: both scores are bad (**underfitting**). Too complex: training is perfect but validation drops (**overfitting**). The sweet spot is in between.

A **learning curve** varies the amount of training data. If the two curves are still converging at the end, more data will help. If they have already met at a low score, you need a better model, not more data.`,
      math: md`The expected error of a model splits into three parts:

$$\mathbb{E}\big[(y - \hat{f}(x))^2\big] = \underbrace{\text{Bias}[\hat{f}(x)]^2}_{\text{too simple}} + \underbrace{\text{Var}[\hat{f}(x)]}_{\text{too sensitive}} + \underbrace{\sigma^2}_{\text{noise}}$$

- high bias: train and validation errors both high and close together
- high variance: train error low, validation error much higher
- $\sigma^2$ is the noise floor no model can beat`,
      paper: {
        task: md`Diagnose each model and suggest one fix:

| model | train accuracy | validation accuracy |
|---|---|---|
| A | 0.99 | 0.70 |
| B | 0.72 | 0.70 |
| C | 0.91 | 0.89 |`,
        solution: md`- **A: overfitting** (high variance). Fix: regularize, reduce depth or complexity, or get more data.
- **B: underfitting** (high bias). Fix: a more flexible model or better features.
- **C: healthy.** Small gap, good score. Tune a little or ship it.`,
      },
      code: {
        task: md`Build a validation curve by hand: fit polynomials of degree 1 to 15 to noisy sine data and plot training and validation error. Find the degree where validation error is lowest.`,
        starter: md`import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures

rng = np.random.default_rng(0)
x = np.sort(rng.uniform(0, 1, 40)).reshape(-1, 1)
y = np.sin(2 * np.pi * x).ravel() + rng.normal(0, 0.2, 40)
x_tr, x_va, y_tr, y_va = train_test_split(x, y, test_size=0.4, random_state=0)

degrees = range(1, 16)
train_err, val_err = [], []
for d in degrees:
    model = make_pipeline(PolynomialFeatures(d), LinearRegression()).fit(x_tr, y_tr)
    train_err.append(mean_squared_error(y_tr, model.predict(x_tr)))
    val_err.append(mean_squared_error(y_va, model.predict(x_va)))

print("best degree:", list(degrees)[int(np.argmin(val_err))])
plt.plot(degrees, train_err, "o-", label="train")
plt.plot(degrees, val_err, "o-", label="validation")
plt.yscale("log"); plt.xlabel("polynomial degree"); plt.ylabel("MSE"); plt.legend()
plt.show()
`,
      },
      real: {
        task: md`Draw a learning curve for an SVM on the digits data. Would collecting more handwriting samples improve the model?`,
        starter: md`import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.model_selection import learning_curve
from sklearn.svm import SVC

X, y = load_digits(return_X_y=True)
sizes, train_scores, val_scores = learning_curve(
    SVC(gamma=0.001), X, y, cv=5, train_sizes=np.linspace(0.1, 1.0, 6))

for n, tr, va in zip(sizes, train_scores.mean(axis=1), val_scores.mean(axis=1)):
    print(f"{n:5} samples: train {tr:.3f}  validation {va:.3f}")

plt.plot(sizes, train_scores.mean(axis=1), "o-", label="train")
plt.plot(sizes, val_scores.mean(axis=1), "o-", label="validation")
plt.xlabel("training samples"); plt.ylabel("accuracy"); plt.legend()
plt.show()
`,
      },
    },
    {
      id: 'i-inspection',
      title: 'Inspecting models',
      summary: 'Which features matter, and how does the prediction change when a feature changes?',
      guide: { label: 'User Guide 5: Inspection', url: SK + 'inspection.html' },
      intuition: md`A model that works but can't be explained is hard to trust. Two tools help with any model:

**Permutation importance**: shuffle one feature's column so it becomes useless, and measure how much the score drops. A big drop means the model relied on that feature.

**Partial dependence**: vary one feature across its range while keeping everything else as it is, and plot the average prediction. It shows the shape of the effect: rising, falling, flat, or with a threshold.

Watch out: when two features are strongly correlated, shuffling one can look harmless because the model leans on the other.`,
      math: md`**Permutation importance** of feature $j$, with baseline score $s$ and $K$ shuffles:

$$i_j = s - \frac{1}{K} \sum_{k=1}^{K} s_{k,j}$$

where $s_{k,j}$ is the score after the $k$-th random shuffle of column $j$.

**Partial dependence** of the prediction on feature $j$ at value $v$:

$$\text{PD}_j(v) = \frac{1}{n} \sum_{i=1}^{n} \hat{f}\big(x_i \text{ with } x_{ij} := v\big)$$`,
      paper: {
        task: md`A model has a baseline accuracy of 0.90. After shuffling:

- feature A three times: 0.70, 0.72, 0.68
- feature B three times: 0.89, 0.90, 0.88

Compute both importances. Does a low importance for B prove that B has no relationship with the target?`,
        solution: md`A: mean $= 0.70$, importance $= 0.90 - 0.70 = 0.20$. B: mean $= 0.89$, importance $= 0.01$.

**No.** B might be strongly related to the target but redundant with another feature the model uses instead. Importance describes this **model**, not the real world.`,
      },
      code: {
        task: md`Implement permutation importance by hand on a random forest and compare with «sklearn.inspection.permutation_importance».`,
        starter: md`import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.model_selection import train_test_split

data = load_breast_cancer()
X_tr, X_te, y_tr, y_te = train_test_split(data.data, data.target, random_state=0)
model = RandomForestClassifier(random_state=0).fit(X_tr, y_tr)
base = model.score(X_te, y_te)

rng = np.random.default_rng(0)
mine = []
for j in range(X_te.shape[1]):
    drops = []
    for _ in range(5):
        X_shuf = X_te.copy()
        X_shuf[:, j] = rng.permutation(X_shuf[:, j])
        drops.append(base - model.score(X_shuf, y_te))
    mine.append(np.mean(drops))

ref = permutation_importance(model, X_te, y_te, n_repeats=5, random_state=0)
for j in np.argsort(mine)[::-1][:5]:
    print(f"{data.feature_names[j]:22} mine {mine[j]:.4f}   sklearn {ref.importances_mean[j]:.4f}")
`,
      },
      real: {
        task: md`On the diabetes data, train gradient boosting and plot the partial dependence of disease progression on BMI and blood pressure. Describe each effect in one sentence a doctor would understand.`,
        starter: md`import matplotlib.pyplot as plt
from sklearn.datasets import load_diabetes
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.inspection import PartialDependenceDisplay, permutation_importance

data = load_diabetes(as_frame=True)
X, y = data.data, data.target
model = HistGradientBoostingRegressor(random_state=0).fit(X, y)

imp = permutation_importance(model, X, y, n_repeats=5, random_state=0)
for name, val in sorted(zip(X.columns, imp.importances_mean), key=lambda t: -t[1])[:5]:
    print(f"{name:4} {val:.3f}")

PartialDependenceDisplay.from_estimator(model, X, ["bmi", "bp"])
plt.show()
`,
      },
    },
    {
      id: 'i-persistence',
      title: 'Saving and loading models',
      summary: 'A trained model is just its learned numbers. Save them, load them, use them later.',
      guide: { label: 'User Guide 11: Model persistence', url: SK + 'model_persistence.html' },
      intuition: md`Training can take hours; using a model takes milliseconds. So you train once, save the result, and load it wherever you need predictions: a web app, a script, a colleague's laptop.

For a linear model, "the result" is literally a few numbers: the coefficients and the intercept. For a random forest it's thousands of tree nodes. scikit-learn usually saves the whole fitted object with **joblib**.

Two warnings from the guide: load models only from sources you trust (loading a pickle can run code), and use the same scikit-learn version for loading as for saving.`,
      math: md`A fitted linear model is fully described by its parameters:

$$\hat{f}(x) = w^\top x + b, \qquad \text{saved state} = \{w, b\}$$

A fitted pipeline also stores every preprocessing statistic, for example a scaler's $\mu$ and $\sigma$:

$$\hat{f}(x) = w^\top \left(\frac{x - \mu}{\sigma}\right) + b$$

Saving the model means saving all of $\{\mu, \sigma, w, b\}$.`,
      paper: {
        task: md`A saved pipeline contains: scaler mean $\mu = [50, 3]$, scale $\sigma = [10, 1]$, weights $w = [2, -1]$, intercept $b = 5$.

Compute the prediction for the new input $x = [70, 2]$ by hand.`,
        solution: md`Scale: $(70 - 50)/10 = 2$ and $(2 - 3)/1 = -1$.

Predict: $2 \cdot 2 + (-1)(-1) + 5 = 4 + 1 + 5 = 10$.

That is everything «pipeline.predict» does for this model.`,
      },
      code: {
        task: md`Save a fitted linear pipeline to JSON by hand (just the numbers), reload it, and rebuild the prediction with NumPy. Confirm it matches «pipeline.predict» exactly.`,
        starter: md`import json
import numpy as np
from sklearn.datasets import load_diabetes
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

X, y = load_diabetes(return_X_y=True)
pipe = make_pipeline(StandardScaler(), Ridge(alpha=1.0)).fit(X, y)

state = {
    "mean": pipe[0].mean_.tolist(), "scale": pipe[0].scale_.tolist(),
    "coef": pipe[1].coef_.tolist(), "intercept": float(pipe[1].intercept_),
}
with open("model.json", "w") as f:
    json.dump(state, f)

with open("model.json") as f:
    s = json.load(f)
pred = ((X[:5] - np.array(s["mean"])) / np.array(s["scale"])) @ np.array(s["coef"]) + s["intercept"]
print("from JSON :", np.round(pred, 3))
print("predict() :", np.round(pipe.predict(X[:5]), 3))
`,
      },
      real: {
        task: md`Save a full random forest pipeline with joblib, reload it as if in a new program, and check the predictions are identical. Record the scikit-learn version next to it, as the guide recommends.`,
        starter: md`import json
import joblib
import numpy as np
import sklearn
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

X, y = load_breast_cancer(return_X_y=True)
pipe = make_pipeline(StandardScaler(), RandomForestClassifier(random_state=0)).fit(X, y)

joblib.dump(pipe, "cancer_model.joblib")
with open("cancer_model.meta.json", "w") as f:
    json.dump({"sklearn_version": sklearn.__version__}, f)

loaded = joblib.load("cancer_model.joblib")        # only load files you trust
print("identical predictions:", np.array_equal(loaded.predict(X), pipe.predict(X)))
print("saved with scikit-learn", sklearn.__version__)
`,
      },
    },
  ],
}
