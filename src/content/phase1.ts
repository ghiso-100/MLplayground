import type { Phase } from '../types'

const md = String.raw
const SK = 'https://scikit-learn.org/stable/'

export const phase1: Phase = {
  id: 'p1',
  title: 'The workflow',
  goal: 'How every ML project is set up: data, preprocessing, pipelines, honest evaluation, and the mistakes that fool beginners.',
  weeks: '2–3 weeks',
  topics: [
    {
      id: 'w-data',
      title: 'Datasets: X and y',
      summary: 'Samples, features, targets, and why you always keep some data hidden.',
      guide: { label: 'User Guide 9.1: Toy datasets', url: SK + 'datasets/toy_dataset.html' },
      intuition: md`Every scikit-learn problem has the same shape. **X** is the table of inputs (one row per example, one column per feature). **y** is what you want to predict.

If y is a number (a price, a temperature), it's **regression**. If y is a category (spam or not, which digit), it's **classification**.

A model that memorizes its training data is useless, just like a student who memorizes past exam answers. So you hide part of the data as a **test set** and only look at it at the very end.`,
      math: md`$$X \in \mathbb{R}^{n \times p}, \qquad y \in \mathbb{R}^{n} \text{ (regression) or } y \in \{c_1, \dots, c_K\}^n \text{ (classification)}$$

A **train/test split** partitions the row indices:

$$\{1, \dots, n\} = I_{\text{train}} \cup I_{\text{test}}, \qquad I_{\text{train}} \cap I_{\text{test}} = \emptyset$$

Typical split: 75–80% train, 20–25% test. **Stratified** splitting keeps the class proportions the same in both parts.`,
      paper: {
        task: md`A real estate table has columns: «size_m2», «rooms», «year_built», «district», «price». It has 1,200 rows.

1. What are $n$, $p$, X and y if you want to predict price?
2. Regression or classification? What if you predicted "sold within 30 days: yes/no" instead?
3. With a 75/25 split, how many rows end up in each set?`,
        solution: md`1. $n = 1200$, $p = 4$. X holds «size_m2», «rooms», «year_built», «district»; y is «price».
2. Price is a number, so **regression**. "Sold within 30 days" is yes/no, so **classification**.
3. 900 training rows and 300 test rows.`,
      },
      code: {
        task: md`Write your own «train_test_split»: shuffle the row indices with a random permutation and cut them. Make sure the same random seed gives the same split.`,
        starter: md`import numpy as np

def my_train_test_split(X, y, test_size=0.25, seed=42):
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(X))
    n_test = int(round(len(X) * test_size))
    test_idx, train_idx = idx[:n_test], idx[n_test:]
    return X[train_idx], X[test_idx], y[train_idx], y[test_idx]

X = np.arange(20).reshape(10, 2)
y = np.arange(10)
X_tr, X_te, y_tr, y_te = my_train_test_split(X, y)
print("train y:", y_tr)
print("test  y:", y_te)
`,
      },
      real: {
        task: md`Explore the Iris dataset: shape, feature names, class names and class counts. Split it with «stratify=y» and confirm each class keeps its proportion in the test set.`,
        starter: md`import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

data = load_iris()
X, y = data.data, data.target
print("shape   :", X.shape)
print("features:", data.feature_names)
print("classes :", list(data.target_names), np.bincount(y))

X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, stratify=y, random_state=0)
print("train counts:", np.bincount(y_tr))
print("test counts :", np.bincount(y_te))
`,
      },
    },
    {
      id: 'w-preprocess',
      title: 'Scaling and encoding',
      summary: 'Putting features on the same scale and turning categories into numbers.',
      guide: { label: 'User Guide 8.3: Preprocessing data', url: SK + 'modules/preprocessing.html' },
      intuition: md`Many models compare features by distance or size. If one feature is "income in euros" (around 40,000) and another is "number of children" (around 2), the income completely drowns out the other one. **Standardization** puts every feature on a comparable scale.

Models also only understand numbers. A column like "color: red / green / blue" must be turned into numbers without inventing a fake order (blue is not "bigger" than red). **One-hot encoding** makes one 0/1 column per category.`,
      math: md`**Standardization** (StandardScaler):

$$z = \frac{x - \mu}{\sigma}$$

After it, each feature has mean 0 and standard deviation 1.

**Min-max scaling** (MinMaxScaler) maps to $[0, 1]$:

$$x' = \frac{x - x_{\min}}{x_{\max} - x_{\min}}$$

**One-hot encoding** of a category with $K$ values creates $K$ columns, with a 1 in the column of the observed category and 0 elsewhere.

Important: $\mu$ and $\sigma$ are computed on the **training set only**, then reused on the test set.`,
      paper: {
        task: md`1. Standardize the values $[10, 20, 30]$ (use the population standard deviation).
2. Min-max scale the same values.
3. One-hot encode the column [red, blue, red, green].`,
        solution: md`1. $\mu = 20$. $\sigma = \sqrt{(100 + 0 + 100)/3} = \sqrt{66.67} \approx 8.165$. So $z \approx [-1.225,\ 0,\ 1.225]$.
2. $[0,\ 0.5,\ 1]$.
3. With columns (blue, green, red): red → $[0,0,1]$, blue → $[1,0,0]$, red → $[0,0,1]$, green → $[0,1,0]$.`,
      },
      code: {
        task: md`Build your own «StandardScaler» class with «fit» and «transform». Fit on training data, transform the test data, and compare with scikit-learn.`,
        starter: md`import numpy as np
from sklearn.preprocessing import StandardScaler

class MyScaler:
    def fit(self, X):
        self.mean_ = X.mean(axis=0)
        self.scale_ = X.std(axis=0)
        return self
    def transform(self, X):
        return (X - self.mean_) / self.scale_

X_train = np.array([[10, 1], [20, 2], [30, 3]], dtype=float)
X_test = np.array([[25, 5]], dtype=float)

mine = MyScaler().fit(X_train)
ref = StandardScaler().fit(X_train)
print("mine   :", mine.transform(X_test))
print("sklearn:", ref.transform(X_test))
`,
      },
      real: {
        task: md`The wine dataset has features on very different scales («proline» is in the hundreds, «hue» is around 1). Train k-nearest neighbors with and without scaling and compare accuracy. Explain the difference in one sentence.`,
        starter: md`import numpy as np
from sklearn.datasets import load_wine
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

X, y = load_wine(return_X_y=True)
print("feature ranges (min, max) of first 5 columns:")
print(np.round(np.c_[X.min(axis=0), X.max(axis=0)][:5], 2))
print("proline range:", X[:, 12].min(), "-", X[:, 12].max())

raw = KNeighborsClassifier(n_neighbors=5)
scaled = make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=5))
print("accuracy without scaling:", cross_val_score(raw, X, y, cv=5).mean().round(3))
print("accuracy with scaling   :", cross_val_score(scaled, X, y, cv=5).mean().round(3))
`,
      },
    },
    {
      id: 'w-impute',
      title: 'Missing values',
      summary: 'Filling gaps in the data without fooling yourself.',
      guide: { label: 'User Guide 8.4: Imputation', url: SK + 'modules/impute.html' },
      intuition: md`Real data has holes: a customer skipped the age field, a sensor failed for an hour. Most models crash on missing values (NaN), so you **impute** them: fill in a reasonable guess.

The simplest guess is the column's mean or median. The **median** is safer when there are extreme values. Smarter imputers look at similar rows (k-nearest neighbors) or predict the missing value from other columns.

The fact that a value is missing can itself be informative, so sometimes you add a "was missing" flag column.`,
      math: md`Mean imputation for feature $j$, using only observed values $O_j$:

$$\hat{x}_{ij} = \frac{1}{|O_j|} \sum_{k \in O_j} x_{kj}$$

Median imputation uses the middle value of the sorted observed values instead.

KNN imputation averages the feature over the $k$ nearest rows (distance measured on the features that are present):

$$\hat{x}_{ij} = \frac{1}{k} \sum_{m \in N_k(i)} x_{mj}$$`,
      paper: {
        task: md`Ages: $[22,\ \text{NaN},\ 30,\ 35,\ \text{NaN},\ 80]$.

1. What value does mean imputation insert? Median imputation?
2. Which one would you pick here, and why?`,
        solution: md`1. Observed: 22, 30, 35, 80. Mean $= 167/4 = 41.75$. Median $= (30 + 35)/2 = 32.5$.
2. The **median**. The 80 pulls the mean up, so the mean is not a typical age in this group.`,
      },
      code: {
        task: md`Write a mean imputer and a median imputer with «np.nanmean» and «np.nanmedian». Compare with «SimpleImputer».`,
        starter: md`import numpy as np
from sklearn.impute import SimpleImputer

ages = np.array([[22], [np.nan], [30], [35], [np.nan], [80]])

def impute(X, how="mean"):
    X = X.copy()
    stat = np.nanmean(X, axis=0) if how == "mean" else np.nanmedian(X, axis=0)
    rows, cols = np.where(np.isnan(X))
    X[rows, cols] = stat[cols]
    return X

print("mine mean  :", impute(ages, "mean").ravel())
print("mine median:", impute(ages, "median").ravel())
print("sklearn    :", SimpleImputer(strategy="median").fit_transform(ages).ravel())
`,
      },
      real: {
        task: md`Take the diabetes data, hide 20% of the BMI values on purpose, then fill them with mean, median and KNN imputation. Because you know the true values, you can measure which imputer guesses best.`,
        starter: md`import numpy as np
from sklearn.datasets import load_diabetes
from sklearn.impute import SimpleImputer, KNNImputer

X, _ = load_diabetes(return_X_y=True)
rng = np.random.default_rng(0)
mask = rng.random(len(X)) < 0.2
X_missing = X.copy()
X_missing[mask, 2] = np.nan          # column 2 is BMI
truth = X[mask, 2]

imputers = {
    "mean": SimpleImputer(strategy="mean"),
    "median": SimpleImputer(strategy="median"),
    "knn (k=5)": KNNImputer(n_neighbors=5),
}
for name, imp in imputers.items():
    filled = imp.fit_transform(X_missing)[mask, 2]
    rmse = np.sqrt(np.mean((filled - truth) ** 2))
    print(f"{name:10} RMSE = {rmse:.4f}")
`,
      },
    },
    {
      id: 'w-pipeline',
      title: 'Pipelines',
      summary: 'Chaining preprocessing and a model so they always run together, in the right order.',
      guide: { label: 'User Guide 8.1: Pipelines', url: SK + 'modules/compose.html' },
      intuition: md`A pipeline is an assembly line: scale, then impute, then predict. You call «fit» once and every step is fitted in order, each one on the output of the previous step.

The real benefit is safety. Inside a pipeline, the scaler only ever learns from training data, even during cross-validation. Without a pipeline, it is very easy to scale the whole dataset first and accidentally leak information from the test set.`,
      math: md`A pipeline is a **function composition**:

$$f(x) = g\big(h(x)\big)$$

When fitting on training data $X_{\text{tr}}$:

- fit $h$ on $X_{\text{tr}}$
- compute $Z_{\text{tr}} = h(X_{\text{tr}})$
- fit $g$ on $(Z_{\text{tr}}, y_{\text{tr}})$

When predicting, the already-fitted $h$ is applied to new data, never re-fitted.`,
      paper: {
        task: md`Training values: $[1, 2, 3]$. A test value: $10$.

A pipeline has a StandardScaler followed by a model. What scaled value does the model see for the test point? What would go wrong if you fitted the scaler on all four values instead?`,
        solution: md`Fit on training: $\mu = 2$, $\sigma = \sqrt{2/3} \approx 0.816$. Test value: $(10 - 2)/0.816 \approx 9.8$.

If you fitted on all four values, the mean and spread would include the test point. The scaler would "know" about future data, which is a small **data leakage** and makes test scores look better than they really are.`,
      },
      code: {
        task: md`Write a tiny «MyPipeline» class that takes a list of steps, fits the transformers in order and then the final model. Compare its predictions with scikit-learn's «Pipeline».`,
        starter: md`import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

class MyPipeline:
    def __init__(self, steps):
        self.steps = steps
    def fit(self, X, y):
        for step in self.steps[:-1]:
            X = step.fit(X).transform(X)
        self.steps[-1].fit(X, y)
        return self
    def predict(self, X):
        for step in self.steps[:-1]:
            X = step.transform(X)
        return self.steps[-1].predict(X)

X, y = load_breast_cancer(return_X_y=True)
mine = MyPipeline([StandardScaler(), LogisticRegression(max_iter=1000)]).fit(X, y)
ref = Pipeline([("scale", StandardScaler()), ("clf", LogisticRegression(max_iter=1000))]).fit(X, y)
print("predictions identical:", np.array_equal(mine.predict(X), ref.predict(X)))
`,
      },
      real: {
        task: md`Build a pipeline for breast cancer diagnosis: scaling plus logistic regression. Evaluate it with 5-fold cross-validation and print the pipeline structure.`,
        starter: md`from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

X, y = load_breast_cancer(return_X_y=True)
pipe = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
print(pipe)

scores = cross_val_score(pipe, X, y, cv=5)
print("fold accuracies:", scores.round(3))
print("mean accuracy  :", scores.mean().round(3))
`,
      },
    },
    {
      id: 'w-cv',
      title: 'Cross-validation',
      summary: 'Why one train/test split can lie to you, and how k-fold fixes it.',
      guide: { label: 'User Guide 3.1: Cross-validation', url: SK + 'modules/cross_validation.html' },
      intuition: md`One test split is like judging a student from a single exam: they might have been lucky with the questions. **k-fold cross-validation** gives the model $k$ different exams.

Split the data into $k$ equal parts (folds). Train on $k-1$ parts, test on the remaining one, and repeat until every fold has been the test set once. The average score is more reliable, and the spread tells you how much luck was involved.`,
      math: md`With $k$ folds and a score $s_m$ measured on fold $m$:

$$\text{CV score} = \frac{1}{k} \sum_{m=1}^{k} s_m$$

Report the standard deviation of the $s_m$ too. It estimates how much the score depends on which data happened to be in the test set.

Each sample is used for testing exactly once and for training $k - 1$ times.`,
      paper: {
        task: md`You run 5-fold cross-validation and get accuracies $[0.80,\ 0.90,\ 0.85,\ 0.70,\ 0.95]$.

1. What is the CV score and the standard deviation?
2. If you had only done a single split and got the 0.95 fold, what would you have believed?`,
        solution: md`1. Mean $= 4.2/5 = 0.84$. Deviations: $-0.04,\ 0.06,\ 0.01,\ -0.14,\ 0.11$. Squares sum to $0.037$, so the variance is $0.0074$ and the standard deviation $\approx 0.086$.
2. That the model is 95% accurate, when 84% is more realistic. A single split can easily be off by 10 points.`,
      },
      code: {
        task: md`Generate the k-fold indices yourself, train and score a model on each fold, and compare your mean with «cross_val_score» using the same folds.`,
        starter: md`import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import KFold, cross_val_score
from sklearn.tree import DecisionTreeClassifier

X, y = load_iris(return_X_y=True)

def my_kfold(n, k, seed=0):
    idx = np.random.default_rng(seed).permutation(n)
    folds = np.array_split(idx, k)
    for m in range(k):
        test = folds[m]
        train = np.concatenate([folds[j] for j in range(k) if j != m])
        yield train, test

scores = []
for train, test in my_kfold(len(X), 5):
    model = DecisionTreeClassifier(random_state=0).fit(X[train], y[train])
    scores.append(model.score(X[test], y[test]))
print("my folds:", np.round(scores, 3), "mean", round(np.mean(scores), 3))

ref = cross_val_score(DecisionTreeClassifier(random_state=0), X, y,
                      cv=KFold(5, shuffle=True, random_state=0))
print("sklearn :", ref.round(3), "mean", ref.mean().round(3))
`,
      },
      real: {
        task: md`On the handwritten digits dataset, evaluate the same model with 20 different single train/test splits, then with 5-fold CV. Plot how much the single-split scores jump around.`,
        starter: md`import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.tree import DecisionTreeClassifier

X, y = load_digits(return_X_y=True)
model = DecisionTreeClassifier(random_state=0)

single = []
for seed in range(20):
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=seed)
    single.append(model.fit(X_tr, y_tr).score(X_te, y_te))

cv = cross_val_score(model, X, y, cv=5)
print("single splits: min %.3f  max %.3f" % (min(single), max(single)))
print("5-fold CV    : %.3f +/- %.3f" % (cv.mean(), cv.std()))

plt.plot(single, "o-", label="single split")
plt.axhline(cv.mean(), linestyle="--", label="5-fold mean")
plt.xlabel("random seed"); plt.ylabel("accuracy"); plt.legend()
plt.show()
`,
      },
    },
    {
      id: 'w-metrics',
      title: 'Metrics',
      summary: 'Accuracy, precision, recall, F1, MSE and R²: measuring what actually matters.',
      guide: { label: 'User Guide 3.4: Metrics and scoring', url: SK + 'modules/model_evaluation.html' },
      intuition: md`"99% accurate" sounds great until you learn that 99% of emails are not spam, so a model that never flags anything is also 99% accurate.

The **confusion matrix** counts the four outcomes: true positives, false positives, false negatives, true negatives. From it come **precision** (when I say yes, how often am I right?) and **recall** (of all real yes cases, how many did I find?).

Which one matters depends on the cost of mistakes. Missing a cancer (false negative) is much worse than an extra check-up (false positive), so medicine cares about recall.

For regression, **MSE** measures average squared error and **R²** says how much better you are than always predicting the mean.`,
      math: md`From the confusion matrix (TP, FP, FN, TN):

$$\text{Accuracy} = \frac{TP + TN}{TP + FP + FN + TN}$$

$$\text{Precision} = \frac{TP}{TP + FP}, \qquad \text{Recall} = \frac{TP}{TP + FN}$$

$$F_1 = 2 \cdot \frac{\text{Precision} \cdot \text{Recall}}{\text{Precision} + \text{Recall}}$$

For regression:

$$\text{MSE} = \frac{1}{n}\sum_i (y_i - \hat{y}_i)^2, \qquad R^2 = 1 - \frac{\sum_i (y_i - \hat{y}_i)^2}{\sum_i (y_i - \bar{y})^2}$$`,
      paper: {
        task: md`A fraud detector is tested on 1,000 transactions: TP = 40, FP = 10, FN = 20, TN = 930.

Compute accuracy, precision, recall and F1. Then: what accuracy would a model get that says "not fraud" every time?`,
        solution: md`- Accuracy $= 970/1000 = 0.97$
- Precision $= 40/50 = 0.80$
- Recall $= 40/60 \approx 0.667$
- $F_1 = 2 \cdot 0.8 \cdot 0.667 / 1.467 \approx 0.727$

There are 60 real frauds, so "never fraud" gets $940/1000 = 0.94$ accuracy while catching **zero** frauds. Accuracy hides that; recall shows it.`,
      },
      code: {
        task: md`Implement the confusion matrix counts, precision, recall, F1 and R² from scratch. Check each against «sklearn.metrics».`,
        starter: md`import numpy as np
from sklearn import metrics

y_true = np.array([1, 0, 1, 1, 0, 0, 1, 0, 1, 0])
y_pred = np.array([1, 0, 0, 1, 0, 1, 1, 0, 1, 0])

TP = np.sum((y_true == 1) & (y_pred == 1))
FP = np.sum((y_true == 0) & (y_pred == 1))
FN = np.sum((y_true == 1) & (y_pred == 0))
TN = np.sum((y_true == 0) & (y_pred == 0))
precision = TP / (TP + FP)
recall = TP / (TP + FN)
f1 = 2 * precision * recall / (precision + recall)

print("TP FP FN TN:", TP, FP, FN, TN)
print("precision %.3f  sklearn %.3f" % (precision, metrics.precision_score(y_true, y_pred)))
print("recall    %.3f  sklearn %.3f" % (recall, metrics.recall_score(y_true, y_pred)))
print("f1        %.3f  sklearn %.3f" % (f1, metrics.f1_score(y_true, y_pred)))

y = np.array([3.0, 5.0, 7.0]); y_hat = np.array([2.5, 5.5, 6.0])
r2 = 1 - np.sum((y - y_hat) ** 2) / np.sum((y - y.mean()) ** 2)
print("R2        %.3f  sklearn %.3f" % (r2, metrics.r2_score(y, y_hat)))
`,
      },
      real: {
        task: md`Train a classifier on the breast cancer data and look at the confusion matrix and classification report. Compare with a «DummyClassifier» that always predicts the most common class. Which number would you show a doctor?`,
        starter: md`from sklearn.datasets import load_breast_cancer
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

data = load_breast_cancer()
X_tr, X_te, y_tr, y_te = train_test_split(data.data, data.target, stratify=data.target, random_state=0)

dummy = DummyClassifier(strategy="most_frequent").fit(X_tr, y_tr)
model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000)).fit(X_tr, y_tr)

print("dummy accuracy:", round(dummy.score(X_te, y_te), 3))
print("model accuracy:", round(model.score(X_te, y_te), 3))
print(confusion_matrix(y_te, model.predict(X_te)))
print(classification_report(y_te, model.predict(X_te), target_names=data.target_names))
`,
      },
    },
    {
      id: 'w-pitfalls',
      title: 'Pitfalls and data leakage',
      summary: 'The mistakes that make a useless model look excellent. Read this one twice.',
      guide: { label: 'User Guide 12: Common pitfalls', url: SK + 'common_pitfalls.html' },
      intuition: md`**Data leakage** means information from the test set (or from the future) sneaks into training. The model then looks great in your notebook and fails in the real world.

Classic examples:

- scaling or selecting features on the full dataset before splitting
- a feature that is only known after the event you predict (e.g. "refund issued" when predicting fraud)
- the same patient appearing in both training and test data

The cure is simple to state: everything that learns from data, including preprocessing, must only see the training part. Pipelines make that automatic.`,
      math: md`An honest estimate of performance needs the test data to be independent of every fitted quantity. If a preprocessing step computes a statistic $\phi$ (a mean, a feature ranking) from the data, then:

$$\phi = \phi(X_{\text{train}}) \quad \text{(correct)} \qquad \text{vs.} \qquad \phi = \phi(X_{\text{train}} \cup X_{\text{test}}) \quad \text{(leak)}$$

With pure noise features and a balanced binary target, any honest accuracy must be close to $0.5$. A higher score on noise is proof of leakage.`,
      paper: {
        task: md`For each scenario, is there leakage? Why?

1. You standardize all data, then split into train and test.
2. To predict whether a customer will cancel, you use the feature "days since cancellation".
3. You pick the 20 features most correlated with y using all rows, then run cross-validation.
4. You fit a pipeline (scaler + model) inside cross-validation.`,
        solution: md`1. **Yes, small leak.** The mean and standard deviation include the test rows.
2. **Yes, severe.** That feature only exists after the event you want to predict.
3. **Yes, severe.** The feature choice already looked at the test folds. On noise data this alone can produce high fake accuracy.
4. **No.** The pipeline refits the scaler on the training folds only.`,
      },
      code: {
        task: md`Prove leakage with pure noise: 10,000 random features and a random target. Select the "best" features using all the data, then cross-validate. Watch the fake accuracy.`,
        starter: md`import numpy as np
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score

rng = np.random.default_rng(0)
X = rng.normal(size=(100, 10000))     # pure noise
y = rng.integers(0, 2, size=100)       # random labels

# WRONG: feature selection on all data, before CV
X_selected = SelectKBest(f_classif, k=20).fit_transform(X, y)
leaky = cross_val_score(LogisticRegression(), X_selected, y, cv=5).mean()
print("leaky accuracy on noise:", round(leaky, 3))
`,
      },
      real: {
        task: md`Fix the leak from the previous step by putting feature selection inside a pipeline. The accuracy should drop to about 0.5, which is the honest truth for random data.`,
        starter: md`import numpy as np
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import make_pipeline

rng = np.random.default_rng(0)
X = rng.normal(size=(100, 10000))
y = rng.integers(0, 2, size=100)

honest_pipe = make_pipeline(SelectKBest(f_classif, k=20), LogisticRegression())
honest = cross_val_score(honest_pipe, X, y, cv=5).mean()
print("honest accuracy on noise:", round(honest, 3))
`,
      },
    },
  ],
}
