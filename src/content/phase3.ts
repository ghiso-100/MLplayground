import type { Phase } from '../types'

const md = String.raw
const SK = 'https://scikit-learn.org/stable/'

export const phase3: Phase = {
  id: 'p3',
  title: 'Unsupervised learning',
  goal: 'Finding structure without labels: groups, directions of variation, and things that don’t fit.',
  weeks: '3–4 weeks',
  topics: [
    {
      id: 'u-kmeans',
      title: 'K-means clustering',
      summary: 'Split data into k groups, each around its own center.',
      guide: { label: 'User Guide 2.3.2: K-means', url: SK + 'modules/clustering.html#k-means' },
      widget: 'kmeans',
      intuition: md`A shop wants to group its customers, but nobody has labeled them. K-means finds $k$ groups by itself with a simple loop:

- place $k$ centers somewhere
- assign every point to its nearest center
- move each center to the average of its points
- repeat until nothing changes

You must choose $k$ yourself, and results depend on the starting centers, so scikit-learn runs it several times. Step through the loop in the playground to watch the centers walk.`,
      math: md`K-means minimizes the **inertia**, the total squared distance from each point to its cluster center $\mu_c$:

$$J = \sum_{i=1}^{n} \lVert x_i - \mu_{c(i)} \rVert^2$$

Alternating two steps:

$$c(i) = \arg\min_c \lVert x_i - \mu_c \rVert^2 \qquad \text{(assign)}$$

$$\mu_c = \frac{1}{|S_c|} \sum_{i \in S_c} x_i \qquad \text{(update)}$$

Each step can only lower $J$, so the loop always stops, but possibly at a local minimum.`,
      paper: {
        task: md`1D points: $1, 2, 3, 10, 11, 12$. Run k-means with $k = 2$, starting centers at $1$ and $2$. Do assign and update steps until nothing changes.`,
        solution: md`Round 1, assign: 1 → center 1; 2, 3, 10, 11, 12 → center 2. Update: $\mu_1 = 1$, $\mu_2 = 38/5 = 7.6$.

Round 2, assign: 1, 2, 3 are closer to 1; 10, 11, 12 are closer to 7.6. Update: $\mu_1 = 2$, $\mu_2 = 11$.

Round 3: assignments don't change. Done: clusters $\{1, 2, 3\}$ and $\{10, 11, 12\}$, centers 2 and 11.`,
      },
      code: {
        task: md`Implement k-means in NumPy with the assign/update loop. Run it on generated blobs and plot the result next to scikit-learn's «KMeans».`,
        starter: md`import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs

X, _ = make_blobs(n_samples=300, centers=3, random_state=4)

def kmeans(X, k, iters=20, seed=0):
    rng = np.random.default_rng(seed)
    centers = X[rng.choice(len(X), k, replace=False)]
    for _ in range(iters):
        dist = ((X[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2)
        labels = dist.argmin(axis=1)
        centers = np.array([X[labels == c].mean(axis=0) for c in range(k)])
    inertia = ((X - centers[labels]) ** 2).sum()
    return labels, centers, inertia

labels, centers, inertia = kmeans(X, 3)
ref = KMeans(3, n_init=10, random_state=0).fit(X)
print("inertia mine: %.1f  sklearn: %.1f" % (inertia, ref.inertia_))

fig, ax = plt.subplots(1, 2, figsize=(9, 4))
ax[0].scatter(X[:, 0], X[:, 1], c=labels, s=12); ax[0].scatter(*centers.T, marker="x", s=150, c="k")
ax[0].set_title("mine")
ax[1].scatter(X[:, 0], X[:, 1], c=ref.labels_, s=12); ax[1].set_title("sklearn")
plt.show()
`,
      },
      real: {
        task: md`Cluster wines by their chemistry without looking at the labels. Use the elbow plot to choose $k$, then compare the clusters with the real grape varieties using the adjusted Rand index (1 = perfect match, 0 = random).`,
        starter: md`import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.datasets import load_wine
from sklearn.metrics import adjusted_rand_score
from sklearn.preprocessing import StandardScaler

X, y = load_wine(return_X_y=True)
X = StandardScaler().fit_transform(X)

ks = range(1, 9)
inertias = [KMeans(k, n_init=10, random_state=0).fit(X).inertia_ for k in ks]
plt.plot(list(ks), inertias, "o-"); plt.xlabel("k"); plt.ylabel("inertia"); plt.title("Elbow plot")
plt.show()

labels = KMeans(3, n_init=10, random_state=0).fit_predict(X)
print("agreement with true varieties (ARI):", round(adjusted_rand_score(y, labels), 3))
`,
      },
    },
    {
      id: 'u-pca',
      title: 'Principal component analysis',
      summary: 'Find the directions where the data varies most, and keep only those.',
      guide: { label: 'User Guide 2.5.1: PCA', url: SK + 'modules/decomposition.html#principal-component-analysis-pca' },
      intuition: md`Photograph a teapot: from the side you see its shape clearly, from straight above it's just a circle. PCA finds the "best camera angle" for your data, the directions where the points spread out most.

The first principal component is the direction of greatest spread, the second is the next best direction at a right angle to the first, and so on. Keeping only the first few compresses the data while losing as little information as possible, and lets you plot 64-dimensional data in 2D.`,
      math: md`Center the data ($X \leftarrow X - \bar{X}$) and compute the covariance matrix:

$$\Sigma = \frac{1}{n-1} X^\top X$$

Its **eigenvectors** $v_k$ are the principal directions and its **eigenvalues** $\lambda_k$ are the variances along them:

$$\Sigma v_k = \lambda_k v_k$$

Explained variance ratio of component $k$:

$$\frac{\lambda_k}{\sum_j \lambda_j}$$

Projection onto the first $m$ components: $Z = X V_m$.`,
      paper: {
        task: md`A centered dataset has covariance matrix

$$\Sigma = \begin{bmatrix} 2 & 1 \\ 1 & 2 \end{bmatrix}$$

1. Check that $v_1 = (1, 1)/\sqrt{2}$ and $v_2 = (1, -1)/\sqrt{2}$ are eigenvectors and find their eigenvalues.
2. What fraction of the variance does the first component explain?`,
        solution: md`1. $\Sigma (1, 1) = (3, 3) = 3 \cdot (1, 1)$, so $\lambda_1 = 3$. $\Sigma (1, -1) = (1, -1)$, so $\lambda_2 = 1$.
2. $3 / (3 + 1) = 75\%$. The data stretches along the diagonal $x = y$.`,
      },
      code: {
        task: md`Implement PCA with «np.linalg.eigh» on the covariance matrix. Compare explained variance ratios with scikit-learn. (Component signs may be flipped; that's normal, a direction and its opposite are the same axis.)`,
        starter: md`import numpy as np
from sklearn.datasets import load_iris
from sklearn.decomposition import PCA

X, _ = load_iris(return_X_y=True)
Xc = X - X.mean(axis=0)
cov = Xc.T @ Xc / (len(X) - 1)

eigvals, eigvecs = np.linalg.eigh(cov)          # ascending order
order = np.argsort(eigvals)[::-1]
eigvals, eigvecs = eigvals[order], eigvecs[:, order]

print("mine ratio   :", np.round(eigvals / eigvals.sum(), 4))
pca = PCA().fit(X)
print("sklearn ratio:", np.round(pca.explained_variance_ratio_, 4))
print("first component mine   :", np.round(eigvecs[:, 0], 3))
print("first component sklearn:", np.round(pca.components_[0], 3))
`,
      },
      real: {
        task: md`Compress the 64-pixel digit images to 2 dimensions and plot them colored by digit. Which digits overlap? Then check how many components you need to keep 90% of the variance.`,
        starter: md`import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.decomposition import PCA

X, y = load_digits(return_X_y=True)
Z = PCA(n_components=2).fit_transform(X)

full = PCA().fit(X)
cum = np.cumsum(full.explained_variance_ratio_)
print("components for 90% variance:", np.argmax(cum >= 0.9) + 1, "of 64")

fig, ax = plt.subplots(1, 2, figsize=(11, 4.5))
sc = ax[0].scatter(Z[:, 0], Z[:, 1], c=y, cmap="tab10", s=8)
fig.colorbar(sc, ax=ax[0]); ax[0].set_title("Digits in 2D")
ax[1].plot(cum); ax[1].axhline(0.9, linestyle="--")
ax[1].set_xlabel("components"); ax[1].set_ylabel("cumulative variance")
plt.show()
`,
      },
    },
    {
      id: 'u-dbscan',
      title: 'DBSCAN',
      summary: 'Clusters as dense regions, with noise left out.',
      guide: { label: 'User Guide 2.3.7: DBSCAN', url: SK + 'modules/clustering.html#dbscan' },
      intuition: md`K-means assumes round blobs and forces every point into a cluster. Real data is messier: GPS points along winding roads, galaxies in strings, and lonely outliers that belong nowhere.

**DBSCAN** defines a cluster as a crowded region. A point with enough neighbors within distance $\varepsilon$ is a **core point**. Core points that are close to each other chain together into one cluster, of any shape. Points that are not near any core point are labeled **noise**.

You don't choose the number of clusters; you choose what "crowded" means.`,
      math: md`The $\varepsilon$-neighborhood of a point:

$$N_\varepsilon(x) = \{\, x' : d(x, x') \leq \varepsilon \,\}$$

- **core point**: $|N_\varepsilon(x)| \geq \text{min\_samples}$ (the point counts itself)
- **border point**: not core, but inside the neighborhood of a core point
- **noise**: neither

A cluster is a maximal set of core points connected through overlapping neighborhoods, plus their border points.`,
      paper: {
        task: md`1D points: $1, 2, 3, 8, 9, 20$. Use $\varepsilon = 1.5$ and min_samples $= 2$.

Find each point's neighborhood, mark core points, and list the clusters and noise.`,
        solution: md`- 1: $\{1, 2\}$ → core
- 2: $\{1, 2, 3\}$ → core
- 3: $\{2, 3\}$ → core
- 8: $\{8, 9\}$ → core
- 9: $\{8, 9\}$ → core
- 20: $\{20\}$ → not core, not near a core point → **noise**

Clusters: $\{1, 2, 3\}$ and $\{8, 9\}$. Point 20 is noise.`,
      },
      code: {
        task: md`Implement a simple DBSCAN: find neighbors, mark core points, and grow clusters with a queue. Compare labels with scikit-learn on the moons dataset.`,
        starter: md`import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.datasets import make_moons
from sklearn.metrics import adjusted_rand_score

X, _ = make_moons(n_samples=300, noise=0.06, random_state=0)

def dbscan(X, eps, min_samples):
    dist = np.sqrt(((X[:, None] - X[None]) ** 2).sum(axis=2))
    neighbors = [np.where(row <= eps)[0] for row in dist]
    core = np.array([len(n) >= min_samples for n in neighbors])
    labels = np.full(len(X), -1)
    cluster = 0
    for i in range(len(X)):
        if not core[i] or labels[i] != -1:
            continue
        queue = [i]; labels[i] = cluster
        while queue:
            j = queue.pop()
            if core[j]:
                for nb in neighbors[j]:
                    if labels[nb] == -1:
                        labels[nb] = cluster; queue.append(nb)
        cluster += 1
    return labels

mine = dbscan(X, eps=0.2, min_samples=5)
ref = DBSCAN(eps=0.2, min_samples=5).fit_predict(X)
print("clusters found:", len(set(mine) - {-1}), " noise points:", np.sum(mine == -1))
print("agreement with sklearn (ARI):", round(adjusted_rand_score(ref, mine), 3))
`,
      },
      real: {
        task: md`Compare k-means and DBSCAN on two interleaving half-moons (think: two winding roads in GPS data). Plot both results. Which one finds the real shapes, and why?`,
        starter: md`import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN, KMeans
from sklearn.datasets import make_moons

X, _ = make_moons(n_samples=400, noise=0.07, random_state=1)
km = KMeans(2, n_init=10, random_state=0).fit_predict(X)
db = DBSCAN(eps=0.18, min_samples=6).fit_predict(X)

fig, ax = plt.subplots(1, 2, figsize=(10, 4))
ax[0].scatter(X[:, 0], X[:, 1], c=km, cmap="coolwarm", s=10); ax[0].set_title("k-means")
ax[1].scatter(X[:, 0], X[:, 1], c=db, cmap="coolwarm", s=10); ax[1].set_title("DBSCAN (noise = -1)")
plt.show()
`,
      },
    },
    {
      id: 'u-outliers',
      title: 'Outlier and anomaly detection',
      summary: 'Spotting the transactions, sensors or patients that don’t look like the rest.',
      guide: { label: 'User Guide 2.7: Novelty and outlier detection', url: SK + 'modules/outlier_detection.html' },
      intuition: md`Fraud, machine failures and data entry errors have one thing in common: they look different from normal data. Outlier detection learns what "normal" looks like and flags what doesn't fit.

The simplest method is the z-score: flag values far from the mean. But a big outlier inflates the standard deviation and can hide itself (this is called **masking**). Robust methods use the median instead.

**Isolation Forest** has a neat idea: build random splits. Normal points sit in crowds and need many splits to be isolated; outliers are isolated after just a few.`,
      math: md`z-score rule: flag $x$ if $|z| > 3$ where $z = (x - \mu)/\sigma$.

Robust version with the median and the **median absolute deviation**:

$$\text{MAD} = \operatorname{median}(|x_i - \operatorname{median}(x)|), \qquad z_{\text{robust}} = \frac{0.6745\,(x - \operatorname{median}(x))}{\text{MAD}}$$

IQR rule: flag values below $Q_1 - 1.5\,\text{IQR}$ or above $Q_3 + 1.5\,\text{IQR}$.

Isolation Forest score from the average path length $E[h(x)]$:

$$s(x) = 2^{-E[h(x)] / c(n)}$$

Short paths give scores near 1, meaning "anomaly".`,
      paper: {
        task: md`Card payments (€): $20, 25, 22, 30, 24, 500$.

1. Compute the mean, standard deviation (population) and the z-score of 500. Is it flagged by the $|z| > 3$ rule?
2. Compute the median. Why is the median-based view more useful here?`,
        solution: md`1. Mean $= 621/6 = 103.5$. Squared deviations sum to about $188{,}711.5$, variance $\approx 31{,}452$, standard deviation $\approx 177.3$. z of 500 $= 396.5 / 177.3 \approx 2.24$. **Not flagged!** The outlier inflated the standard deviation and hid itself.
2. Median $= (24 + 25)/2 = 24.5$. The median ignores the 500, so 500 is obviously far from the typical payment. Robust statistics resist masking.`,
      },
      code: {
        task: md`Implement three detectors on the payment data: the z-score rule, the robust (MAD) z-score and the IQR rule. Which ones catch the 500?`,
        starter: md`import numpy as np

x = np.array([20, 25, 22, 30, 24, 500], dtype=float)

z = (x - x.mean()) / x.std()
print("z-scores    :", np.round(z, 2), "-> flagged:", x[np.abs(z) > 3])

med = np.median(x)
mad = np.median(np.abs(x - med))
z_robust = 0.6745 * (x - med) / mad
print("robust z    :", np.round(z_robust, 2), "-> flagged:", x[np.abs(z_robust) > 3.5])

q1, q3 = np.percentile(x, [25, 75])
iqr = q3 - q1
flag = (x < q1 - 1.5 * iqr) | (x > q3 + 1.5 * iqr)
print("IQR fences  :", q1 - 1.5 * iqr, q3 + 1.5 * iqr, "-> flagged:", x[flag])
`,
      },
      real: {
        task: md`Simulate a screening setting: take all benign tumors plus only 10 malignant ones, and see whether Isolation Forest flags the malignant cases as anomalies without ever being told the labels.`,
        starter: md`import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

X, y = load_breast_cancer(return_X_y=True)     # 0 = malignant, 1 = benign
rng = np.random.default_rng(0)
malignant = rng.choice(np.where(y == 0)[0], 10, replace=False)
idx = np.r_[np.where(y == 1)[0], malignant]
X_s, is_malignant = StandardScaler().fit_transform(X[idx]), (y[idx] == 0)

iso = IsolationForest(contamination=0.03, random_state=0).fit(X_s)
flagged = iso.predict(X_s) == -1
print("points flagged:", flagged.sum())
print("malignant caught:", (flagged & is_malignant).sum(), "of", is_malignant.sum())
print("benign flagged by mistake:", (flagged & ~is_malignant).sum())
`,
      },
    },
  ],
}
