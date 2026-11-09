# ML Playground

A study workbench for learning machine learning with the
[scikit-learn User Guide](https://scikit-learn.org/stable/user_guide.html),
built around one repeating **study loop**:

1. **Intuition**: what problem does this solve? (with interactive playgrounds for least squares, gradient descent, k-NN and k-means)
2. **Math**: the formulas, rendered with KaTeX, with every symbol explained
3. **Paper**: a tiny example to calculate by hand, with a hidden worked solution
4. **From scratch**: implement the algorithm in NumPy and compare with scikit-learn
5. **Real data**: apply it to a real dataset and interpret the result

Python runs **directly in the browser** (Pyodide in a Web Worker, with NumPy,
scikit-learn, SciPy and matplotlib). Plots appear under the code. Nothing to install.

## The curriculum

| Phase | Topics |
|---|---|
| 0. Math foundations | linear algebra, calculus and gradients, probability and statistics, optimization |
| 1. The workflow | X and y, scaling and encoding, missing values, pipelines, cross-validation, metrics, data leakage |
| 2. Supervised learning | least squares, gradient descent, Ridge and Lasso, logistic regression, k-NN, Naive Bayes, decision trees, ensembles, SVMs, tuning |
| 3. Unsupervised learning | k-means, PCA, DBSCAN, outlier detection |
| 4. Trusting models | validation and learning curves, inspection, saving models |

28 topics, 140 steps. Progress, notes and edited code are saved in your browser.
Use **Export progress** in the sidebar to back them up or move them to another device.

## Run it locally

```bash
npm install
npm run dev
```

Then open the address it prints (usually http://localhost:5173).

## Publish it with GitHub Pages

A ready-made workflow is in `deploy/github-pages.yml`. To turn it on (once):

1. On GitHub, open **Add file → Create new file** and name it `.github/workflows/deploy.yml`.
2. Paste the contents of `deploy/github-pages.yml` and commit.
3. Go to **Settings → Pages → Build and deployment → Source** and choose **GitHub Actions**.

After that, every push to `main` rebuilds the site at `https://<your-user>.github.io/MLplayground/`.

## Exercises that need your computer

Two real-data exercises (20 Newsgroups and Titanic) download datasets, which the
browser sandbox can't do. Copy their code and run it in a virtual environment:

```bash
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate
pip install scikit-learn pandas matplotlib
python exercise.py
```

## Project structure

```
src/
  content/        the curriculum: one file per phase, each topic has all five steps
  components/     sidebar, loop track, step content, code runner, markup renderer
  widgets/        interactive playgrounds (least squares, gradient descent, k-NN, k-means)
  lib/            progress storage and the Pyodide worker
```

To add or edit a topic, change the matching file in `src/content/`. Text uses a small
markup: `**bold**`, `«code»`, `$inline math$`, `$$display math$$`, `- ` lists and `| tables |`.
