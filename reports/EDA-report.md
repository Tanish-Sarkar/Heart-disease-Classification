## EDA Summary

**Dataset:** Loaded from `data/raw/heart.csv`. Basic checks (head, info, describe, shape, nulls) were performed.

**Missing values:** `df.isnull().sum()` and `df.isna()` were run; no missing-value handling is shown (dataset appears complete).

**Target balance:** The target class distribution is visualized with counts and percentages to assess balance before modeling.

**Univariate insights:** Key numeric features (e.g., `age`) are shown with smoothed histograms (KDE) and mean annotations to expose skew, spread, and obvious outliers.

**Correlation structure:** A masked correlation heatmap (upper triangle removed) with a divergent colormap highlights positive and negative relationships; useful to detect multicollinearity.

**Pairwise relationships:** A curated pairplot for selected numeric features (`age`, `trestbps`, `chol`, `thalach`, `oldpeak`, `thal`) shows how target classes separate across feature pairs.

**Visual improvements applied:** Seaborn `whitegrid` theme, higher DPI for crisper figures, annotated countplot (counts + %), KDE-enhanced histograms with mean lines, masked heatmap with `vlag` palette, and focused pairplot.

**High-level findings:**
- Some features show clearer separation by target in pairwise views — promising for predictive models.
- Correlation clusters indicate groups of related features; consider multicollinearity when using linear models.
- If class imbalance is non-trivial, use stratified evaluation or class-weighting / resampling.
