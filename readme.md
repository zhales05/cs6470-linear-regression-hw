# Testing

### Running Tests

```
 pytest -q
```

---

# Deliverables Checklist

### Implementation

- [ ] **Main** class

  - [x] load data succesfully
  - [ ] split into training and testing data
  - [ ] how to call normally

- [x] **BaseLinearRegression** class

  - [x] `fit`, `predict`, `score` methods
  - [x] `coef_`, `intercept_`, `fit_intercept` attributes

- [ ] **LinearRegressionGD** class (Gradient Descent)

  - [ ] Supports `learning_rate`, `max_iter`, `batch_size`, `random_state`
  - [ ] Uses SSE cost function
  - [ ] Tracks `cost_history_`
  - [ ] fit
    - [x] handle fit_intercept = true
    - [ ] how to use mini batches

- [ ] **LinearRegressionNE** class (Normal Equation)
  - [ ] Implements closed-form solution
  - [ ] Sets `coef_` and `intercept_` correctly

---

### Driver Script (`main.py`)

- [ ] Loads dataset (`housing_data.csv`)
- [ ] Trains GD, NE, and sklearn’s LinearRegression models
- [ ] Saves metrics to `outputs/metrics.csv`
- [ ] Saves plots:
  - [ ] Actual vs. Predicted (for GD, NE, sklearn)
  - [ ] Cost history (for GD)

---

### Results & Comparison

- [ ] Report **coefficients** and **intercept**
- [ ] Compute **R²** and **MSE**
- [ ] Measure and report **fit time**
- [ ] Compare results against sklearn

---

### Analysis

- [ ] Effect of **learning rate** on convergence
- [ ] Effect of **batch size** (full batch vs. mini-batch GD)
- [ ] Discussion of **cost function choice** (SSE vs. MSE)
- [ ] Comparison of **coefficients** (your models vs. sklearn)
- [ ] Reflection on **implementation details** and challenges

---

### Documentation

- [ ] **README.md** includes:
  - [ ] Setup instructions (env + requirements)
  - [ ] How to run `main.py` with examples
  - [ ] Description of outputs (plots + CSV)
  - [ ] This deliverables checklist (completed)
