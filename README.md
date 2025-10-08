The repository contains the code used in my dissertation thesis.  
It includes separate Jupyter notebooks for each architecture and each task type (**classification vs. regression**). 


### Machine Learning Architectures
Two machine learning architectures are compared:  
- A **baseline** derived from the ARMED framework (Nguyen et al., 2023)  
- An **improved version** proposed in this work, using a **Kolmogorov–Arnold Network (KAN) backbone**  

**Citation for ARMED:**  
Nguyen, K. P., Treacher, A. H., & Montillo, A. A. (2023). *Adversarially-regularized mixed effects deep learning (ARMED) models improve interpretability, performance, and generalization on clustered (non-iid) data.* IEEE Transactions on Pattern Analysis and Machine Intelligence, 45(7), 8081–8093. https://doi.org/10.1109/TPAMI.2022.3228874  

### Statistical Baselines
In addition to ML models, we also implemented **Bayesian G/LMMs** as statistical baselines for both classification and regression tasks.  
- Implemented using the **`MCMCglmm`** and **`nlme`** packages in R  
- Scripts for these models are provided in separate `.R` files  

! The dataset is not included, as it is still unpublished.  

### GUI Application
A basic **GUI application** written in raw Python is provided in the dedicated folder.  
- Runs the newly proposed models  
- Includes a `data/` subfolder with a **sample dataset** for demonstration purposes  

⚠️ The main dataset used in the thesis is not included, as it is still unpublished. 

# A Mixed-Effects Machine Learning Approach for Forecasting Mental Health in Student Populations  

This thesis develops and evaluates a **mixed-effects machine learning architecture** for forecasting outcomes in clustered, high-frequency mental health data.  

---

## Motivation  
Conventional ML models assume data points are **independent and identically distributed (i.i.d.)**, but intensive longitudinal mental health data violate this assumption:  
- Observations are **clustered** (repeated measures within individuals).  
- Measurements occur at **irregular time lags**.  
- Data are **unbalanced** across clusters.  
- Observations are **correlated** within clusters.  

---

## Proposed Architecture  
The architecture is inspired by **linear mixed-effects models (LMM/GLMM)** and extends them with deep learning:  
1. **Fixed effects** – capture population-level trends.  
2. **Random effects** – provide cluster-specific adjustments.  
3. **Attention-based temporal decay** – predicts future outcomes from lagged values with variable time horizons.  
4. **Kolmogorov–Arnold Network (KAN) backbone** – spline-based approximator instead of standard weights and biases.  

---

## Baselines and Benchmarks  
- **Statistical baseline**: (Generalized) Linear Mixed Models (LMM/GLMM).  
- **ML benchmarks**: standard architectures including MLP.  

---

## Experiments  
- **Three datasets** focused on forecasting mental health outcomes with different temporal structures:  
  - Long-term trajectories (weekly or longer lags).  
  - Short-term intra-day behavioral incidents.  
  - Continuous depression outcomes measured multiple times per day.  

- **Evaluation setups**:  
  - Forecasting outcomes for **new clusters** (generalization to unseen individuals).  
  - Forecasting **future time points** for previously observed clusters.  

- **Ablation studies**:  
  - Removed architectural components (attention head, decomposition of fixed/random effects, backbone type).  
  - Compared **KAN vs MLP backbone**.  

---

## Results (Technical Summary)  
- The proposed architecture outperformed baselines on the **long-term dataset**, especially in generalizing to unseen clusters and new time points.  
- On short-term datasets, results were mixed, with smaller differences between models.  
- **Ablation findings**:  
  - All components generally improved performance in two datasets.  
  - Attention head negatively impacted performance in one dataset.  
  - Backbone choice (KAN vs MLP) yielded dataset-dependent effects.  

---

## Conclusion  
Mixed-effects inspired ML architectures are promising for **forecasting in clustered, irregular, and correlated data**, particularly in mental health research. Further work is needed to:  
- Refine conditions under which different architectural components help or hinder performance.  
- Explore robustness across datasets with varying temporal structures.  
