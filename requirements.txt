# ===== SYSTEM DEPENDENCY (Graphviz) =====
# To render proper causal graphs with directed arrows:
#
# 1. Download the Graphviz installer:
#    👉 https://graphviz.org/download/
#
# 2. Choose the Windows `.exe` under Stable Releases and install it.
#
# 3. During installation, make sure to check:
#       "Add Graphviz to the system PATH for all users"
#
# 4. After installation, verify it's available by running in PowerShell:
#       dot -V
#    You should see something like:
#       dot - graphviz version 2.49.0 (20210306.1121)
#
# 5. Then you can install the Python requirements below:

# ===== PYTHON DEPENDENCIES =====
# Recommended Python version: 3.11
#
#  Install requirements:
# PS C:\Users\...\causal_app> py -3.11 -m pip install -r requirements.txt
#
#  Run the app:
# PS C:\Users\...\causal_app> py -3.11 -m streamlit run app.py

streamlit
pandas
dill
lingam
matplotlib
networkx
git+https://github.com/cmu-phil/causal-learn.git
numpy
scikit-learn
tqdm
graphviz
pydot
pygam
gcastle
sortedcontainers
dill
tensorflow
autograd

# optional for neural DAG methods:
torch
torchvision

# ===== R DEPENDENCIES (install manually in R) =====
# Required for iMIIC (R):
# install.packages("miic", dependencies = TRUE)
# install.packages("igraph", dependencies = TRUE)
# install.packages("devtools")
# library(devtools); install_github("MIICTeam/MIIC")

# ===== SYSTEM REQUIREMENTS =====
# R must be installed and available on PATH:
#   https://cran.r-project.org/
# Verify with:
#   Rscript --version
