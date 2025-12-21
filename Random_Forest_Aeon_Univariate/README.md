# README.md

This repository contains a set of Python modules and a main script that:

- load a univariate time-series dataset from aeon;
- optionally perform Bayesian hyperparameter optimisation;
- train a Random Forest classifier;
- convert the trained model into custom tree/forest structures;
- save all results into the `results/` directory.

### Using Python

To start the application follow these steps:

1. **Environment Setup**

	- **Using Conda**
	  ```bash
	  conda create --name datamining python=3.12
	  conda activate datamining
	  ```

	- **Using venv**
	  ```bash
	  python3.12 -m venv datamining_env
	  source datamining_env/bin/activate  # On macOS/Linux
	  datamining_env\Scripts\activate     # On Windows
	  ```

2. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Run**:
	
	python  `init_aeon_univariate.py <dataset_name>` 

	Example:
	```bash
	python init_aeon_univariate.py --list-datasets
	```
	```bash
	python init_aeon_univariate.py Coffee
	```
	```bash
	python init_aeon_univariate.py Coffee --optimize
	```
	
	Results are stored in: `results/`
