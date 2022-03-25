# SPSoil - Regression
Repository for our spectral based estimation of soil properties.
Knowing chemical soil properties might be determinant in crop management and total yield production. Traditional approaches for property estimation are time-consuming and require complex lab setups, refraining farmers from taking steps towards optimal practices in their crops promptly. Property estimation from spectral signals emerged as a low-cost, non-invasive, and non-destructive alternative. Current approaches use regression, excluding tools from the machine learning framework. Here we test both regression and classification machine techniques and assess performance on common prediction metrics. We achieved similar performance on similar setups reported in literature.

We implement some pre-processiong and regression scripts running over python, with a simple CSV formatted data on input.

##Install requirements

1. Clone repository:

```
git clone https://github.com/cavargar/SPSoil
```

2. Install [Install](https://docs.conda.io/projects/conda/en/latest/user-guide/install/linux.html) Conda or Miniconda, and activate it.

3. Change working directory to `SPSoil`

```
cd SPSoil
```

4. Create a conda environment from spsoil.yaml file:
```
conda env create -f spsoil.yaml
```
##Running SPSoil Regression scripts

1. Edit `config.yaml` file according with your requirements:

```
#SPSoil Regreession config.yaml file

data:
  #NIRS spectrum and properties data path for output and input data
  # input dataset must be CSV formatted with ';' column separator
  dataPath:
    - data/
  dataFile:
    - soil_cane_vis-NIR.csv
  #Thresholds table data set
  # must be CSV formatted with ';' column separator  
  thresholdsPath:
    - data/properties_thresholds.csv
properties:
    # Space separated properties, must be contained in properties data set header
  - pH OM
  #- pH OM Ca Mg K Na
NIRSInfo:
  steps:
    # Space separated steps in NIRS spectrum (float or INT)
    # begin step end
    - 400 8.5 2492
Regression:
  models:
	# Available LR, SVR, LASSO
    - LR SVR

```

2. Activate conda environment:

```
conda activate spsoil
```

3. Run `main.py` script:

```
python main.py
```


