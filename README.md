# SyntheticImageGeneration
Master Thesis

## Clone Repository

```
git clone 
cd SyntheticImageGeneration
```

## Virtual Environment
To set up a virtual environment, follow these steps:
1. Create a virtual environment:

```
python3 -m venv venv
```

2. Activate the virtual environment:

```
source venv/bin/activate
```

3. Install the required packages:

```
pip install -r requirements.txt
```

## Submodules
This project utilizes submodules. To initialize and update the submodules, run the following commands:

```
git submodule init
git submodule update --remote
git submodule add https://github.com/CAMMA-public/rendezvous.git
```
## Results
Before running the code, create a directory to store the results:

```
mkdir results
cd results
mkdir rendevouz
mkdir testing
mkdir training
mkdir TSNE
mkdir tuning
```

## Data
### Download
To download the required datasets (CholecT45, CholecSeg8k, CholecT50, Cholec80), follow these steps:
1. Create a directory to store the data:

```
cd SyntheticImageGeneration
mkdir data
cd data
```

2. Download the datasets in this directory after successful registration: 
-   Cholec80: https://docs.google.com/forms/d/1GwZFM3-GhEduBs1d5QzbfFksKmS1OqXZAz8keYi-wKI
-   CholecT45: https://forms.gle/jTdPJnZCmSe2Daw7A
-   CholecT50: https://forms.gle/GbMj8TwNoNpMUJuv9
-   CholecSeg8k: https://www.kaggle.com/datasets/newslab/cholecseg8k/download?datasetVersionNumber=11

### Preparation
To prepare the data for the experiments, run the following script:

```
cd SyntheticImageGeneration
./scripts/run_test_rendevouz.sh
```

## SLURM
If you are using SLURM for job scheduling, create a temporary directory for SLURM:

mkdir TMP
