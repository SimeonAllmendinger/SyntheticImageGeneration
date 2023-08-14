<link href="./docs/style.css" rel="stylesheet"/>

# About the generation of synthetic laparascopic images using diffusion-based models

#### View research on [Github Page](https://simeonallmendinger.github.io/SyntheticImageGeneration//)

<div class="row"></div>

This repository is the code base used for our research. Please follow the guide:

## Clone Repository

```
git clone https://github.com/SimeonAllmendinger/SyntheticImageGeneration.git
cd SyntheticImageGeneration
```

## Virtual Environment
To set up a virtual environment, follow these steps:
1. Create a virtual environment with python version 3.9:

```
virtualenv venv -p $(which python3.9)
```

2. Activate the virtual environment:

```
source venv/bin/activate
```

3. Install the required packages:

```
pip install --no-cache-dir -r requirements.txt
```

## Download Model Weights
To test the generation of laparoscopic images with the Elucidated Imagen model, please do the following:
```
cd src/assets/
gdown --folder https://drive.google.com/drive/folders/1np4BON_jbQ1-15nVdgMCP1VKSbKS3h2M
gdown --folder https://drive.google.com/drive/folders/1BNdUmmqN18K4_lH0BMk0bwRkiy8Sv6D-
```

## Testing
To test the generation of laparoscopic images with the Elucidated Imagen model, please do the following:
```
python3 src/components/test.py --model=ElucidatedImagen --text='grasper grasp gallbladder in callot triangle dissection'
```

## git LFS

Install git LFS with homebrew: https://brew.sh/index_de
```
brew install git-lfs
git lfs install
git lfs track "*.pt"
git add .gitattributes
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
cd
cd SyntheticImageGeneration
mkdir data
cd data
```

2. Download the datasets in this directory after successful registration: 
-   Cholec80: https://docs.google.com/forms/d/1GwZFM3-GhEduBs1d5QzbfFksKmS1OqXZAz8keYi-wKI
-   CholecT45: https://forms.gle/jTdPJnZCmSe2Daw7A
-   CholecSeg8k: https://www.kaggle.com/datasets/newslab/cholecseg8k/download?datasetVersionNumber=11
(-   CholecT50: https://forms.gle/GbMj8TwNoNpMUJuv9)

### Preparation
To enable dashboards please copy your configs of neptune.ai and wandb.ai in the according .yaml file:

```
cd
cd SyntheticImageGeneration/configs/visualization/
touch config_neptune.yaml
touch config_wandb.yaml
```

1. Neptune.ai (https://neptune.ai):
Insert your acceess configs in the file config_neptune.yaml 
```
project: "your-project-name" 
api_token: "your-api-token"
```
2. Weights&Biases:
Insert your access configs in the file config_neptune.yaml 
```
project: "your-project-name" 
api_key: "your-api-key"
```

To prepare the data for the experiments, run the following script:
```
cd SyntheticImageGeneration
./scripts/run_data_preparation.sh
```

## SLURM
If you are using SLURM for job scheduling and the bwunicluster HPC server, create a temporary directory in the server terminal (valid for 60 days):

```
ws_allocate data-ssd 60
cd $(ws_find data-ssd)
mkdir SyntheticImageGeneration
```
