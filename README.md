# Navigating the Synthetic Realm: Harnessing Diffusion-based Models for Laparoscopic Image Generation
Recent advancements in computer-generated synthetic images open up opportunities for obtaining additional data in the field of surgical imaging. This data can provide reliable supplements for surgical videos supporting training and simulation purposes. Our study presents a method for generating synthetic laparoscopic images from short text prompts using diffusion-based generative models. We demonstrate the usage of state-of-the-art text-to-image architectures in the context of laparoscopic imaging with regard to cholecystectomy surgery as an example. First, the three models Dalle2, Imagen, and Elucidated Imagen were trained to acquire knowledge about the style and semantics of laparoscopic images and assessed with FID, FCD, and KID scores among others to evaluate their potential. Our results show that the Imagen and Eluci- dated Imagen models outperform the Dall-e2 model in terms of fidelity and diversity of the produced images. Second, the performance of a state-of-the-art machine learn- ing task in medical imaging is evaluated, which exhibits promising indications when trained with our additional synthetic images. Moreover, this work also introduces a questionnaire to measure the persuasive power of the synthetic images on partici- pants with medical and non-medical backgrounds and discusses current limitations of computer-generated laparoscopic images. Overall, the findings contribute to the development of computer-generated images that can certainly become an essential data tool in surgical research and practice.

This repository is the code base used for our research. Please follow the guide:

## Clone Repository

```
git clone https://git.scc.kit.edu/xq5986/master-thesis-simeon-allmendinger.git
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
## Testing
To test the generation of laparoscopic images by the Elucidated Imagen model, please do the following:
```
./test.sh
```

## Submodules
This project utilizes submodules. To initialize and update the submodules, run the following commands:

```
git submodule init
git submodule update --remote
git submodule add https://github.com/CAMMA-public/rendezvous.git
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
-   CholecT50: https://forms.gle/GbMj8TwNoNpMUJuv9
-   CholecSeg8k: https://www.kaggle.com/datasets/newslab/cholecseg8k/download?datasetVersionNumber=11

### Preparation
To enable proper visualization please copy your configs in the according .yaml file:

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
