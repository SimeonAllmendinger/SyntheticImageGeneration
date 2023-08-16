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
gdown --folder https://drive.google.com/drive/folders/1Y0yQmP3THRzP8UFlAyMFHYUymTUu7ZUu
cd ../../
```

## Testing
To test the generation of laparoscopic images with the pre-trained Elucidated Imagen model, please do the following:
```
python3 src/components/test.py --model=ElucidatedImagen --text='grasper grasp gallbladder in callot triangle dissection' --cond_scale=3
```

You can apply the Imagen and Elucidated Imagen model, various conditiong scales and a suitable text prompt according to your desire! Feel free to try everything out. (The sampling of the Elucidated Imagen model also works well on a machine without GPU).

The hyperparameter configurations of the diffusion-based models are contained in the config file respectively ([Model Config Folder](https://github.com/SimeonAllmendinger/SyntheticImageGeneration/tree/main/configs/models)). Their weights can be found in the table:

| Model             | Training Dataset          |    Link                           |
| ---               | ---                       | ---                               |
| Dall-e2 Prior     | CholecT45                 |  [Dalle2_Prior_T45](https://drive.google.com/file/d/17hUYgOPMuIA7twX7IcWAjlMAdwkTPPVo/view?usp=share_link)  |
| Dall-e2 Decoder   | CholecT45                 |  [Dalle2_Decoder_T45](https://drive.google.com/file/d/1zy2oiSGlXTxPtjIbM1_QV1_Qi8f3QgZK/view?usp=share_link)  |
| Imagen            | CholecT45                 |  [Imagen_T45](https://drive.google.com/file/d/1Nk_Pskv5lphDzERDPyaafyl4Hf_597S0/view?usp=share_link)  |
| Imagen            | CholecT45 + CholecSeg8k   |  [Imagen_T45_Seg8k](https://drive.google.com/file/d/1myQYlwYWlmnxIvJHkI_tSAQ2yQuXIk7j/view?usp=share_link)  |
| Elucidated Imagen | CholecT45                 |  [ElucidatedImagen_T45](https://drive.google.com/file/d/1RVHM3jzsMtqRNwuyU2Wi9RExtIYlwDVp/view?usp=share_link)  |
| Elucidated Imagen | CholecT45 + CholecSeg8k   |  [ElucidatedImagen_T45_Seg8k](https://drive.google.com/file/d/1EdFsQB0RYvVUvonan16RKgIzDoSr3NiK/view?usp=share_link)  |


## Results
Before running the code for training, tuning and extensive testing purposes, please create a directory to store the results:

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
### git LFS

Install git LFS with homebrew: https://brew.sh/index_de
```
brew install git-lfs
git lfs install
git lfs track "*.pt"
git add .gitattributes
```

### Download
To download the required datasets (CholecT45, CholecSeg8k, CholecT50, Cholec80), follow these steps:
1. Create a directory to store the data:

```
cd
cd SyntheticImageGeneration
mkdir data
cd data
```

2. Download the datasets into this directory after successful registration: 
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

Now, you are prepared to explore the code base in full extense!


## Rendezvouz ([GitHub](https://github.com/CAMMA-public/rendezvous))

In the following, we provide trained rendezvous model weights from the 3-fold cross-validation for various proportions of generated samples:

| Model   | %2 samples | %5 samples  | %10 samples | %20 samples | %25 samples |
| --- | --- | --- | --- | --- | --- |
| I5-RDV  | [Weights](https://drive.google.com/file/d/1LC7OTJMVn_lPy1pucmHMnkRCz1BNnTBI/view?usp=share_link) | [Weights](https://drive.google.com/file/d/1rDyXPNh8vgj4yJwXotEGMNCYfOwVV9z6/view?usp=share_link) | [Weights](https://drive.google.com/file/d/1yqxXXb4-OfWJ4QniyZSJJgG_nyqMrsfy/view?usp=share_link) | [Weights](https://drive.google.com/file/d/1Cas_dBBBCqYOmzJacH_toFQqhSu_Naum/view?usp=share_link) | [Weights](https://drive.google.com/file/d/1a6STIJThlRYqIAgHRnbZACiceVrBMlPS/view?usp=share_link) |
| EI5-RDV | [Weights](https://drive.google.com/file/d/1LC7OTJMVn_lPy1pucmHMnkRCz1BNnTBI/view?usp=share_link) | [Weights](https://drive.google.com/file/d/1Miqj6ZPjqCLvuPjXEe9xgUuB3bQnLMC6/view?usp=share_link) | [Weights](https://drive.google.com/file/d/1bNSahBYI1AdATPypjQ3_Xht03pawGBRR/view?usp=share_link) | [Weights](https://drive.google.com/file/d/1bNSahBYI1AdATPypjQ3_Xht03pawGBRR/view?usp=share_link) | [Weights](https://drive.google.com/file/d/1Cas_dBBBCqYOmzJacH_toFQqhSu_Naum/view?usp=share_link) |


# Acknowledgements

We acknowledge support by the state of Baden-Württemberg through bwHPC.