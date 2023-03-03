# Imports
import requests
import os
import zipfile

# Method for downloading datasets
def download_dataset(download_url: str,
                     local_path_to_zip_file: str,
                     directory_to_extract_to: str):

    # Create directory if not already done
    if not os.path.exists('SyntheticImageGeneration/data'):
        os.makedirs('SyntheticImageGeneration/data')

    # This is a 2-factor security boolean to avoid unnecessary CO2 footprint
    if not os.path.exists(local_path_to_zip_file):

        print('Start downloading dataset...')

        # Create response from url
        with requests.get(download_url, stream=True) as req:
            req.raise_for_status()

            # Initilize local file
            with open(local_path_to_zip_file, 'wb') as file:
                
                # Write content of response into local file
                for chunk in req.iter_content(chunk_size=8192):
                    file.write(chunk)

        print('...finished downloading dataset...')
    
    else:
        print('{} already exists!'.format(local_path_to_zip_file))

    
    print('...start to unzip dataset...')

    # if the directory to extract to already exists, do not unzip
    if not os.path.exists(directory_to_extract_to):
        print('...start to unzip dataset...')

        # Load zip and unzip to directory_to_extract_to directory
        with zipfile.ZipFile(local_path_to_zip_file, 'r') as zip_ref:
            zip_ref.extractall(directory_to_extract_to)
        
        print('...finished to unzip dataset!')
    
    else:
        print('{} already exists, so maybe it is already extracted!'.format(directory_to_extract_to))


if __name__ == '__main__':
    CHOLECT45_URL="http://lnkiy.in/cholect45dataset"
    CHOLECSEG8K_URL='https://www.kaggle.com/datasets/newslab/cholecseg8k/download?datasetVersionNumber=11'
    CHOLEC80 = 'https://drive.google.com/drive/folders/1EzG06wlp-HG9fW6keg3nFFd9q7DNGY0g?usp=share_link'
    url_list = [CHOLEC80]
    name_list = ['Cholec80']
    
    for i, url in enumerate(url_list):
        download_dataset(download_url=url,
                        local_path_to_zip_file=f'data/{name_list[i]}', 
                        directory_to_extract_to = f'data/{name_list[i]}'.strip('.zip'))