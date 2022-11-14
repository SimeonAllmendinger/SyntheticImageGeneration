import requests

def get_download_from_url(download_url):
    response = requests.get(download_url)
    return response

def get_content_from_download(response):
    return response.content

if __name__ == '__main__':
    #CholeecT45: http://lnkiy.in/cholect45dataset
    download_url="http://lnkiy.in/cholect45dataset"
    response=get_download_from_url(download_url=download_url)
    get_content_from_download(response=response)