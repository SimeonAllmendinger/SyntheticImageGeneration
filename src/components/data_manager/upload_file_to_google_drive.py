from download_from_url import get_content_from_download, get_download_from_url
from pydrive.auth import GoogleAuth
from pydrive.drive import GoogleDrive

def authenticate_google_access():
    gauth = GoogleAuth()
    gauth.LocalWebserverAuth()        
    drive = GoogleDrive(gauth)
    return drive

def upload_file_to_google_drive(drive, file, filename :str, drive_id :str):
    gfile = drive.CreateFile({'parents': [{'id': drive_id}],'title': filename})
    # Read file and set it as the content of this instance.
    gfile.SetContentFile(file)
    # Upload the file.
    gfile.Upload() 
    
if __name__ == '__main__':
    # MasterThesis: 1xrkZcPELSlJdYpJ4f7xt6F06WjP-pqGZ
    # datatsets: 1KFejHZ2oCrmo_DdlqoflnMZwTz9OPlc2
    drive_id='1KFejHZ2oCrmo_DdlqoflnMZwTz9OPlc2'
    download_url="http://lnkiy.in/cholect45dataset"
    
    file=get_content_from_download(get_download_from_url(download_url=download_url))
    drive=authenticate_google_access()
    upload_file_to_google_drive(drive=drive, file=file, filename='test.pdf', drive_id=drive_id)