from pydrive.auth import GoogleAuth
from pydrive.drive import GoogleDrive

def authenticate_google_access():
    gauth = GoogleAuth()
    gauth.LocalWebserverAuth()        
    drive = GoogleDrive(gauth)
    return drive