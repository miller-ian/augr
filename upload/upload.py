# -*- coding: utf-8 -*-
"""
Created on Sun Oct 11 11:19:33 2020

@author: liamc
"""
import dropbox
import random


access_token = '8iy7bR50q-EAAAAAAAAAAWemuNEdn7RGALEcCH6Qb1H3y4kVobkzhOGR99uTl3Cb'

def upload_file(file_from, file_to):
    """upload a file to Dropbox using API v2
    """
    dbx = dropbox.Dropbox(access_token)

    with open(file_from, 'rb') as f:
        dbx.files_upload(f.read(), file_to)
        
    return dbx.sharing_create_shared_link_with_settings(file_to, settings=None).url
    
    