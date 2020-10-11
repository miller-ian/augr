# -*- coding: utf-8 -*-
"""
Created on Sun Oct 11 10:32:56 2020

@author: liamc
"""

# Download the helper library from https://www.twilio.com/docs/python/install
from twilio.rest import Client


# Your Account Sid and Auth Token from twilio.com/console
# DANGER! This is insecure. See http://twil.io/secure


def send_message(url):
    account_sid = 'AC3ac159b57ca5d5391ddfcb6fbf324ab4'
    auth_token = 'fb9575b1e1feba59ca54c7112f5695fd'
    client = Client(account_sid, auth_token)
    
    message = client.messages \
                    .create(
                         body=url,
                         from_='+12017201190',to='+16128048403')