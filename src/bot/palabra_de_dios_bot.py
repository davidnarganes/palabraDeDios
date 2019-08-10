import tweepy
import json
import datetime
import random

def initialize(consumer_key="",
               consumer_secret="",
               access_token="",
               access_secret=""):

    # Authentication
    auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
    auth.set_access_token(access_token, access_secret)
    
    api = tweepy.API(auth)
    
    print "Twitter api initialized"
    
    profile_text = "Falsos versículos de la Biblia \nGuillermo Serrano Nájera \n"
    
    api.update_profile("PalabraDeDios",
                       "https://gserranonajera.wordpress.com",
                       "Cartago",
                       profile_text)

    return(api, auth)

class bot():
    
    def __init__(self):
        api, auth = initialize()
        self.delayInMinutes=30
        self.prevPublicationTime=api.user_timeline(id = api.me().id, count = 1)[0].created_at
        self.detectionTime=datetime.datetime.now()
        self.day = datetime.datetime.now().day
        print("Bot initialized")

    def write_verse():
        pass
    
    def publish_verse():
        pass


def main():
    api, auth = initialize()

