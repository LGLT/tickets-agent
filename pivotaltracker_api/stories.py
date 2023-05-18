from dotenv import load_dotenv
import requests
import json
import os

class StoryAPI:
    def __init__(self, base_url='https://www.pivotaltracker.com/services/v5/projects/1416286/'):
        load_dotenv()
        self.base_url = base_url
        self.headers = {'X-TrackerToken': os.environ['TRACKER_TOKEN']}

    def get_all_stories(self, state=''):
        try:
            response = requests.get('{}stories?{}'.format(self.base_url, state), headers=self.headers)
            response.raise_for_status() # raise exception if invalid response
            return response.json()
        except requests.exceptions.RequestException as e:
            print("Request error:{}".format(e))
            return None
