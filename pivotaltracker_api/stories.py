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
            params = { 'with_state': state }
            url = '{}stories?'.format(self.base_url)
            response = requests.get(url, params=params, headers=self.headers)
            response.raise_for_status() # raise exception if invalid response
            parsed_response = self.parse_response(response.json())
            return parsed_response
        except requests.exceptions.RequestException as e:
            print("Request error:{}".format(e))
            return None
    
    def update_story(self, story_id, story_state):
        try:
            params = { 'current_state': story_state }
            url = f'{self.base_url}stories/{story_id}'
            requests.put(url, params=params, headers=self.headers)
            return 
        except requests.exceptions.RequestException as e:
            print("Request error:{}".format(e))
            return e
    
    def parse_response(self, tickets):
        return [{'id':ticket['id'], 'name': ticket['name'], 'url': ticket['url'], 'labels': {label['name'] for label in ticket['labels']}} for ticket in tickets]
