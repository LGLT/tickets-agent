import requests

url = "https://www.pivotaltracker.com/services/v5/projects/1416286/stories?with_state=unstarted"

payload = {}
headers = {
  'X-TrackerToken': ''
}

response = requests.request("GET", url, headers=headers, data=payload)

print(response.text)
