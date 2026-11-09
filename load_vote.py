import json
vote_json = '/home1/users/tangyuan01/YOLOC/runs/detect/train/vote.json'
with open(vote_json, 'r') as f:
    vote_data = json.load(f)
print(vote_data)