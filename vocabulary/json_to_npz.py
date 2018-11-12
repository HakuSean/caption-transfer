# Change the json of coco to npz for further usage
# Only works for json

import json
import argparse
import os
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('json_file', type=str)
args = parser.parse_args()

json_file = args.json_file
npz_file = os.path.splitext(os.path.basename(json_file))[0]

with open(json_file, 'r') as f:
    content = json.load(f)

img = content['images']
sent = content['annotations']

coco = {}

for i in img:
    coco[i['id']] = []

for s in sent:
    coco[s['image_id']].append(s['caption'])

# save 
np.savez(npz_file, image=coco.keys(), sent=coco.values())




