# Change the json of coco to npz for further usage
# Only works for json
# the input is dataset_coco.json

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
coco = {}

for i in img:
    idx = str(i['cocoid']).encode("utf-8").decode("utf-8")
    coco[idx] = []
    for s in i['sentences']:
        coco[idx].append(s['raw'])

# save 
np.savez(npz_file, image=coco.keys(), sent=coco.values())




