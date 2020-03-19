import json

files = []
with open('data/xls/filtered.ndjson') as f:
    for line in f:
        files.append(json.loads(line)['filename'])

files1 = []
with open('data/xls/filtered_inspect.ndjson') as f:
    for line in f:
        files1.append(json.loads(line)['filename'])

print(len(files))
print(len(files1))
print(len(list(set(files) - set(files1))))