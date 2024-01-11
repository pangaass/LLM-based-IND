import json
path = '/workspace/pangyunhe/dataset/whoiswho/train_author.json'

with open(path, 'r', encoding='utf-8') as f:
    dataset = json.load(f)
data = []
for author in dataset.keys():
    data.extend(dataset[author]['normal_data'])
    data.extend(dataset[author]['outliers'])
data1 = {}
for i in data:
    if i in data1.keys():
        data1[i] += 1
    else:
        data1[i] = 1

repeat = []
max1 = 1
for i in data1.keys():
    if data1[i] >1:
        max1 = max(data1[i],max1)
        repeat.append(i)
print(repeat)