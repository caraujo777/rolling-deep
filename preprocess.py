import json

file_path = "political_tweets.ndjson"

# load from file-like objects
data =[]
with open(file_path) as f:
    for line in f:
        j_content = json.loads(line)
        data.append(j_content)
print(len(data))
print(data[0])
