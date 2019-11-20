import json

file_path = "political_tweets.ndjson"

with open("democrat_users.txt", "r") as f:
    for line in f:
        dem = line.replace(" ", "").split(',')
f.close()

with open("republican_users.txt", "r") as f:
    for line in f:
        rep = line.replace(" ", "").split(',')
f.close()

with open("parsed_parties.txt", "w+") as f:
    json.dump("hallo", f)

data=[]
with open(file_path) as f:
    for line in f:
        j_content = json.loads(line)
        user = (j_content['user']['screen_name'])
        if(user in dem):
            j_content['user']['party'] = 'democratic'
            data.append(j_content)
        elif(user in rep):
            j_content['user']['party'] = 'republican'
            data.append(j_content)

with open("parsed_parties.txt", "w+") as f:
    for d in data:
        json.dump(d, f)
f.close()
