import json

with open("input_data/democrat_users.txt", "r") as f:
    for line in f:
        dem = line.replace(" ", "").split(',')
f.close()

with open("input_data/republican_users.txt", "r") as f:
    for line in f:
        rep = line.replace(" ", "").split(',')
f.close()

data=[]
i = 0
with open("input_data/political_tweets.ndjson") as f:
    for line in f:
        i = i + 1
        j_content = json.loads(line)
        user = (j_content['user']['screen_name'])
        if(user in dem):
            j_content['user']['party'] = 'democratic'
            data.append(line)
        elif(user in rep):
            j_content['user']['party'] = 'republican'
            data.append(line)

with open("parsed_parties.txt","w+") as f:
    for d in data:
        json.dump(json.JSONDecoder().decode(d), f)
        f.write('\n')
f.close()
