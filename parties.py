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
with open("input_data/political_tweets.ndjson") as f:
    for line in f:
        j_content = json.loads(line)
        user = (j_content['user']['screen_name'])
        if(user in dem):
            j_content['user']['party'] = 'democratic'
            data.append(j_content)
        elif(user in rep):
            j_content['user']['party'] = 'republican'
            data.append(j_content)
        print(len(data))

with open("parsed_parties.txt","w+") as f:
    for d in data:
        json.dump(d, f)
f.close()
