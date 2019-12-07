import json
import re

with open("input_data/keywords.txt", "r") as f:
    for line in f:
        hashtags = list(line.replace(" ", "").replace("\n","").split(','))
        words = list(line.replace("\n","").split(', '))
f.close()

singular_len = len(hashtags)
for i in range(singular_len):
    hashtags.append(hashtags[i]+"s")
    words.append(words[i]+"s")

print("hashtags: ",hashtags, len(hashtags))
print("\n")
print("words: ",words, len(words))

data=[]
correct_labels=[]
counter = 0
# shorter_set =0 
with open('parsed_parties.txt') as f:
    for line in f:
        # if shorter_set > 2000:
        #     break
        # shorter_set += 1
        appended = False
        j_content = json.loads(line)
        for val in j_content['entities']['hashtags']:
            if(val['text'].casefold() in hashtags):
                counter+=1
                data.append(j_content['full_text'])
                correct_labels.append(j_content['user']['party'])
                appended = True
                print("hashtag")
                print(val['text'].casefold())
                print('\n')
                break
        if not appended:
            text = j_content['full_text'].casefold()
            for word in words:
                if(word in text):
                    data.append(j_content['full_text'])
                    correct_labels.append(j_content['user']['party'])
                    print("text")
                    print(word)
                    print('\n')
                    break

with open("parsed_climate_inputs.txt","w+") as file_inputs:
    for d in data:
        text = str(json.dumps(d))
        if text[1:3] == "RT":
            continue
        text = re.sub(r'http\S+', '', text)
        text = re.sub(r'\\\S+', '', text)
        text = re.sub(r'[^\w\s\"]','',text)
        file_inputs.write(text)
        file_inputs.write('\n')
file_inputs.close()
with open("parsed_climate_labels.txt","w+") as file_labels:
    for l in correct_labels:
        json.dump(l, file_labels)
        file_labels.write('\n')
file_labels.close()
