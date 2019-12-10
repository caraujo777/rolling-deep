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

data=[]
correct_labels=[]
counter = 0
shorten = 0
dem = 0
rep = 0
with open('parsed_parties.txt') as f:
    for line in f:
        appended = False
        j_content = json.loads(line)
        for val in j_content['entities']['hashtags']:
            if(val['text'].casefold() in hashtags):
                if j_content['user']['party'] == 0:
                    rep += 1
                if j_content['user']['party'] == 1:
                    if dem >= 7719:
                        continue
                    dem += 1

                counter+=1
                data.append(j_content['full_text'])
                correct_labels.append(j_content['user']['party'])
                appended = True
                break
        if not appended:
            text = j_content['full_text'].casefold()
            text = re.sub(r'http\S+', '', text)
            text = re.sub(r'www\S+', '', text)
            text = re.sub(r'\\\S+', '', text)
            text = re.sub(r'[^\w\s]',' ',text)
            text = re.sub(r'\s+', ' ', text)
            for word in words:
                cont = True
                text_words = text.split()
                for single_word in word.split():
                    if single_word not in text_words:
                        cont = False 
                if cont:
                    if j_content['user']['party'] == 0:
                        rep += 1
                    if j_content['user']['party'] == 1:
                        if dem >= 7719:
                            continue
                        dem += 1                    
                    data.append(j_content['full_text'])
                    correct_labels.append(j_content['user']['party'])
                    break

print("rep, dem",rep, dem)

with open("parsed_climate_inputs.txt","w+") as file_inputs:
    for d in data:
        text = str(json.dumps(d))
        if text[1:3] == "RT":
            continue
        text = re.sub(r'http\S+', '', text)
        text = re.sub(r'www\S+', '', text)
        text = re.sub(r'\\\S+', '', text)
        text = re.sub(r'[^\w\s]',' ',text)
        text = re.sub(r'\s+', ' ', text)
        text = text.casefold()
        file_inputs.write(text)
        file_inputs.write('\n')
file_inputs.close()
with open("parsed_climate_labels.txt","w+") as file_labels:
    for l in correct_labels:
        json.dump(l, file_labels)
        file_labels.write('\n')
file_labels.close()
