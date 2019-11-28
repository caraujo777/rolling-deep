import json

# for now, the hashtags and words lists are the same, one with no spaces and one with spaces
# in the future, we could have 2 separate txt files maybe
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
with open('parsed_parties.txt') as f:
    for line in f:
        appended = False
        j_content = json.loads(line)
        for val in j_content['entities']['hashtags']:
            if(val['text'].casefold() in hashtags):
                data.append(j_content)
                appended = True
                print("hashtag")
                print(val['text'].casefold())
                print('\n')
                break
        if not appended:
            text = j_content['full_text'].casefold()
            for word in words:
                if(word in text):
                    data.append(j_content)
                    print("text")
                    print(word)
                    print('\n')
                    break

with open("parsed_climate.txt","w+") as f:
    for d in data:
        json.dump(d, f)
        f.write('\n')
f.close()
