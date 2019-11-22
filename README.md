# rolling-deep
Deep Learning final project

1. If needed, download political_tweets.njson and unizip the file inside the 'input_data folder', download can be found here: https://t.co/ZZbyFb8BfZ

- parties.py: Gets initial file with every single tweet ('political_tweets.njson') and filters only tweets by republicans and democrats, writes to 'parsed_parties.txt'. Computer might run out of memory, should be run on a Virtual Environment.

- climate.py: Uses 'keywords.txt' to create 2 lists: hashtags and key terms. Get as input a file with only tweets by republicans and democrats ('parsed_parties.txt') and writes these tweets to a new file ('parsed_climate.txt')
