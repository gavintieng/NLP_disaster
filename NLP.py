import numpy as np
import pandas as pd
import nltk as nltk
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

warnings.filterwarnings("ignore")

from collections import defaultdict
from collections import Counter
from nltk.corpus import stopwords

# Import Data

test = pd.read_csv("/Users/Gavin/Desktop/NLP_proj/NLP_disaster/test.csv")
train = pd.read_csv("/Users/Gavin/Desktop/NLP_proj/NLP_disaster/train.csv") #tweet

# 1) Introductory Data Exploration

#print(test.head())
print("Test Dataset has {} rows and {} columns".format(test.shape[0],test.shape[1]))

#print(train.head())
print("Train Dataset has {} rows and {} columns".format(train.shape[0],train.shape[1]))

# Basic visualizations of data

# Barplot to show number of disaster tweets vs. number of non-disaster tweets
train_zeros = train[train.target==0] #non-disaster tweets
train_ones = train[train.target==1] #disaster tweets
names = ['Non-Disaster Related (0)','Disaster Related (1)']
values = [len(train_zeros),len(train_ones)]
barlist=plt.bar(names,values)
plt.title('Number of Non-Disaster vs. Disaster Related Tweets')
barlist[0].set_color('r')
barlist[1].set_color('g')
plt.show()

# Histogram comparing the number of characters in disaster vs. non-disaster tweets

disaster_length = train[train.target==1]['text'].str.len()
nondisaster_length = train[train.target==0]['text'].str.len()

fig,(ax1,ax2)=plt.subplots(1,2,figsize=(10,5))
fig.suptitle('Length of Non-Diaster Tweets vs. Length of Disaster Tweets')
ax1.hist(nondisaster_length,color='red')
ax1.set_title('Non-Disaster Tweets')
plt.xlabel('Number of Tweets')
plt.ylabel('Length (characters)')
ax2.hist(disaster_length,color='green')
ax2.set_title('Disaster Tweets')
plt.show()

print(train.head())
# Average word length in disaster vs. non-disaster tweets

avg_disaster_length = 0
for words in disaster_length:
    avg_disaster_length += words

simplified_avg_disaster = int(avg_disaster_length/len(disaster_length))
print("Average Word Length of Disaster Tweets:",simplified_avg_disaster)

avg_nondisaster_length = 0
for words in nondisaster_length:
    avg_nondisaster_length += words

simplified_avg_nondisaster = int(avg_nondisaster_length/len(nondisaster_length))
print("Average Word Length of Non-Disaster Tweets:",simplified_avg_nondisaster)

#it appears that disaster tweets are a little bit longer on average, but not by a lot

#analyzing common stopwords and punctuation (in preparation for data cleaning)

stopw = set(stopwords.words('english'))

def create_corpus(target):

    corpus = []
    # iterate through tweets with the matching target and split their text
    for x in train[train['target']==target]['text'].str.split():
        # iterate through words in each tweet and append all to a list (corpus)
        for i in x:
            corpus.append(i)
    return corpus

corpus0 = create_corpus(0)
dict0 = defaultdict(int)
for word in corpus0:
    if word in stopw:
        dict0[word]+=1
top0 = sorted(dict0.items(),key=lambda x:x[1],reverse=True)[:10]

corpus1 = create_corpus(1)
dict1 = defaultdict(int)
for word in corpus1:
    if word in stopw:
        dict1[word]+=1
top1 = sorted(dict1.items(),key=lambda x:x[1],reverse=True)[:10]

a,b=zip(*top0)
a1,b1=zip(*top1)
colors=['r','c','c','c','c','c','c','c','c','c']

fig,(plot1,plot2)=plt.subplots(1,2,figsize=(10,5))
fig.suptitle('Most Common Stopwords')
plot1.bar(a,b,color=colors)
plot1.set_title('Non Disaster Tweets')
#plt.ylabel('Number of Occurences')
plot2.bar(a1,b1,color=colors)
plot2.set_title('Disaster Tweets')
plt.show()


#analyzing common words (not stopwords)

# container that stores elements as keys with their count being the value
# starting with non-disaster related tweets
count0 = Counter(corpus0)
most0 = count0.most_common()
x0=[]
y0=[]
for word, count in most0[:40]:
    if (word not in stopw):
        x0.append(word)
        y0.append(count)

#print(x0)
#print(y0)

# most common words for disaster-related tweets (1)
count1 = Counter(corpus1)
most1 = count1.most_common()
x1=[]
y1=[]
for word,count in most1[:40]:
    if(word not in stopw):
        x1.append(word)
        y1.append(count)

#print(x1)
#print(y1)
colors1 = ['r','c','c','c','c','c','c','c','c','c','c',
           'c']
fig,(bar1,bar2) = plt.subplots(1,2,figsize=(15,10))
fig.suptitle('Most Common Words (not stopwords)')
bar1.bar(x0,y0,color=colors)
bar1.set_title('Non-Disaster Related Tweets')
bar2.bar(x1,y1,color=colors1)
bar2.set_title('Disaster Related Tweets')
plt.show()


#2) data preprocessing part 1: removing common stopwords and creating a corpus



