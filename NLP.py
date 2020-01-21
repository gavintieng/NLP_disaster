import numpy as np
import pandas as pd
import nltk as nltk
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

warnings.filterwarnings("ignore")

# Import Data

test = pd.read_csv("/Users/Gavin/Desktop/NLP_proj/NLP_disaster/test.csv")
train = pd.read_csv("/Users/Gavin/Desktop/NLP_proj/NLP_disaster/train.csv") #tweet

# Data Exploration

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

# Average word length in disaster vs. non-disaster tweets

