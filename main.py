#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Look up nltk library

import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer as SIA


import praw
import matplotlib.pyplot as plt
import math
import datetime as dt
import pandas as pd
import numpy as np


# In[2]:

# download external libraries?

nltk.download('vader_lexicon')
nltk.download('stopwords')


# In[3]:

# login to praw

reddit = praw.Reddit(client_id='*********',
                    client_secret='******************',
                    user_agent='*********') ## to use this, make a Reddit app. Client ID is in top left corner, client secret is given, and user agent is the username that the app is under


# In[4]:


# subreddit and stocks

sub_reddits = reddit.subreddit('wallstreetbets')
stocks = ["GME", "AMC"] 
# For example purposes. To use this as a live trading tool, you'd want to populate this with tickers that have been mentioned on the pertinent community (WSB in our case) in a specified period.


# In[5]:


# comment sentiment function

def commentSentiment(ticker, urlT):
  
  '''
  commentSentiment function taking in ticker and urlT variable
  '''
    subComments = [] # sub comments
    bodyComment = [] # body comments
    try:
        check = reddit.submission(url=urlT) # check if urlT 
        subComments = check.comments # Subcomments is list of comments
    except:
        return 0 # return 0 comments if an error
    
    for comment in subComments: # for comments in subcommnets
        try: 
            bodyComment.append(comment.body) # append body to comment 
        except:
            return 0
          
    
    
    sia = SIA() # look up SIA
    results = [] # results
    for line in bodyComment: # analyze line in each body comment
        scores = sia.polarity_scores(line) # calculate polarity scores? And pass in each line of text
        scores['headline'] = line # scores under headline index is that line 

        results.append(scores) # append scores to that result 
    
    df =pd.DataFrame.from_records(results) # create dataframe from result
    df.head() # generate head
    df['label'] = 0 # label = 0 
    
    try:
        df.loc[df['compound'] > 0.1, 'label'] = 1
        df.loc[df['compound'] < -0.1, 'label'] = -1
    except:
        return 0
    
    averageScore = 0
    position = 0
    while position < len(df.label)-1:
        averageScore = averageScore + df.label[position]
        position += 1
    averageScore = averageScore/len(df.label) 
    
    return(averageScore)


# In[6]:


def latestComment(ticker, urlT):
    subComments = []
    updateDates = []
    try:
        check = reddit.submission(url=urlT)
        subComments = check.comments
    except:
        return 0
    
    for comment in subComments:
        try: 
            updateDates.append(comment.created_utc)
        except:
            return 0
    
    updateDates.sort()
    return(updateDates[-1])


# In[7]:


def get_date(date):
    return dt.datetime.fromtimestamp(date)


# In[8]:


submission_statistics = []
d = {}
for ticker in stocks:
    for submission in reddit.subreddit('wallstreetbets').search(ticker, limit=130):
        if submission.domain != "self.wallstreetbets":
            continue
        d = {}
        d['ticker'] = ticker
        d['num_comments'] = submission.num_comments
        d['comment_sentiment_average'] = commentSentiment(ticker, submission.url)
        if d['comment_sentiment_average'] == 0.000000:
            continue
        d['latest_comment_date'] = latestComment(ticker, submission.url)
        d['score'] = submission.score
        d['upvote_ratio'] = submission.upvote_ratio
        d['date'] = submission.created_utc
        d['domain'] = submission.domain
        d['num_crossposts'] = submission.num_crossposts
        d['author'] = submission.author
        submission_statistics.append(d)
    
dfSentimentStocks = pd.DataFrame(submission_statistics)

_timestampcreated = dfSentimentStocks["date"].apply(get_date)
dfSentimentStocks = dfSentimentStocks.assign(timestamp = _timestampcreated)

_timestampcomment = dfSentimentStocks["latest_comment_date"].apply(get_date)
dfSentimentStocks = dfSentimentStocks.assign(commentdate = _timestampcomment)

dfSentimentStocks.sort_values("latest_comment_date", axis = 0, ascending = True,inplace = True, na_position ='last') 

dfSentimentStocks


# In[9]:


dfSentimentStocks.author.value_counts()


# In[10]:


dfSentimentStocks.to_csv('Reddit_Sentiment_Equity.csv', index=False) 


# In[ ]:
