import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
import nltk
from nltk.corpus import stopwords
from collections import Counter
import string
# NLTK stopwords 
nltk.download('stopwords')
nltk.download('vader_lexicon')

# JSON files to pandas
def load_data():
    data = []
    for season in range(1, 8):
        with open(f'data/season{season}.json', 'r', encoding='utf-8') as f:
            season_data = json.load(f)
            for episode, subtitles in season_data.items():
                for key, line in subtitles.items():
                    data.append({'season': season, 'episode': episode, 'line_number': key, 'text': line})
    return pd.DataFrame(data)

df = load_data()
#print(df)
# text clean
def clean_text(text):
    text = text.lower()
    text = ''.join([char for char in text if char not in string.punctuation])
    text = ' '.join([word for word in text.split() if word not in stopwords.words('english')])
    return text

# Sentiment Func
def categorize_sentiment(score):
    if score >= 0.05:
        return 'Positive'
    elif score <= -0.05:
        return 'Negative'
    else:
        return 'Neutral'

df['cleaned_text'] = df['text'].apply(lambda x: clean_text(str(x)))

# text legth
df['text_length'] = df['cleaned_text'].apply(lambda x: len(x.split()))


# episode and season
season_episode_lengths = df.groupby(['season', 'episode'])['text_length'].sum().reset_index()

# freq words
all_words = ' '.join([text for text in df['cleaned_text']])
word_freq = Counter(all_words.split())
common_words = word_freq.most_common(20)

# generade wordcloud
wordcloud = WordCloud(width=800, height=400, max_words=100).generate(all_words)
plt.figure(figsize=(10, 8))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.show()

#most visited places (frequency in conversations)
locations = ['winterfell', 'kings landing', 'wall', 'essos', 'braavos', 'meereen', 'dorne','lys','highgarden','dragonstone']
location_counts = {location: all_words.count(location) for location in locations}

plt.figure(figsize=(10, 6))
plt.bar(location_counts.keys(), location_counts.values())
plt.title('En Çok İsmi Geçen Mekanlar')
plt.xlabel('Mekanlar')
plt.ylabel('Frekans')
plt.show()

#most used name
names = ['jon', 'daenerys', 'tyrion', 'cersei', 'arya', 'sansa', 'bran', 'jaime']
name_counts = {name: all_words.count(name) for name in names}

plt.figure(figsize=(10, 6))
plt.bar(name_counts.keys(), name_counts.values())
plt.title('En Çok Hitap Edilen İsimler')
plt.xlabel('İsimler')
plt.ylabel('Frekans')
plt.show()


from nltk.sentiment.vader import SentimentIntensityAnalyzer

sid = SentimentIntensityAnalyzer()
df['sentiment'] = df['cleaned_text'].apply(lambda x: sid.polarity_scores(x)['compound'])

# sentiment analysis
df['sentiment_category'] = df['sentiment'].apply(categorize_sentiment)

# Duygu kategorilerinin frekanslarını hesaplayın
sentiment_counts = df['sentiment_category'].value_counts()

# Duygu kategorilerini görselleştirme
plt.figure(figsize=(10, 6))
sentiment_counts.plot(kind='bar', color=['green', 'red', 'blue'])
plt.title('Duygu Kategorileri')
plt.xlabel('Duygu İfadeleri')
plt.ylabel('Frekans')
plt.show()

#summaroy of issues
issues = ['war', 'death', 'betrayal', 'power', 'fear', 'love', 'loyalty']
issue_counts = {issue: all_words.count(issue) for issue in issues}

plt.figure(figsize=(10, 6))
plt.bar(issue_counts.keys(), issue_counts.values())
plt.title('Game of Thrones Dünyasındaki Sorunlar')
plt.xlabel('Sorunlar')
plt.ylabel('Frekans')
plt.show()

