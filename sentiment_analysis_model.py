# %%
import pandas as pd

# Load dataset
df = pd.read_csv("D:\BA Notes\Projects\Sentiment Analysis on Customer Reviews\Reviews\Reviews.csv")

# Preview data
df.head()

# %%
print("Total number of rows:", len(df))

# %%
import nltk
nltk.download('stopwords')

# %%

# 🧹 Text Preprocessing
# cleaning text

import string
import re
from nltk.corpus import stopwords

stop_words = set(stopwords.words('english'))

def preprocess(text):
    text = text.lower()
    text = re.sub(r"http\S+", "", text)  # remove URLs
    text = re.sub(r'\d+', '', text)      # remove digits
    text = text.translate(str.maketrans('', '', string.punctuation))  # remove punctuation
    words = text.split()
    words = [word for word in words if word not in stop_words]
    return ' '.join(words)

df['clean_text'] = df['Text'].astype(str).apply(preprocess)

# %%
#sentiment analysis block using VADER (Valence Aware Dictionary and sEntiment Reasoner)
# | Step                           | Purpose                     | Output                                            |
# | ------------------------------ | --------------------------- | ------------------------------------------------- |
# | `SentimentIntensityAnalyzer()` | Loads sentiment engine      | Ready to analyze text                             |
# | `sia.polarity_scores()`        | Calculates sentiment values | `{'pos':0.3,'neu':0.6,'neg':0.1,'compound':0.25}` |
# | `get_sentiment()`              | Converts score to label     | “Positive”, “Neutral”, or “Negative”              |
# | Final Output                   | Adds columns to `df`        | `sentiment_score` + `sentiment`                   |

from nltk.sentiment import SentimentIntensityAnalyzer
import nltk
nltk.download('vader_lexicon')

sia = SentimentIntensityAnalyzer()

# Apply sentiment scoring
df['sentiment_score'] = df['clean_text'].apply(lambda x: sia.polarity_scores(x)['compound'])

# Classify sentiment
def get_sentiment(score):
    if score >= 0.05:
        return 'Positive'
    elif score <= -0.05:
        return 'Negative'
    else:
        return 'Neutral'

df['sentiment'] = df['sentiment_score'].apply(get_sentiment)

# %%
df.head()
# %%
#Sentiment by Product
# this line of code is a grouped analysis that tells you the sentiment distribution per product in the dataset.
sentiment_by_product = df.groupby('ProductId')['sentiment'].value_counts(normalize=True).unstack().fillna(0)

# %%
print(sentiment_by_product)

# %%
#showing Positive vs Negative Word Clouds side by side gives a powerful visual comparison of what customers love versus complain about.
from wordcloud import WordCloud
import matplotlib.pyplot as plt

# Create separate text strings for each sentiment
positive_text = ' '.join(df[df['sentiment'] == 'Positive']['clean_text'])
negative_text = ' '.join(df[df['sentiment'] == 'Negative']['clean_text'])

# Generate the word clouds
wc_positive = WordCloud(width=800, height=400, colormap='Greens').generate(positive_text)
wc_negative = WordCloud(width=800, height=400, colormap='Reds').generate(negative_text)

# Create a figure with 2 subplots side by side
plt.figure(figsize=(16, 8))

# Positive reviews word cloud
plt.subplot(1, 2, 1)
plt.imshow(wc_positive, interpolation='bilinear')
plt.axis('off')
plt.title('Top Words in Positive Reviews', fontsize=16, fontweight='bold')

# Negative reviews word cloud
plt.subplot(1, 2, 2)
plt.imshow(wc_negative, interpolation='bilinear')
plt.axis('off')
plt.title('Top Words in Negative Reviews', fontsize=16, fontweight='bold')

# Show both plots
plt.tight_layout()
plt.show()

# %%
# from wordcloud import WordCloud
# import matplotlib.pyplot as plt

# # Positive Reviews WordCloud
# positive_text = ' '.join(df[df['sentiment'] == 'Positive']['clean_text'])
# wc_p = WordCloud(width=800, height=400).generate(positive_text)

# plt.figure(figsize=(10, 5))
# plt.imshow(wc_p, interpolation='bilinear')
# plt.axis('off')
# plt.title('Top Words in Positive Reviews')
# plt.show()

# %%
# from wordcloud import WordCloud
# import matplotlib.pyplot as plt

# # Negative Reviews WordCloud
# negative_text = ' '.join(df[df['sentiment'] == 'Negative']['clean_text'])
# wc_n = WordCloud(width=800, height=400).generate(negative_text)

# plt.figure(figsize=(10, 5))
# plt.imshow(wc_n, interpolation='bilinear')
# plt.axis('off')
# plt.title('Top Words in Negative Reviews')
# plt.show()

# %%
import seaborn as sns
import matplotlib.pyplot as plt

# Sentiment Distribution
# Pie Chart-Sentiment Distribution-How many reviews are Positive, Neutral, Negative
df['sentiment'].value_counts().plot.pie(autopct='%1.1f%%', title="Overall Sentiment Distribution")
plt.show()

# Sentiment by Rating
# Bar Chart-Sentiment vs Rating-Whether sentiment matches user ratings
sns.barplot(x='Score', y='sentiment_score', data=df)
plt.title('Average Sentiment Score by Rating')
plt.show()

# Sentiment Over Time
# Line Chart-Sentiment Over Time-How customer mood changes month to month
df['review_time'] = pd.to_datetime(df['Time'], unit='s')
df.set_index('review_time', inplace=True)
df.resample('M')['sentiment_score'].mean().plot(title='Sentiment Trend Over Time')
plt.ylabel('Average Sentiment Score')
plt.show()


# %%
