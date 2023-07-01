# airbnb_project
# This is a code use NLTK to generate each line of comments to rating from 1 to 5
import pandas as pd
import re
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer


# Load the data
data = pd.read_excel(r"reviews_boston.csv.xlsx",sheet_name="reviews")


# Define a function to preprocess the text
def preprocess_text(text):
    # Convert the text to lowercase
    text = str(text).lower()
    # Remove numbers and punctuation
    text = re.sub(r'\d+', '', text)
    text = re.sub(r'[^\w\s]', '', text)
    # Remove whitespace
    text = text.strip()
    return text


# Handle NaN values in the comments column
data['comments'] = data['comments'].fillna('')


# Apply the preprocessing function to the comments column
data['comments'] = data['comments'].apply(preprocess_text)


# Initialize the sentiment analyzer from NLTK
nltk.download('vader_lexicon')
sia = SentimentIntensityAnalyzer()


# Define a function to calculate sentiment scores using the NLTK sentiment analyzer
def get_sentiment(text):
    sentiment = sia.polarity_scores(text)
    return sentiment['compound']


# Apply the sentiment analysis function to the comments column
data['sentiment'] = data['comments'].apply(get_sentiment)


# Define a function to map the sentiment score to a rating (1-5)
def map_rating(sentiment):
    if sentiment >= 0.5:
        return 5
    elif sentiment >= 0.0:
        return 4
    elif sentiment >= -0.5:
        return 3
    elif sentiment >= -1.0:
        return 2
    else:
        return 1


# Apply the rating mapping function to the sentiment column
data['rating'] = data['sentiment'].apply(map_rating)


# Export the result to an Excel file
data.to_excel(r"sentiment_results.xlsx", index=False)
