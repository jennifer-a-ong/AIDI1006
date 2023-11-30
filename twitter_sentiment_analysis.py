import pandas as pd
from textblob import TextBlob

# Load the CSV file into a DataFrame
df = pd.read_csv('twitter.csv')

# Replace 'text' with the actual column name containing the text you want to analyze
column_name = 'clean_text'

# Function to perform sentiment analysis and return the sentiment label
def analyze_sentiment(text):
    blob = TextBlob(str(text))
    sentiment_score = blob.sentiment.polarity
    return sentiment_score

# Apply sentiment analysis to each entry in the specified column
df['score'] = df[column_name].apply(analyze_sentiment)

# Convert 'score' column to numeric
df['score'] = pd.to_numeric(df['score'], errors='coerce')

# Create a new 'sentiment' column based on the 'score' column
df['sentiment'] = pd.cut(df['score'], bins=[-1, -0.01, 0.01, 1], labels=['Negative', 'Neutral', 'Positive'], include_lowest=True)

# Display the DataFrame with the new columns
print(df)

# Save the DataFrame with the new columns to a new CSV file
df.to_csv('twitter_with_sentiment.csv', index=False)