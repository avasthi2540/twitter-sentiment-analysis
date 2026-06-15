# Twitter Sentiment Analysis Project

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

print("\n=== Twitter Sentiment Analysis Project ===\n")

# Sample Dataset
data = {
    "text": [
        "I love this product",
        "This is the worst experience",
        "Amazing service",
        "I hate this",
        "Very good and helpful",
        "Not good at all",
        "Excellent work",
        "Terrible experience",
        "I am very happy",
        "I am disappointed",
        "Great customer support",
        "Bad quality product",
        "Fantastic experience",
        "Waste of money",
        "Highly recommended",
        "Very poor service"
    ],
    "sentiment": [
        "positive",
        "negative",
        "positive",
        "negative",
        "positive",
        "negative",
        "positive",
        "negative",
        "positive",
        "negative",
        "positive",
        "negative",
        "positive",
        "negative",
        "positive",
        "negative"
    ]
}

# Create DataFrame
df = pd.DataFrame(data)

# Convert text into numerical features
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(df["text"])
y = df["sentiment"]

# Split data into training and testing
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Train Model
model = MultinomialNB()
model.fit(X_train, y_train)

# Evaluate Model
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print(f"Model Accuracy: {accuracy * 100:.2f}%")

# Interactive Prediction
print("\nEnter text to analyze sentiment.")
print("Type 'exit' to quit.\n")

while True:
    user_input = input("Enter Text: ")

    if user_input.lower() == "exit":
        print("Program terminated.")
        break

    prediction = model.predict(
        vectorizer.transform([user_input])
    )

    print("Predicted Sentiment:", prediction[0])
    print("-" * 40)
