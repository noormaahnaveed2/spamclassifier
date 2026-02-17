#pip install pandas scikit-learn
# Import necessary libraries
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt

# Load the dataset (you can download the 'spam.csv' from Kaggle or any other source)
# Replace the path with the correct path to your dataset
data = pd.read_csv('emails.csv', encoding='latin-1')

# Preprocess data: Select relevant columns
data = data[['v1', 'v2']]  # 'v1' is the label (spam/ham), 'v2' is the message
data.columns = ['Label', 'Message']

# Clean the text data (you can expand this step to remove more unwagffsnted characters)
data['Message'] = data['Message'].str.replace(r'\W', ' ')  # Remove non-word characters
data['Message'] = data['Message'].str.lower()  # Convert all text to lowercase

# Split the data into features (X) and target (y)
X = data['Message']  # Text messages
y = data['Label']    # Spam or Ham labels

# Split the dataset into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Vectorize the text data: Convert text messages to numerical values using TF-IDF
vectorizer = TfidfVectorizer(stop_words='english')  # Removing common English stopwords
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# Train a Naive Bayes Classifier on the training data
nb_classifier = MultinomialNB()
nb_classifier.fit(X_train_tfidf, y_train)

# Make predictions on the test data
y_pred = nb_classifier.predict(X_test_tfidf)

# Evaluate the model's performance
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy * 100:.2f}%')

# Display a more detailed classification report (precision, recall, f1-score)
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Display a confusion matrix to see how many messages were correctly/incorrectly classified
cm = confusion_matrix(y_test, y_pred)
print("\nConfusion Matrix:")
print(cm)

# Plot the confusion matrix
plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
plt.title('Confusion Matrix')
plt.colorbar()
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.xticks([0, 1], ['Ham', 'Spam'])
plt.yticks([0, 1], ['Ham', 'Spam'])
plt.show()


#python spam_classifier.py
