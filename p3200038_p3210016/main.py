import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.feature_extraction.text import CountVectorizer
from NaiveBayes import BernoulliNaiveBayes # Make sure your NaiveBayes class is correctly imported
from sklearn.metrics import classification_report
from sklearn.naive_bayes import BernoulliNB


# Load IMDB dataset
(x_train_imdb, y_train_imdb), (x_test_imdb, y_test_imdb) = tf.keras.datasets.imdb.load_data()

# Concatenate training and testing sets for splitting
x = np.concatenate((x_train_imdb, x_test_imdb), axis=0)
y = np.concatenate((y_train_imdb, y_test_imdb), axis=0)

# Split the data into 80% training and 20% testing
x_train_imdb, x_test_imdb, y_train_imdb, y_test_imdb = train_test_split(x, y, test_size=0.2, random_state=42)

# Convert word indices to words
word_index = tf.keras.datasets.imdb.get_word_index()
index2word = dict((i + 3, word) for (word, i) in word_index.items())
index2word[0] = '[pad]'
index2word[1] = '[bos]'
index2word[2] = '[oov]'
x_train_imdb = np.array([' '.join([index2word[idx] for idx in text]) for text in x_train_imdb])
x_test_imdb = np.array([' '.join([index2word[idx] for idx in text]) for text in x_test_imdb])

# Vectorize text data
binary_vectorizer = CountVectorizer(binary=True, max_df=0.85, min_df=100)
x_train_imdb_binary = binary_vectorizer.fit_transform(x_train_imdb)
x_test_imdb_binary = binary_vectorizer.transform(x_test_imdb)
print('Vocabulary size:', len(binary_vectorizer.vocabulary_))

# Convert to numpy arrays
x_train_imdb_binary = x_train_imdb_binary.toarray()
x_test_imdb_binary = x_test_imdb_binary.toarray()

# for storing metrics
training_acc = []
sklearn_training_acc = []
test_acc = []
sklearn_test_acc = []
prec = []
sklearn_prec = []
rec = []
sklearn_rec = [] 
f = []
sklearn_f = []

# # of samples for training in each iteration
i = 4000
counter = 1

while (i <= len(y_train_imdb)):
    print(" ")
    
    # Initialize p3210016 NaiveBayes model & also the sklearn model
    clf = BernoulliNaiveBayes()
    clf_sklearn = BernoulliNB()
    
    # Prepare training data
    x_train = x_train_imdb_binary[:i,:].copy()
    y_train = y_train_imdb[:i].copy()
    
    # Train p3210016 model
    clf.fit(x_train, y_train)
    #train sklearn model
    clf_sklearn.fit(x_train, y_train)
    
    # Prepare training data
    x_train = x_train_imdb_binary[:i,:].copy()
    y_train = y_train_imdb[:i].copy()
    
    clf.fit(x_train, y_train)
    clf_sklearn.fit(x_train, y_train)
    
    # Predictions k gia ta 2 models
    predicted_labels_train = clf.predict(x_train)
    predicted_labels_test = clf.predict(x_test_imdb_binary)
    
    sklearn_predicted_labels_train = clf_sklearn.predict(x_train)
    sklearn_predicted_labels_test = clf_sklearn.predict(x_test_imdb_binary)
    
    # yplogismos metrikwn
    accuracy_train = accuracy_score(y_train, predicted_labels_train)
    accuracy_test = accuracy_score(y_test_imdb, predicted_labels_test)
    precision = precision_score(y_test_imdb, predicted_labels_test)
    recall = recall_score(y_test_imdb, predicted_labels_test)
    f1 = f1_score(y_test_imdb, predicted_labels_test)
    
    sklearn_accuracy_train = accuracy_score(y_train, sklearn_predicted_labels_train)
    sklearn_accuracy_test = accuracy_score(y_test_imdb, sklearn_predicted_labels_test)
    sklearn_precision = precision_score(y_test_imdb, sklearn_predicted_labels_test)
    sklearn_recall = recall_score(y_test_imdb, sklearn_predicted_labels_test)
    sklearn_f1 = f1_score(y_test_imdb, sklearn_predicted_labels_test)
    
    # Append metrics to the respective lists
    training_acc.append(accuracy_train)
    sklearn_training_acc.append(sklearn_accuracy_train)
    test_acc.append(accuracy_test)
    sklearn_test_acc.append(sklearn_accuracy_test)
    prec.append(precision)
    sklearn_prec.append(sklearn_precision)
    rec.append(recall)
    sklearn_rec.append(sklearn_recall)
    f.append(f1)
    sklearn_f.append(sklearn_f1)
    
    # Print summary for 3210016 BernoulliNaiveBayes model
    print("3210016 Bernoulli Naive Bayes model summary:")
    print(f"Accuracy in training set: {accuracy_train:.2f}")
    print(f"Accuracy in test set: {accuracy_test:.2f}")
    print(f"Precision: {precision:.2f}")
    print(f"Recall: {recall:.2f}")
    print(f"F1 Score: {f1:.2f}")

    # Print summary for sklearn model
    print("\nSklearn Bernoulli Naive Bayes model summary:")
    print(f"Accuracy in training set: {sklearn_accuracy_train:.2f}")
    print(f"Accuracy in test set: {sklearn_accuracy_test:.2f}")
    print(f"Precision: {sklearn_precision:.2f}")
    print(f"Recall: {sklearn_recall:.2f}")
    print(f"F1 Score: {sklearn_f1:.2f}")
    # Prepare for the next epoch
    i += 4000
    counter+=1
    print("-------------------------------\n")


print(classification_report(y_test_imdb, clf_sklearn.predict(x_test_imdb_binary)))

# Visualization p320016 implementation accuracy
# Note: epochs need to be defined based on the actual number of training samples used
#epochs = [4000, 8000, 12000, 16000, 20000, 24000, 28000, 32000, 36000, 40000]
epochs = list(range(4000, len(y_train_imdb) + 1, 4000))

xpoints = epochs
ypoints = training_acc
plt.plot(xpoints, ypoints, label="Training Data", color="green")
y1points = test_acc
plt.plot(xpoints, y1points, label="Test Data", color="red")
plt.xlabel("Samples")
plt.ylabel("Accuracy")
plt.title("Comparison of accuracy in Training and Test Data")
plt.legend()
plt.grid(True)
plt.show()

#p3210016 implementation precision
xpoints=epochs
ypoints=prec
# Plot precision
plt.figure(figsize=(10, 5))
plt.plot(xpoints, prec, label="Precision", marker='o')
plt.title("Bernoulli Naive Bayes: Precision over Number of Training Samples")
plt.xlabel("Number of Training Samples")
plt.ylabel("Precision")
plt.legend()
plt.grid(True)
plt.show()

# Plot recall
xpoints=epochs
ypoints=rec
plt.figure(figsize=(10, 5))
plt.plot(xpoints, rec, label="Recall", marker='o')
plt.title("Bernoulli Naive Bayes: Recall over Number of Training Samples")
plt.xlabel("Number of Training Samples")
plt.ylabel("Recall")
plt.legend()
plt.grid(True)
plt.show()

# Plot F1-Score
plt.figure(figsize=(10, 5))
plt.plot(xpoints, f, label="F1 Score", marker='o')
plt.title("Bernoulli Naive Bayes: F1 Score over Number of Training Samples")
plt.xlabel("Number of Training Samples")
plt.ylabel("F1 Score")
plt.legend()
plt.grid(True)
plt.show()


#visualize p3210016 implementation recall
xpoints=epochs
ypoints=rec
plt.plot(xpoints,ypoints,label= "Test Data",color="red")
plt.xlabel("Samples")
plt.ylabel("Recall")
plt.title("Recall in Test Data")
plt.show()

#visualize p3210016 implementation F1
xpoints=epochs
ypoints=f
plt.plot(xpoints,ypoints,label= "Test Data",color="orange")
plt.xlabel("Samples")
plt.ylabel("F1")
plt.title("F1 in Test Data")
plt.show()

#confusion matrix
confusion_matrix=confusion_matrix(y_test_imdb,predicted_labels_test)
cm_display=ConfusionMatrixDisplay(confusion_matrix=confusion_matrix,display_labels=[False,True])
cm_display.plot()
plt.show()

#translate a review from binary to english
#reurn np.array with indices of words in vocab
def translate(array,i):
    out_ind = np.transpose(np.nonzero(array[i]))
    return out_ind 