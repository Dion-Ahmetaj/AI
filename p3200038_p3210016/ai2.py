import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.feature_extraction.text import CountVectorizer
from logisticregression import logisticregression
from sklearn.metrics import accuracy_score

# ------------------- p3200038 implementation--------------------
#------------------MEROS A with MEROS B--------------------------

(x_train_imdb, y_train_imdb), (x_test_imdb, y_test_imdb) = tf.keras.datasets.imdb.load_data()

#Concatenate training and testing sets for splitting
x = np.concatenate((x_train_imdb, x_test_imdb), axis=0)
y = np.concatenate((y_train_imdb, y_test_imdb), axis=0)

# Split the data into 80% training and 20% testing
x_train_imdb, x_test_imdb, y_train_imdb, y_test_imdb = train_test_split(x, y, test_size=0.2, random_state=42)



word_index = tf.keras.datasets.imdb.get_word_index()
index2word = dict((i + 3, word) for (word, i) in word_index.items())
index2word[0] = '[pad]'
index2word[1] = '[bos]'
index2word[2] = '[oov]'
x_train_imdb = np.array([' '.join([index2word[idx] for idx in text]) for text in x_train_imdb])
x_test_imdb = np.array([' '.join([index2word[idx] for idx in text]) for text in x_test_imdb])
#print(x_train_imdb.shape)



# min-df --> when building the vocabulary ignore terms that have a document frequency strictly lower than the given threshold.
binary_vectorizer = CountVectorizer(binary=True,max_df=0.85,min_df=100)
x_train_imdb_binary = binary_vectorizer.fit_transform(x_train_imdb)
x_test_imdb_binary = binary_vectorizer.transform(x_test_imdb)
print(
    'Vocabulary size:', len(binary_vectorizer.vocabulary_)
)

#all the words in vocabulary
vocab=binary_vectorizer.get_feature_names_out()

#np.arrays
x_train_imdb_binary = x_train_imdb_binary.toarray()
x_test_imdb_binary = x_test_imdb_binary.toarray()



#10 epochs
#accuracy
training_acc=[]
test_acc=[]
training_acc_sk=[]
test_acc_sk=[]
#precision
prec=[]
prec_sk=[]
#recall
rec=[] 
rec_sk=[]
#F1
f=[]
f_sk=[]

i=4000
j=1




while (i<=len(y_train_imdb)):
    print(" ")
    #CREATE instance of p3200038 model
    clf=logisticregression(size_of_voc=len(binary_vectorizer.vocabulary_))

    # Create instance of sklearn Logistic Regression model
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import classification_report

    log = LogisticRegression()

    x_train=x_train_imdb_binary[:i,:].copy()
    y_train=y_train_imdb[:i].copy()
    
    #p3200038 model_Logistic_Regression
    clf.fit(x_train,y_train)
    #sklearn logistic regression model
    log.fit(x_train, y_train)

    #training data
    predicted_probs_train = clf.predict(x_train, clf.weights)
    predicted_labels_train = np.where(predicted_probs_train > 0.5, 1, 0)
    training_acc.append(len(predicted_labels_train[predicted_labels_train == np.array(y_train).reshape(-1,1)]) / len(y_train)*100)
    print(f"p3200038 implementation of logistic regression accuracy TRAINING DATA is {len(predicted_labels_train[predicted_labels_train == np.array(y_train).reshape(-1,1)]) / len(y_train)*100:.2f}")
    #sklearn logisitc regression model
    predicted_probs_train_sk = log.predict(x_train)
    training_acc_sk.append(accuracy_score(y_train,predicted_probs_train_sk))
    
    
    #test data
    predicted_probs = clf.predict(x_test_imdb_binary, clf.weights)
    predicted_labels = np.where(predicted_probs > 0.5, 1, 0)
    test_acc.append(len(predicted_labels[predicted_labels == np.array(y_test_imdb).reshape(-1,1)]) / len(y_test_imdb)*100)
    precision=precision_score(y_test_imdb,predicted_labels)
    prec.append(precision)
    recall=recall_score(y_test_imdb,predicted_labels)
    rec.append(recall)
    f1=2 * (precision*recall)/(precision+recall)
    f.append(f1)
    print(f"P3200038 implementation of logistic regression accuracy TEST DATA is {len(predicted_labels[predicted_labels == np.array(y_test_imdb).reshape(-1,1)]) / len(y_test_imdb)*100:.2f}")
    
    #sklearn logisitc regression model
    predicted_labels_sk= log.predict(x_test_imdb_binary)
    test_acc_sk.append(accuracy_score(y_test_imdb,predicted_labels_sk))
    precision_sk=precision_score(y_test_imdb,predicted_labels_sk)
    prec_sk.append(precision_sk)
    recall_sk=recall_score(y_test_imdb,predicted_labels_sk)
    rec_sk.append(recall_sk)
    f1_sk=2 * (precision_sk*recall_sk)/(precision_sk+recall_sk)
    f_sk.append(f1_sk)
    
    
    # print summary
    print("p3200038 logistic regression model")
    print("number of training samples: "+str(i)+" summary: ")
    print ("accuracy in trainig set " + f"is {len(predicted_labels_train[predicted_labels_train == np.array(y_train).reshape(-1,1)]) / len(y_train)*100:.2f}")
    print ("accuracy in test set " +f"is {len(predicted_labels[predicted_labels == np.array(y_test_imdb).reshape(-1,1)]) / len(y_test_imdb)*100:.2f}" )
    print("precision : "+ str(precision))
    print("recall : " +str(recall))
    print ("F1 : " +str(f1))

    
    
    print("sk learn logistic regression model")
    print("number of training samples: "+str(i)+" summary: ")
    print ("accuracy in trainig set " + f"is {accuracy_score(y_train,predicted_probs_train_sk):.2f}")
    print ("accuracy in test set " +f"is {accuracy_score(y_test_imdb,predicted_labels_sk):.2f}" )
    print("precision : "+ str(precision_sk))
    print("recall : " +str(recall_sk))
    print ("F1 : " +str(f1_sk))
    
    #next epoch
    i+=4000
    j+=1
    print("-------------------------------")
    print("  ")

print(classification_report(y_test_imdb, log.predict(x_test_imdb_binary)))

#visualize p3200038 implementation accuracy
#epochs =[4000,8000,12000,16000,20000,24000,28000,32000,36000,40000]
epochs=[20000,40000]
xpoints=epochs
ypoints=training_acc
plt.plot(xpoints,ypoints,label= "Training Data",color="green")
y1points=test_acc
plt.plot(xpoints,y1points,label= "Test Data", color="red")
plt.xlabel("Samples")
plt.ylabel("Accuracy")
plt.title("Comparison of accuracy in Training and Test Data")
plt.legend()
plt.grid(True)
plt.show()

#visualize p3200038 implementation precision
xpoints=epochs
ypoints=prec
plt.plot(xpoints,ypoints,label= "Test Data",color="blue")
plt.xlabel("Samples")
plt.ylabel("Precision")
plt.title("Precision in Test Data")
plt.show()

#visualize p3200038 implementation recall
xpoints=epochs
ypoints=rec
plt.plot(xpoints,ypoints,label= "Test Data",color="red")
plt.xlabel("Samples")
plt.ylabel("Recall")
plt.title("Recall in Test Data")
plt.show()

#visualize p3200038 implementation F1
xpoints=epochs
ypoints=f
plt.plot(xpoints,ypoints,label= "Test Data",color="orange")
plt.xlabel("Samples")
plt.ylabel("F1")
plt.title("F1 in Test Data")
plt.show()

#confusion matrix
confusion_matrix=confusion_matrix(y_test_imdb,predicted_labels)
cm_display=ConfusionMatrixDisplay(confusion_matrix=confusion_matrix,display_labels=[False,True])
cm_display.plot()
plt.show()


#------------------MEROS B-----------------------


#visualize sklearn implementation accuracy


ypoints=training_acc_sk
plt.plot(xpoints,ypoints,label= "Training Data",color="green")
y1points=test_acc_sk
plt.plot(xpoints,y1points,label= "Test Data", color="red")
plt.xlabel("Samples")
plt.ylabel("Accuracy")
plt.title("SK LEARN Comparison of accuracy in Training and Test Data")
plt.legend()
plt.grid(True)
plt.show()

#visualize sklearn implementation precision
xpoints=epochs
ypoints=prec_sk
plt.plot(xpoints,ypoints,label= "Test Data",color="blue")
plt.xlabel("Samples")
plt.ylabel("Precision")
plt.title("SK LEARN Precision in Test Data")
plt.show()

#visualize sklearn implementation recall
xpoints=epochs
ypoints=rec_sk
plt.plot(xpoints,ypoints,label= "Test Data",color="red")
plt.xlabel("Samples")
plt.ylabel("Recall")
plt.title("SK LEARN Recall in Test Data")
plt.show()

#visualize sklearn implementation F1
xpoints=epochs
ypoints=f_sk
plt.plot(xpoints,ypoints,label= "Test Data",color="orange")
plt.xlabel("Samples")
plt.ylabel("F1")
plt.title("SK LEARN F1 in Test Data")
plt.show()

"""
#confusion matrix sklearn

predicted_labels_sk = predicted_labels_sk.reshape((10000, 1))
confusion_matrix_sk=confusion_matrix(y_test_imdb,predicted_labels_sk)
cm_display=ConfusionMatrixDisplay(confusion_matrix=confusion_matrix_sk,display_labels=[False,True])
cm_display.plot()
plt.show()
"""

#------------------MEROS G---------------------------------

#translate a review from binary to english
#reurn np.array with indices of words in vocab
def translate(array,i):
    out_ind = np.transpose(np.nonzero(array[i]))
    return out_ind 
    

#---------------------end of p3200038 impl-----------------