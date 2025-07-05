# AI
The project focuses on classifying texts into two categories (positive/negative opinion) using machine learning algorithms implemented from scratch, without the use of off-the-shelf classifier libraries.  This known dataset was used(https://ai.stanford.edu/~amaas/data/sentiment/)

In the first part, the following algorithms were implemented: naive Bayes (in multivariate Bernoulli form) & Logistic Regression with stochastic gradient ascent and added regularization. The texts are represented as binary word vectors, based on vocabulary extracted from the training data, and controlled by hyperparameters m (most frequent words), n (skipped common words) and k (skipped rare words). Experiments are included that present learning curves with accuracy on the training and control data, as well as precision, recall and F1 curves for one of the two categories as a function of the number of training examples. The selection of hyperparameters based on separate development sets is also documented.

In the second part, the above implementations are compared with corresponding ready-made implementations of the same or similar algorithms (via Scikit-learn). The comparisons are made with the same property vectors and vocabulary as in the first part, and the same curves and performance tables are presented.

In the third part, the results of the previous methods are compared with neural networks (RNN) implemented using TensorFlow/Keras. MLPs are based on average word embeddings, while RNNs exploit either their final state or self-attention mechanism. Accuracy, precision, recall, F1 and additional loss curves per training epoch are included.

The project also includes technical details such as preprocessing of texts with ready-made libraries, visualization of results with matplotlib, and implementation of the algorithms entirely from scratch.
