# 
#Source code for toxic Comment Classification Dataset of Kaggle
#

# packages and import statements
import pandas as pd
from itertools import cycle
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from nltk.corpus import stopwords 
from scipy import sparse
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
import re, string
from sklearn.metrics import log_loss
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Conv1D, GlobalMaxPooling1D, Dense, Dropout, Activation, Embedding
from keras.preprocessing import text, sequence

# Readscv files
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

label_list = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
label_train = train[label_list]

#stopwords removal function
stopwords_list = stopwords.words('english')

def stopwords_removal(text_sentence):
    sw_list = stopwords.words('english')
    words = text_sentence.split()
    sw_removed_words = [word for word in words if (word not in sw_list) and len(word) > 1]
    return " ".join(sw_removed_words)

def tokenize_text(text):
	punctuations = re.compile(f'([{string.punctuation}“”¨«»®´·º½¾¿¡§£₤‘’])')
	return punctuations.sub(r' \1 ', text).split()
	

# Function for ebsemble model building techniques (NB-SVM, Gradient Boosting, Random Forest)
def ensemble_methods():
	
	classifier_list = [RandomForestClassifier(criterion = 'entropy', min_samples_leaf = 8, min_samples_split = 2,  max_depth = 15), GradientBoostingClassifier(n_estimators = 200, max_depth = 20, min_samples_leaf = 5, min_impurity_decrease = 3), LogisticRegression(C = 5, dual = True)]
	classifier_name_list = ["Random Forest", "Gradient Boost", "Logistic"]
	tfid_vector = TfidfVectorizer(ngram_range = (1,2), strip_accents = 'unicode', tokenizer = tokenize_text, min_df = 3, max_df = 0.9, use_idf = 1, smooth_idf = 1, sublinear_tf = 1)
	train_tfid_vector = tfid_vector.fit_transform(train["comment_text"])
	test_tfid_vector = tfid_vector.transform(test["comment_text"])
	X_train_target = train_tfid_vector
	X_test = test_tfid_vector
	
	# Function to calculate probabilties
	def calculate(y_class, y):
		i = X_train[y == y_class].sum(0)
		return (i + 1) / ((y == y_class).sum()+1)

	# Function to build NB-SVM
	def get_logistic_model(y, classifier):
	    y = y.values
	    r = sparse.csr_matrix(np.log(calculate(1,y) / calculate(0,y)))	   
	    X_naivebayes_train = X_train.multiply(r)
	    return classifier.fit(X_naivebayes_train, y)   

	# Function to fit teh classifier
	def fit_classifier(y, clf):
	    return clf.fit(X_train, y)


	X_train, X_valid, y_train, y_valid = train_test_split(X_train_target, label_train, test_size=0.4)
	predicted_test = np.zeros((len(test), len(label_list)))
	train_loss = []
	valid_loss = []
	cv_scores = []
	accuracy = []
	for name, classifier in zip(classifier_name_list, classifier_list):
		print("+"*30)
		print("Classifier is ", name)
		print("+"*30)
		fpr = dict()
		tpr = dict()
		roc_auc = dict()
		c_name = dict()
		for count, class_name in enumerate(label_list):
			y_train_col = train[class_name].values
			preds_train = np.zeros([X_train.shape[0], 1], dtype = int) 
			preds_valid = np.zeros([X_valid.shape[0], 1], dtype = int)
			print('Fitting Label, ' + class_name + ':')
			if name == "Logistic":
				classifier = get_logistic_model(y_train[class_name], classifier)
			else:
				classifier = fit_classifier(y_train[class_name], classifier)

			cv_score = np.mean(cross_val_score(classifier, X_train_target, y_train_col, cv=3, scoring ='roc_auc'))
			cv_scores.append(cv_score)
			print('CV score for class {} is {}'.format(class_name, cv_score))

			preds_valid = classifier.predict_proba(X_valid)[:,1]
			preds_train = classifier.predict_proba(X_train)[:,1]
			train_loss_class = log_loss(y_train[class_name], preds_train)
			valid_loss_class = log_loss(y_valid[class_name], preds_valid)
			print('Trainloss = log loss:', train_loss_class)
			print('Validloss = log loss:', valid_loss_class)
			
			predicted_test[:,count] = classifier.predict_proba(X_test)[:,1]

			fpr[count], tpr[count], threshold = roc_curve(y_valid.iloc[:, count], preds_valid)
			roc_auc[count] = auc(fpr[count], tpr[count])
			c_name[count] = class_name

			train_loss.append(train_loss_class)
			valid_loss.append(valid_loss_class)
			
			train_predictions = classifier.predict(X_valid)		
			acc = accuracy_score(y_valid[class_name], train_predictions )
			accuracy.append(acc)
			print("Accuracy: {:.4%}".format(acc))
			print("-"*30)

		print('Mean Log loss of training dataset', np.mean(train_loss))
		print('Mean Log loss of validation dataset', np.mean(valid_loss))
		print('Mean CV score is {}'.format(np.mean(cv_scores)))
		print('Mean accuracy is {:.4%}'.format(np.mean(accuracy)))
		print("#"*30)

		# Plot for ROC 
		plt.figure()
		lw = 2
		colors = cycle(['red', 'blue', 'green', 'orange', 'magenta', 'yellow'])
		
		for i, color in zip(range(len(label_list)), colors):
		        plt.plot(fpr[i], tpr[i], color=color, lw=lw, label='ROC curve of class {} (area = {:0.2f})'.format(c_name[i], roc_auc[i]))
		plt.plot([0, 1], [0, 1], 'k--', lw=lw)
		plt.xlim([0.0, 1.0])
		plt.ylim([0.0, 1.05])
		plt.xlabel('False Positive Rate')
		plt.ylabel('True Positive Rate')
		plt.title('ROC curve for classifier: ' + str(name))
		plt.legend(loc="lower right")
		plt.show()

# Sequential Model building using keras

def sequential_model():
#preprocessing for removing stopwords
	x = train['comment_text'].apply(stopwords_removal)
	label_train = train[label_list].values
	x_test = test['comment_text'].apply(stopwords_removal)
	max_features = 20000
	max_text_length = 300
	dimensions_initial = 50
	filters = 250
	kernel_size = 3
	dimensions_final = 100
	batch_count = 32
	nEpochs = 1

	#tokenizer
	x_tokenizer = text.Tokenizer(num_words=max_features)
	x_tokenizer.fit_on_texts(list(x))
	x_tokenized = x_tokenizer.texts_to_sequences(x)
	x_test_tokenized = x_tokenizer.texts_to_sequences(x_test)
	x_train_val = sequence.pad_sequences(x_tokenized, maxlen=max_text_length)
	x_testing = sequence.pad_sequences(x_test_tokenized, maxlen=max_text_length)

	#train - validation split
	x_train, x_validation, y_train, y_validation = train_test_split(x_train_val, label_train, test_size=0.1, random_state=1)


	#defining sequential model
	classifier_model = Sequential()
	classifier_model.add(Embedding(max_features, dimensions_initial, input_length=max_text_length))
	classifier_model.add(Dropout(0.2))
	classifier_model.add(Conv1D(filters, kernel_size, activation='relu',  padding='valid', strides=1))
	classifier_model.add(GlobalMaxPooling1D())
	classifier_model.add(Dense(dimensions_final))
	classifier_model.add(Dropout(0.2))
	classifier_model.add(Activation('sigmoid'))
	classifier_model.add(Dense(6))
	classifier_model.add(Activation('relu'))


	#compilation and metrics definition
	classifier_model.compile(optimizer='rmsprop', metrics=['accuracy'], loss='binary_crossentropy')
	classifier_model.summary()
	classifier_model.fit(x_train, y_train, batch_size=batch_count, epochs=nEpochs, verbose=1,
	                     validation_data=(x_validation, y_validation))


	#predict model on testing data set
	y_prediction = classifier_model.predict(x_testing, verbose=1)


	#validation feature and label
	val_pred = classifier_model.predict_proba(x_validation)
	val_true = y_validation


	#read sample submission csv and write predicted values to a new csv
	sample_submission = pd.read_csv("sample_submission.csv")
	sample_submission[label_list] = y_prediction
	sample_submission.to_csv("results.csv", index=False)

if __name__ == "__main__":
	ensemble_methods()
	sequential_model()



