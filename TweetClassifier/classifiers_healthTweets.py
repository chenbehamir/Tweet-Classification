
# coding: utf-8

# In[62]:


import pandas as pd
import numpy as np
import nltk
#nltk.download()
from sklearn.model_selection import train_test_split, KFold
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
import re
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer 
from nltk.stem.porter import PorterStemmer
from nltk.tokenize import TweetTokenizer
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
from gensim.models import KeyedVectors
from sklearn.base import TransformerMixin, BaseEstimator, ClassifierMixin
from sklearn.ensemble import VotingClassifier

##########################################################

def tokenizer_Snowball(text):
            tokens = nltk.wordpunct_tokenize(text)
            stems = []
            for item in tokens:
                    stems.append(SnowballStemmer('english').stem(item))
            return stems





def tokenizer_Porter(text):
            tokens = nltk.wordpunct_tokenize(text)
            stems = []
            for item in tokens:
                    stems.append(PorterStemmer().stem(item))
            return stems

def tokenizer_Tw(text):
    tknzr = TweetTokenizer()
    tokens = tknzr.tokenize(text)
    stems = []
    for item in tokens:
        stems.append(PorterStemmer().stem(item))
    return stems




def make_feature_vec2(data, model, num_features):
    """
    make the feature vecs for all the tweets (the words are from data.processed)
    """
    output = pd.Series([])  # series is the data (data.processed)
    index = 0 # the index of the element in the series 
    for words in data:
        feature_vec = np.zeros((num_features,),dtype="float32") 
        nwords = 0.
        nwords_not_in_set = 0
        index2word_set = set(model.index2word)  # words known to the model
        words_set = words.split(" ")

        for word in words_set:
            if word in index2word_set: 
                nwords = nwords + 1.
                feature_vec = np.add(feature_vec,model[word])
            """
            else:
                nwords_not_in_set = nwords_not_in_set + 1
            """    
        if np.sum(feature_vec) == 0:
            nwords = 1.
        feature_vec = np.divide(feature_vec, nwords)
        #print('missing words in the pre trained data ', nwords_not_in_set)
        feature_vec = pd.Series([feature_vec], index=[index])    #rhe element to add to the series
        index = index + 1  #update the index
        output = output.append(feature_vec)   # add the element to the ouput with the conviniant index
    print("word2vec done ")
    return np.vstack(output)    # vstack used to make X_train able to fit the classifier 





class word2vec_transformer(BaseEstimator, TransformerMixin):
   
	def __init__(self, model, dim):
		self.model = model
		self.dim = dim
        
    
	def transform(self, X, *_):
		return make_feature_vec2(X, self.model, self.dim)
    
    
	def fit(self, *_):
		return self



class my_own_classifier(BaseEstimator, ClassifierMixin):
    
	def __init__(self, clf_tfidf, clf_w2v, y_input):
		"""
		#Called when initializing the classifier
		"""
		self.clf_tfidf = clf_tfidf
		self.clf_w2v = clf_w2v
		self.y_input = y_input
        
	def fit(self, X, y):
		"""
        This should fit classifier. All the "work" should be done here.

        Note: assert is not a good choice here and you should rather
        use try/except blog with exceptions. This is just for short syntax.
		 	"""
	        
		self.clf_tfidf.fit(X, y)
		print("tf-idf fitted")
		self.clf_w2v.fit(X, y)
		print("w2v fitted")
		self.X_ = X
		self.y_ = y
		# Return the classifier
		print("fitting is done")
		return self
        
	def predict(self, X, y=None):
        
		y_pred_tfidf = self.clf_tfidf.predict(X)
		print("tfidf predicted")
		print("y tfidf shape   ",y_pred_tfidf.shape)
		y_pred_w2v =  self.clf_w2v.predict(X)
		print("w2v predicted")
		print("y w2vec shape   ",y_pred_w2v.shape)
		y_pred_voting = np.vstack((y_pred_tfidf, y_pred_w2v)).transpose()   # this is the new X_train
		print("new X_train created")  
		print("y voting shape   ",y_pred_voting.shape)
		print(np.count_nonzero(y_pred_voting))
		logreg = LogisticRegression(class_weight={0:1, 1:2})
		logreg.fit(y_pred_voting, self.y_input)
  
		return logreg.predict(y_pred_voting)

	def predict_proba(self, X, y=None):
		
		y_pred_tfidf = self.clf_tfidf.predict(X)
		print("tfidf predicted")
		print("y tfidf shape   ",y_pred_tfidf.shape)
		y_pred_w2v =  self.clf_w2v.predict(X)
		print("w2v predicted")
		print("y w2vec shape   ",y_pred_w2v.shape)
		y_pred_voting = np.vstack((y_pred_tfidf, y_pred_w2v)).transpose()   # this is the new X_train
		print("new X_train created")  
		print("y voting shape   ",y_pred_voting.shape)
		print(np.count_nonzero(y_pred_voting))
		logreg = LogisticRegression(class_weight={0:1, 1:2})
		logreg.fit(y_pred_voting, self.y_input)
  
		return logreg.predict_proba(y_pred_voting)

class my_own_classifier_svm(BaseEstimator, ClassifierMixin):
    
	def __init__(self, clf_tfidf, clf_w2v, y_input):
		"""
		Called when initializing the classifier
		"""
		self.clf_tfidf = clf_tfidf
		self.clf_w2v = clf_w2v
		self.y_input = y_input
        

	def fit(self, X, y):
		# Check that X and y have correct shape
		#X, y = check_X_y(X, y)
		self.clf_tfidf.fit(X, y)
		print("tf-idf fitted")
		self.clf_w2v.fit(X, y)
		print("w2v fitted")
		self.X_ = X
		self.y_ = y
		 # Return the classifier
		print("fitting is done")
		return self
        
	def predict(self, X, y=None):
        
		y_pred_tfidf = self.clf_tfidf.predict(X)
		print("tfidf predicted")
		print("y tfidf shape   ",y_pred_tfidf.shape)
		y_pred_w2v =  self.clf_w2v.predict(X)
		print("w2v predicted")
		print("y w2vec shape   ",y_pred_w2v.shape)
		y_pred_voting = np.vstack((y_pred_tfidf, y_pred_w2v)).transpose()   # this is the new X_train
		print("new X_train created")  
		print("y voting shape   ",y_pred_voting.shape)
		svm = SVC(class_weight={0:1, 1:1.7}, probability = True)
		svm.fit(y_pred_voting, self.y_input)
   
		return svm.predict(y_pred_voting)


	def predict_proba(self, X, y=None):
        
		y_pred_tfidf = self.clf_tfidf.predict(X)
		print("tfidf predicted")
		print("y tfidf shape   ",y_pred_tfidf.shape)
		y_pred_w2v =  self.clf_w2v.predict(X)
		print("w2v predicted")
		print("y w2vec shape   ",y_pred_w2v.shape)
		y_pred_voting = np.vstack((y_pred_tfidf, y_pred_w2v)).transpose()   # this is the new X_train
		print("new X_train created")  
		print("y voting shape   ",y_pred_voting.shape)
		svm = SVC(class_weight={0:1, 1:1.7}, probability = True)
		svm.fit(y_pred_voting, self.y_input)
   
		return svm.predict_proba(y_pred_voting)




###################################################
data = pd.read_table('/home/ama/chenbeha/PFE/healthandnonhealth', header=None,names=['tweet_id','label','tweet'])
tweets2017filtred = pd.Series.from_csv(path = '/home/ama/chenbeha/PFE/tweets2017filtred1', sep='\t')

#Preprocess the  data

#set of stopwords
stopWords = set(stopwords.words('english'))
# remove punctuation and set to lowercase
data['processed'] = data['tweet'].apply(lambda x: re.sub(r'[^\w\s]','', x.lower()))
# return the number of words foreach tweet
data['count_word'] = data['processed'].apply(lambda x: len(x.split(' ')))
# retunr the number of non stopwords foreach tweet
data['count_not_stopword'] = data['processed'].apply(lambda x: len([t for t in x.split(' ') if t not in stopWords]))



data.label.value_counts()



data['labelNum']= data.label.map({-1:0, 1:1})


positive_tweets = data.loc[data['labelNum'] == 1].count_word.tolist() # tweets with label1
negative_tweets = data.loc[data['labelNum'] == 0].count_word.tolist()


word_freq_pos = dict((x,positive_tweets.count(x)) for x in set(positive_tweets))  # number of words : number of tweets
word_freq_neg = dict((x,negative_tweets.count(x)) for x in set(negative_tweets))


X = data.processed
y = data.labelNum
print(X.shape)
print(y.shape)


# In[149]:


#split X and y into train and test data 70% and 30%
#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3)

# I did this to execute the same training data in all the models 
#X_train.to_csv('XTrain_file', sep='\t')
#y_train.to_csv('yTrain_file', sep='\t')
#X_test.to_csv('XTest_file', sep='\t')
#y_test.to_csv('yTest_file', sep='\t')


# now read the save and unique data
X_train = pd.Series.from_csv(path= './XTrain_file' ,sep='\t')
X_test = pd.Series.from_csv(path= './XTest_file' ,sep='\t')
y_train = pd.Series.from_csv(path= './yTrain_file' ,sep='\t')
y_test = pd.Series.from_csv(path= './yTest_file' ,sep='\t')
# In[16]:


################################################ MENU
      
def print_menu():      
	print(30 * "-" + "MENU" + 30 * "-")
	print("1. TFIDF + Logistic Regression")
	print("2. TFIDF + SVM")
	print("3. Word2Vec + Logistic Regression")
	print("4. Word2Vec + SVM")
	print("5. (TFIDF + W2V) + Logistic Regression (hard Voting)")
	print("6. (TFIDF + W2V) + Logistic Regression (soft Voting)")
	print("7. (TFIDF + W2V) + SVM (hard Voting)")
	print("8. Logistic Regression (New Strategy)")
	print("9. SVM (New Strategy)")
	print("10. (TFIDF, LogReg) + (W2V, SVM)")
	print("11. (TFIDF, SVM) + (W2V, LogReg)")
	print("0. Exit")
	print(67*"-")

#################################################################
loop=True 
while loop:
	print_menu()    ## Displays menu
	choice = input("Enter your choice [1-11]: ")
	choice = int(choice)
############################################## TF-IDF

###############################################" TFIDF +  Logistic Regression

	if choice==1:

		print("TFIDF + Logistic Regression")
		clf = Pipeline([('tfidf', TfidfVectorizer()),
                ('logreg', LogisticRegression())])


		param_grid = { 'logreg__C' : [1],
               'logreg__penalty' : ['l2'],
               'logreg__class_weight' : [{0:1, 1:2}],
               'tfidf__stop_words' : ['english'],
               'tfidf__max_df': [0.5],
               'tfidf__min_df': [ 5],
               'tfidf__tokenizer' : [tokenizer_Tw],
               'tfidf__ngram_range': [(1, 3)]
             }



		grid = GridSearchCV(clf, cv=10, param_grid=param_grid, scoring = 'f1')


		grid.fit(X_train,y_train)


		print(grid.best_score_)
		print(grid.best_params_)

		y_pred=grid.predict(X_test)

		print(classification_report(y_test, y_pred))

		print("TFIDF + Logistic Regression scores \n")
		print( 'f1_score     ',f1_score(y_test, y_pred ))
		print("roc_score   " , roc_auc_score(y_test, y_pred))
		print("recall  " ,recall_score(y_test, y_pred) )
		print("precision  " , precision_score(y_test, y_pred))
		print("accuracy  " , accuracy_score(y_test, y_pred))

		out = open("TfidfLR.txt", "w")
		out.writelines(str(grid.best_score_)+"\n"+str(f1_score(y_test, y_pred ))+"\n"+str(roc_auc_score(y_test, y_pred))+"\n"+str(recall_score(y_test, y_pred))+"\n"+str(precision_score(y_test, y_pred))+
"\n"+str(accuracy_score(y_test, y_pred)))
		out.close()

		y_pred = grid.predict_proba(tweets2017filtred)
		y_pred[:,1].tofile("tfLRproba", sep='\n')
		


####################################################" TFIDF + SVM

	elif choice==2:

		print("TFIDF + SVM")

		clf_svm = Pipeline([('tfidf', TfidfVectorizer()),
                ('svm', SVC(probability = True))])

		param_grid = { 'svm__gamma' : [ 0.1],
               'svm__kernel' : ['linear'],
               'svm__C' : [0.5],
               'svm__class_weight' : [{0:1, 1:1.7}],
               'tfidf__stop_words' : ['english'],
               'tfidf__max_df': [0.7],
               'tfidf__min_df': [2],
               'tfidf__tokenizer' : [tokenizer_Tw],
               'tfidf__analyzer' : ['word'],
               'tfidf__ngram_range': [(1, 3)]
             }


		grid = GridSearchCV(clf_svm, cv=10, param_grid=param_grid, scoring = 'f1')
		grid.fit(X_train,y_train)
		print(grid.best_score_)
		print(grid.best_params_)

		grid.cv_results_

		y_pred = grid.predict(X_test)

		print(classification_report(y_test, y_pred))

		print("TFIDF + SVM scores \n")
		print( 'f1_score     ',f1_score(y_test, y_pred) )
		print("roc_score   " , roc_auc_score(y_test, y_pred))
		print("recall  " ,recall_score(y_test, y_pred) )
		print("precision  " , precision_score(y_test, y_pred))
		print("accuracy  " , accuracy_score(y_test, y_pred))

		out = open("TfidfSVM.txt", "w")
		out.writelines(str(grid.best_score_)+"\n"+str(f1_score(y_test, y_pred ))+"\n"+str(roc_auc_score(y_test, y_pred))+"\n"+str(recall_score(y_test, y_pred))+"\n"+str(precision_score(y_test, y_pred))+"\n"+str(accuracy_score(y_test, y_pred)))
		out.close()


		y_pred = grid.predict_proba(tweets2017filtred)
		y_pred[:,1].tofile("tfSVMproba", sep='\n')

###############################"" Word2VEC

######################## ######################### W2V + Logistic Regression

	elif choice==3:
		
		print(" Word2Vec + Logistic Regression ")
		
		filename = "/home/ama/chenbeha/PFE/glove.twitter.27B.200d.txt"
		model = KeyedVectors.load_word2vec_format(filename, binary=False)
		print("taille de Vocab" ,len(model.vocab))
		print("dimension ",len(model.word_vec(word='user')))



		w2v_logreg = Pipeline([('w2v', word2vec_transformer(model, 200)),
                	('logreg', LogisticRegression())])



		param_grid = { 'logreg__C' : [10],
               'logreg__penalty' : ['l2'],
               'logreg__class_weight' : [{0:1, 1:2}]
             		}

		grid = GridSearchCV(w2v_logreg, cv=10, param_grid=param_grid, scoring = 'f1')

		grid.fit(X_train,y_train)

		print(grid.best_score_)
		print(grid.best_params_)

		y_pred=grid.predict(X_test)


		print(classification_report(y_test, y_pred))

		print(" Word2Vec + Logistic Regression Scores \n")
		print( 'f1_score     ',f1_score(y_test, y_pred ))
		print("roc_score   " , roc_auc_score(y_test, y_pred))
		print("recall  " ,recall_score(y_test, y_pred) )
		print("precision  " , precision_score(y_test, y_pred))
		print("accuracy  " , accuracy_score(y_test, y_pred))

		out = open("W2VLR.txt", "w")
		out.writelines(str(grid.best_score_)+"\n"+str(f1_score(y_test, y_pred ))+"\n"+str(roc_auc_score(y_test, y_pred))+"\n"+str(recall_score(y_test, y_pred))+"\n"+str(precision_score(y_test, y_pred))+"\n"+str(accuracy_score(y_test, y_pred)))
		out.close()

		

		y_pred = grid.predict_proba(tweets2017filtred)
		y_pred[:,1].tofile("W2VLRproba", sep='\n')

################################################### W2V + SVM
	
	elif choice==4:

		print(" Word2Vec + SVM ")
		
		filename = "/home/ama/chenbeha/PFE/glove.twitter.27B.200d.txt"
		model = KeyedVectors.load_word2vec_format(filename, binary=False)
		print("taille de Vocab" ,len(model.vocab))
		print("dimension ",len(model.word_vec(word='user')))


		w2v_svm = Pipeline([('w2v', word2vec_transformer(model, 200)),
                ('svm', SVC(class_weight ={0:1, 1:1.7}, probability = True))])


		param_grid = { 'svm__C' : [0.1],
   			 'svm__kernel' : ['linear']}


		grid = GridSearchCV(w2v_svm, cv=10, param_grid=param_grid, scoring = 'f1')

		grid.fit(X_train,y_train)

		print("best score for Word2Vec + SVM    ", grid.best_score_)
		print("best params for Word2Vec + SVM    ", grid.best_params_)

		y_pred=grid.predict(X_test)

		print(classification_report(y_test, y_pred))

		print("Word2Vec + SVM scores \n")
		print( 'f1_score     ',f1_score(y_test, y_pred ))
		print("roc_score   " , roc_auc_score(y_test, y_pred))
		print("recall  " ,recall_score(y_test, y_pred) )
		print("precision  " , precision_score(y_test, y_pred))
		print("accuracy  " , accuracy_score(y_test, y_pred))

		out = open("W2VSVM.txt", "w")
		out.writelines(str(grid.best_score_)+"\n"+str(f1_score(y_test, y_pred ))+"\n"+str(roc_auc_score(y_test, y_pred))+"\n"+str(recall_score(y_test, y_pred))+"\n"+str(precision_score(y_test, y_pred))+"\n"+str(accuracy_score(y_test, y_pred)))
		out.close()


		y_pred = grid.predict_proba(tweets2017filtred)
		y_pred[:,1].tofile("W2VSVMproba", sep='\n')


####################################### (TFIDF + W2V) Logistic Regression (hard)


	elif choice==5:

		print(" (TFIDF + W2V) Logistic Regression (hard) ")
		
		filename = "/home/ama/chenbeha/PFE/glove.twitter.27B.200d.txt"
		model = KeyedVectors.load_word2vec_format(filename, binary=False)
		print("taille de Vocab" ,len(model.vocab))
		print("dimension ",len(model.word_vec(word='user')))
		
		clf = Pipeline([('tfidf', TfidfVectorizer()),
 	               ('logreg', LogisticRegression())])
		w2v_logreg = Pipeline([('w2v', word2vec_transformer(model, 200)),
                	('logreg', LogisticRegression())])

		voting_clf = VotingClassifier(estimators= [('tfidf', clf), ('w2v', w2v_logreg)] ,voting='hard')
		
		param_grid = {  'tfidf__logreg__C' : [1],
               			'tfidf__logreg__penalty' : ['l2'],
               			'tfidf__logreg__class_weight' : [{0:1, 1:2}],
               			'tfidf__tfidf__stop_words' : ['english'],
               			'tfidf__tfidf__max_df': [0.5],
               			'tfidf__tfidf__min_df': [ 5],
               			'tfidf__tfidf__tokenizer' : [tokenizer_Tw],
               			'tfidf__tfidf__ngram_range': [(1, 3)],
				'w2v__logreg__C' : [10],
               			'w2v__logreg__penalty' : ['l2'],
               			'w2v__logreg__class_weight' : [{0:1, 1:2}]
             			}	
		
		grid = GridSearchCV(voting_clf, cv=10,param_grid=param_grid, scoring ='f1')

		grid.fit(X_train, y_train)


		y_pred = grid.predict(X_test)

		print(classification_report(y_test, y_pred))

		print(" (TFIDF + W2V) Logistic Regression (hard) scores \n ")
		print( 'f1_score     ',f1_score(y_test, y_pred ))
		print("roc_score   " , roc_auc_score(y_test, y_pred))
		print("recall  " ,recall_score(y_test, y_pred) )
		print("precision  " , precision_score(y_test, y_pred))
		print("accuracy  " , accuracy_score(y_test, y_pred))

		out = open("LR(hard).txt", "w")
		out.writelines(str(grid.best_score_)+"\n"+str(f1_score(y_test, y_pred ))+"\n"+str(roc_auc_score(y_test, y_pred))+"\n"+str(recall_score(y_test, y_pred))+"\n"+str(precision_score(y_test, y_pred))+"\n"+str(accuracy_score(y_test, y_pred)))
		out.close()


		y_pred = grid.predict_proba(tweets2017filtred)
		y_pred[:,1].tofile("hardLRproba", sep='\n')

####################################### (TFIDF + W2V) Logistic Regression (soft)

	elif choice==6:

		print(" (TFIDF + W2V) Logistic Regression (soft) ")
		
		filename = "/home/ama/chenbeha/PFE/glove.twitter.27B.200d.txt"
		model = KeyedVectors.load_word2vec_format(filename, binary=False)
		print("taille de Vocab" ,len(model.vocab))
		print("dimension ",len(model.word_vec(word='user')))
		
		clf = Pipeline([('tfidf', TfidfVectorizer()),
 	               ('logreg', LogisticRegression())])
		w2v_logreg = Pipeline([('w2v', word2vec_transformer(model, 200)),
                	('logreg', LogisticRegression())])

		voting_clf = VotingClassifier(estimators= [('tfidf', clf), ('w2v', w2v_logreg)] ,voting='soft')
		
		param_grid = {  'tfidf__logreg__C' : [1],
               			'tfidf__logreg__penalty' : ['l2'],
               			'tfidf__logreg__class_weight' : [{0:1, 1:2}],
               			'tfidf__tfidf__stop_words' : ['english'],
               			'tfidf__tfidf__max_df': [0.5],
               			'tfidf__tfidf__min_df': [ 5],
               			'tfidf__tfidf__tokenizer' : [tokenizer_Tw],
               			'tfidf__tfidf__ngram_range': [(1, 3)],
				'w2v__logreg__C' : [10],
               			'w2v__logreg__penalty' : ['l2'],
               			'w2v__logreg__class_weight' : [{0:1, 1:2}]
             			}	
		
		grid = GridSearchCV(voting_clf, cv=10,param_grid=param_grid, scoring ='f1')

		grid.fit(X_train, y_train)


		y_pred = grid.predict(X_test)

		print(classification_report(y_test, y_pred))

		print(" (TFIDF + W2V) Logistic Regression (soft) scores \n ")
		print("f1_score     ",f1_score(y_test, y_pred ))
		print("roc_score   " , roc_auc_score(y_test, y_pred))
		print("recall  " ,recall_score(y_test, y_pred) )
		print("precision  " , precision_score(y_test, y_pred))
		print("accuracy  " , accuracy_score(y_test, y_pred))

		out = open("LRSoft.txt", "w")
		out.writelines(str(grid.best_score_)+"\n"+str(f1_score(y_test, y_pred ))+"\n"+str(roc_auc_score(y_test, y_pred))+"\n"+str(recall_score(y_test, y_pred))+"\n"+str(precision_score(y_test, y_pred))+"\n"+str(accuracy_score(y_test, y_pred)))
		out.close()


		y_pred = grid.predict_proba(tweets2017filtred)
		y_pred[:,1].tofile("softLRproba", sep='\n')


####################################### (TFIDF + W2V) SVM (hard)

	elif choice==7:
		
		print(" (TFIDF + W2V) SVM (hard) ")
		
		filename = "/home/ama/chenbeha/PFE/glove.twitter.27B.200d.txt"
		model = KeyedVectors.load_word2vec_format(filename, binary=False)
		print("taille de Vocab" ,len(model.vocab))
		print("dimension ",len(model.word_vec(word='user')))


		w2v_svm = Pipeline([('w2v', word2vec_transformer(model, 200)),
                	('svm', SVC(class_weight ={0:1, 1:1.7}, probability = True))])
		clf_svm = Pipeline([('tfidf', TfidfVectorizer()),
                	('svm', SVC(class_weight ={0:1, 1:1.7}, probability = True))])


		voting_clf =  VotingClassifier(estimators=[('tfidf', clf_svm), ('w2v', w2v_svm)], voting='hard')

		param_grid = { 'w2v__svm__C' : [0.1],
   			       'w2v__svm__kernel' : ['linear'],
			       'tfidf__svm__gamma' : [ 0.1],
                               'tfidf__svm__kernel' : ['linear'],
                               'tfidf__svm__C' : [0.5],
                               'tfidf__svm__class_weight' : [{0:1, 1:1.7}],
                               'tfidf__tfidf__stop_words' : ['english'],
                               'tfidf__tfidf__max_df': [0.7],
                               'tfidf__tfidf__min_df': [2],
                               'tfidf__tfidf__tokenizer' : [tokenizer_Tw],
                               'tfidf__tfidf__analyzer' : ['word'],
                               'tfidf__tfidf__ngram_range': [(1, 3)]
             			}	
		
		grid = GridSearchCV(voting_clf, cv=10,param_grid=param_grid, scoring ='f1')

		grid.fit(X_train, y_train)

		y_pred = grid.predict(X_test)

		print(classification_report(y_test, y_pred))

		print(" (TFIDF + W2V) SVM (hard) scores \n ")
		print( 'f1_score     ',f1_score(y_test, y_pred ))
		print("roc_score   " , roc_auc_score(y_test, y_pred))
		print("recall  " ,recall_score(y_test, y_pred) )
		print("precision  " , precision_score(y_test, y_pred))
		print("accuracy  " , accuracy_score(y_test, y_pred))

		out = open("SVMhard.txt", "w")
		out.writelines(str(grid.best_score_)+"\n"+str(f1_score(y_test, y_pred ))+"\n"+str(roc_auc_score(y_test, y_pred))+"\n"+str(recall_score(y_test, y_pred))+"\n"+str(precision_score(y_test, y_pred))+"\n"+str(accuracy_score(y_test, y_pred)))
		out.close()


		y_pred = grid.predict_proba(tweets2017filtred)
		y_pred[:,1].tofile("hardSVMproba", sep='\n')



####################################### Logistic Regression (New strategy)

	elif choice==8:
		
		print("Logistic Regression (New strategy) ")

		filename = "/home/ama/chenbeha/PFE/glove.twitter.27B.200d.txt"
		model = KeyedVectors.load_word2vec_format(filename, binary=False)
		print("taille de Vocab" ,len(model.vocab))
		print("dimension ",len(model.word_vec(word='user')))

		clf = Pipeline([('tfidf', TfidfVectorizer(min_df = 2, max_df = 0.7, stop_words = 'english', tokenizer = tokenizer_Tw, ngram_range = (1,3))),
 	               ('logreg', LogisticRegression(class_weight ={0:1, 1:2}, C = 1, penalty = 'l2'))])
		
		w2v_logreg = Pipeline([('w2v', word2vec_transformer(model, 200)),
                	('logreg', LogisticRegression(penalty = 'l2', class_weight = {0:1, 1:2}, C = 10))])

		kf = KFold(n_splits=10)
		avg = 0.
		for train_index, valid_index in kf.split(X_train):
			score_kf =0.
			#print("TRAIN:", train_index, "Valid:", valid_index)
			X_train_kf, X_valid = X_train.iloc[train_index], X_train.iloc[valid_index]
			y_train_kf, y_valid = y_train.iloc[train_index], y_train.iloc[valid_index]
			my_own_classifier(clf, w2v_logreg, y_input=y_valid).fit(X_train_kf, y_train_kf)
			y_pred = my_own_classifier(clf, w2v_logreg, y_input=y_valid).predict(X_valid)
			print(y_pred)
			score_kf = f1_score(y_pred, y_valid)
			print("score", score_kf)
			avg = avg + score_kf
		print("final socre is    -----", avg/ kf.get_n_splits(X_train))


		my_own_classifier(clf, w2v_logreg, y_input=y_train).fit(X_train, y_train)

		y_pred =my_own_classifier(clf, w2v_logreg, y_input=y_test).predict(X_test)    #y-test if X-test and y-train if X_trainy_pre

		print(classification_report(y_test, y_pred))

		print("Logistic Regression (New strategy) scores \n")
		print( 'f1_score     ',f1_score(y_test, y_pred ))
		print("roc_score   " , roc_auc_score(y_test, y_pred))
		print("recall  " ,recall_score(y_test, y_pred) )
		print("precision  " , precision_score(y_test, y_pred))
		print("accuracy  " , accuracy_score(y_test, y_pred))

		out = open("LRNew.txt", "w")
		out.writelines( str(avg/ kf.get_n_splits(X_train)) + "\n"+str(f1_score(y_test, y_pred ))+"\n"+str(roc_auc_score(y_test, y_pred))+"\n"+str(recall_score(y_test, y_pred))+"\n"+str(precision_score(y_test, y_pred))+"\n"+str(accuracy_score(y_test, y_pred)))
		out.close()




		y_pred = my_own_classifier(clf, w2v_logreg, y_input=y_test).predict_proba(tweets2017filtred)
		y_pred[:,1].tofile("newLRproba", sep='\n')

####################################### SVM (New strategy)
	elif choice==9:
		
		print("SVM (New strategy) ")
		
		filename = "/home/ama/chenbeha/PFE/glove.twitter.27B.200d.txt"
		model = KeyedVectors.load_word2vec_format(filename, binary=False)
		print("taille de Vocab" ,len(model.vocab))
		print("dimension ",len(model.word_vec(word='user')))

		w2v_svm = Pipeline([('w2v', word2vec_transformer(model, 200)),
                	('svm', SVC(class_weight ={0:1, 1:1.7}, C = 0.1, probability = True))])
		clf_svm = Pipeline([('tfidf', TfidfVectorizer(min_df = 2, max_df = 0.7, stop_words = 'english', tokenizer = tokenizer_Tw, ngram_range = (1,3))),
                	('svm', SVC( C =0.5, class_weight ={0:1, 1:1.7}, probability = True))])


		kf = KFold(n_splits=10)
		avg = 0.
		for train_index, valid_index in kf.split(X_train):
			score_kf =0.
    			#print("TRAIN:", train_index, "Valid:", valid_index)
			X_train_kf, X_valid = X_train.iloc[train_index], X_train.iloc[valid_index]
			y_train_kf, y_valid = y_train.iloc[train_index], y_train.iloc[valid_index]
			my_own_classifier_svm(clf_svm, w2v_svm, y_input=y_valid).fit(X_train_kf, y_train_kf)
			y_pred = my_own_classifier_svm(clf_svm, w2v_svm, y_input=y_valid).predict(X_valid)
			print(y_pred)
			score_kf = f1_score(y_pred, y_valid)
			print("score", score_kf)
			avg = avg + score_kf
		print("final socre is    -----", avg/ kf.get_n_splits(X_train))


		my_own_classifier_svm(clf_svm, w2v_svm, y_input=y_train).fit(X_train, y_train)

		y_pred =my_own_classifier_svm(clf_svm, w2v_svm, y_input=y_test).predict(X_test)    #y-test if X-test and y-train if X_trainy_pre

		print(classification_report(y_test, y_pred))

		print("SVM (New strategy) scores \n ")
		print( 'f1_score     ',f1_score(y_test, y_pred ))
		print("roc_score   " , roc_auc_score(y_test, y_pred))
		print("recall  " ,recall_score(y_test, y_pred) )
		print("precision  " , precision_score(y_test, y_pred))
		print("accuracy  " , accuracy_score(y_test, y_pred))

		out = open("SVMNew.txt", "w")
		out.writelines(str(avg/ kf.get_n_splits(X_train)) + "\n"+str(f1_score(y_test, y_pred ))+"\n"+str(roc_auc_score(y_test, y_pred))+"\n"+str(recall_score(y_test, y_pred))+"\n"+str(precision_score(y_test, y_pred))+"\n"+str(accuracy_score(y_test, y_pred)))
		out.close()




		y_pred = my_own_classifier_svm(clf_svm, w2v_svm, y_input=y_test).predict_proba(tweets2017filtred)
		y_pred[:,1].tofile("newSVMproba", sep='\n')
######################################## (TFIDF + LR) + (W2V + SVM)

######################################## (TFIDF + SVM) + (W2V + LR)

	elif choice== 0:
		print("exit")
		loop=False # This will make the while loop to end as not value of loop is set
	else:
		# Any integer inputs other than values 1-5 we print an error message
		raw_input("Wrong option selection. Enter any key to try again..")



# ### SVM
