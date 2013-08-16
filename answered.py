#{u'__ans__': False,
# u'anonymous': False,
# u'context_topic': {u'followers': 76, u'name': u'Logistic Regression'},
# u'question_key': u'AAEAAPnh9/AlSw3IL2wm5WFRcjy/h/SlSYi4md7qqEQrzY7v',
# u'question_text': u'What is the gradient of the log likelihood function in  multinomial logistic regression?',
# u'topics': [{u'followers': 7240, u'name': u'Data Mining'},
#  {u'followers': 76, u'name': u'Logistic Regression'},
#  {u'followers': 64668, u'name': u'Machine Learning'}]}

import json

import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import RidgeClassifier

from sklearn.linear_model import Perceptron
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.naive_bayes import BernoulliNB, MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import NearestCentroid

from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn import metrics
from sklearn import preprocessing
from sklearn.svm import SVC

import warnings
warnings.filterwarnings("ignore")

inp = open("answered/sample/answered_data_10k.in",'r')

n = int(inp.next())
questions = []
context = []
contextfollowers = []
topics = []
topicsfollowers = []
questionkeys = []
anonymous = []
notopics = []
y = []

a = []
b = []
for i in range(n):
    a.append(json.loads(inp.next()))
#t = int(inp.next())
#for i in range(t):
#    b.append(json.loads(inp.next()))

def meanadj(arr):
    return (sum(arr)+100)/(len(arr)+1)
print "loading train data"
for i in range(n):
    questions.append(a[i]['question_text'])
    if(a[i]['context_topic'] != None):
        context.append(a[i]['context_topic']['name'])
        contextfollowers.append(a[i]['context_topic']['followers'])
    elif(len(a[i]['topics'])!=0):
        context.append(a[i]['topics'][0]['name'])
        contextfollowers.append(a[i]['topics'][0]['followers'])
    else:
        context.append(u'random')
        contextfollowers.append(400)
    temp = [xx['name'] for xx in a[i]['topics']]
    topics.append(reduce(lambda x,y: x+" "+y, temp,""))
    temp = [xx['followers'] for xx in a[i]['topics']]
    topicsfollowers.append(meanadj(temp))
    notopics.append(len(temp)+1)
    anonymous.append(a[i]['anonymous'])
    y.append(a[i]['__ans__'])

print "reading test data"
t = int(inp.next())
tquestions = []
tcontext = []
tcontextfollowers = []
ttopics = []
tnotopics = []
ttopicsfollowers = []
tquestionkeys = []
tanonymous = []
for i in range(t):   
    tt = json.loads(inp.next())
    tquestions.append(tt['question_text'])
    if(tt['context_topic'] != None):
        tcontext.append(tt['context_topic']['name'])
        tcontextfollowers.append(tt['context_topic']['followers'])
    elif(len(tt['topics'])!=0):
        tcontext.append(tt['topics'][0]['name'])
        tcontextfollowers.append(tt['topics'][0]['followers'])
    else:
        tcontext.append(u'random')
        tcontextfollowers.append(400)
    temp = [xx['name'] for xx in tt['topics']]
    ttopics.append(reduce(lambda x,y: x+" "+y, temp,""))
    temp = [xx['followers'] for xx in tt['topics']]
    ttopicsfollowers.append(meanadj(temp))
    tnotopics.append(len(temp)+1)
    tanonymous.append(tt['anonymous'])
    tquestionkeys.append(tt['question_key'])

tcontextfollowers = np.array(tcontextfollowers,dtype='float')
ttopicsfollowers = np.array(ttopicsfollowers,dtype='float')  
notopics = np.array(notopics,dtype='float')
tnotopics = np.array(tnotopics,dtype='float')
contextfollowers = np.array(contextfollowers,dtype='float')
topicsfollowers = np.array(topicsfollowers,dtype='float')  

quevectorizer = TfidfVectorizer(sublinear_tf=True, max_df=0.5,stop_words='english')  
topvectorizer = TfidfVectorizer(sublinear_tf=True, max_df=0.5,stop_words='english')
cfscaler = preprocessing.StandardScaler().fit(contextfollowers)
tfscaler = preprocessing.StandardScaler().fit(topicsfollowers)

quesparse = quevectorizer.fit_transform(questions)
topsparse = topvectorizer.fit_transform(topics)
scaledcf = cfscaler.transform(contextfollowers)
scaledtf = tfscaler.transform(topicsfollowers)

# classifiers

#passive aggressive classifiers.
#passiveq = PassiveAggressiveClassifier(n_iter=10)
#passivet = PassiveAggressiveClassifier(n_iter=10)

# linear svcs
#passiveq = LinearSVC(penalty="l1", dual=False,tol=1e-3)
#passivet = LinearSVC(penalty="l1", dual=False,tol=1e-3)

#perceptron
#passiveq = Perceptron(n_iter=10)
#passivet = Perceptron(n_iter=10)

#kneighbor
#passiveq = KNeighborsClassifier(n_neighbors=10)
#passivet = KNeighborsClassifier(n_neighbors=10)

#SGDClass
passiveq = SGDClassifier(alpha=0.0001,n_iter=60,penalty="l2")
passivet = SGDClassifier(alpha=0.0001,n_iter=60,penalty="l2")

# SVC
#passiveq = SVC(kernel='poly',degree=3)
#passivet = SVC(kernel='poly',degree=3)

#SGDClass
#passiveq = SGDClassifier(alpha=0.001,n_iter=50,penalty="l1")
#passivet = SGDClassifier(alpha=0.001,n_iter=50,penalty="l1")

#SGDClass
#passiveq = SGDClassifier(alpha=0.0001,n_iter=50,penalty="elasticnet")
#passivet = SGDClassifier(alpha=0.0001,n_iter=50,penalty="elasticnet")


#passivef=PassiveAggressiveClassifier(n_iter=10)
#passivef=Perceptron(n_iter=50)
#passivef=KNeighborsClassifier(n_neighbors=10)
passivef=SGDClassifier(alpha=0.0001,n_iter=50,penalty="l2")
#passivef=SGDClassifier(alpha=0.0001,n_iter=50,penalty="l1")
#passivef=SGDClassifier(alpha=0.0001,n_iter=50,penalty="elasticnet")
#passivef=LinearSVC(penalty="l1", dual=False,tol=1e-3)
#passivef = LogisticRegression()




passiveq.fit(quesparse,y)
passivet.fit(topsparse,y)

predq = passiveq.predict(quesparse)
predt = passivet.predict(topsparse)

Xtrainpassive = np.vstack((anonymous,predt,notopics,scaledcf,scaledtf))
Xtrainpassive = Xtrainpassive.transpose()


passivef.fit(Xtrainpassive,y)
predf = passivef.predict(Xtrainpassive)


print "transfroming test data"
tquesparse = quevectorizer.transform(tquestions)
ttopsparse = topvectorizer.transform(ttopics)
tscaledcf = cfscaler.transform(tcontextfollowers)
tscaledtf = tfscaler.transform(ttopicsfollowers)

#predicting final test features
tpredq = passiveq.predict(tquesparse)
tpredt = passivet.predict(ttopsparse)
Xtestpassive = np.vstack((tanonymous,tpredt,tnotopics,tscaledcf,tscaledtf))
Xtestpassive = Xtestpassive.transpose()
tpredf = passivef.predict(Xtestpassive)

tfile = open("answered/sample/answered_data_10k.out",'r')
tans = []
for i in range(1000):
    temp = json.loads(tfile.next())
    tans.append(temp['__ans__'])
print metrics.accuracy_score(tans,tpredq)
print metrics.accuracy_score(tans,tpredt)
print metrics.accuracy_score(tans,tpredf)
#for i in range(t):
#    temp = dict()
#    temp['__ans__'] = tpredf[i]
#    temp['question_key'] = tquestionkeys[i]
#    print """{"__ans__": %s, "question_key":"%s"}""" % ("true" if temp["__ans__"] else "false", temp["question_key"])