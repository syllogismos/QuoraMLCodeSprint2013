import json

import numpy as np

from sklearn.linear_model import LinearRegression
from sklearn.linear_model import PassiveAggressiveRegressor
from sklearn.linear_model import ARDRegression
from sklearn.linear_model import BayesianRidge
from sklearn.linear_model import ElasticNet
from sklearn.linear_model import ElasticNetCV
from sklearn.linear_model import Lars
from sklearn.linear_model import Lasso
from sklearn.linear_model import LassoLars
from sklearn.linear_model import MultiTaskLasso
from sklearn.linear_model import MultiTaskElasticNet
from sklearn.linear_model import Perceptron
from sklearn.linear_model import RandomizedLasso
from sklearn.linear_model import SGDRegressor
from sklearn.linear_model import Ridge

from sklearn.feature_selection import SelectKBest, chi2

from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn import metrics
from sklearn import preprocessing

import pylab as pl
from time import time

import warnings
warnings.filterwarnings("ignore")

y = []
anonymous = []
context = []
contextfollowers = []
numanswers = []
promotedto = []
questionkey = []
question = []
topics = []
topicsfollowers = []
notopics = []

tanonymous = []
tcontext = []
tcontextfollowers = []
tnumanswers = []
tpromotedto = []
tquestionkey = []
tquestion = []
ttopics = []
ttopicsfollowers = []
tnotopics = []


def meanadj1(arr):
    return (sum(arr)+100)/(len(arr)+1)
    
def meanadj(arr):
    return (sum(arr)+100)
    
def rmsle(y1,y2):
    y1 = np.array(y1)
    y2 = np.array(y2)
    y1[y1<0] = 0
    y2[y2<0] = 0
    return np.sqrt(np.mean((np.log10(y1+1)-np.log10(y2+1))**2))
def score(y1,y2):
    return 50/rmsle(y1,y2)

f = open("views\sample\input00.in",'r')
n = int(f.next().strip())
print "loading train data"
for i in range(n):
    temp = json.loads(f.next().strip())
    y.append(temp['__ans__'])
    question.append(temp['question_text'])
    if(temp['context_topic'] != None):
        context.append(temp['context_topic']['name'])
        contextfollowers.append(temp['context_topic']['followers'])
    elif(len(temp['topics'])!=0):
        context.append(temp['topics'][0]['name'])
        contextfollowers.append(temp['topics'][0]['followers'])
    else:
        context.append(u'random')
        contextfollowers.append(400)
    t = [xx['name'] for xx in temp['topics']]
    topics.append(reduce(lambda x,y: x+" "+y, t,""))
    t = [xx['followers'] for xx in temp['topics']]
    topicsfollowers.append(meanadj(t))
    notopics.append(len(t)+1)
    anonymous.append(temp['anonymous'])
    promotedto.append(temp['promoted_to'])
    numanswers.append(temp['num_answers'])
    questionkey.append(temp['question_key'])

print "loading test data"
q = int(f.next().strip())
for i in range(q):
    temp = json.loads(f.next().strip())
    tquestion.append(temp['question_text'])
    if(temp['context_topic'] != None):
        tcontext.append(temp['context_topic']['name'])
        tcontextfollowers.append(temp['context_topic']['followers'])
    elif(len(temp['topics'])!=0):
        tcontext.append(temp['topics'][0]['name'])
        tcontextfollowers.append(temp['topics'][0]['followers'])
    else:
        tcontext.append(u'random')
        tcontextfollowers.append(400)
    t = [xx['name'] for xx in temp['topics']]
    ttopics.append(reduce(lambda x,y: x+" "+y, t,""))
    t = [xx['followers'] for xx in temp['topics']]
    ttopicsfollowers.append(meanadj(t))
    tnotopics.append(len(t)+1)
    tanonymous.append(temp['anonymous'])
    tpromotedto.append(temp['promoted_to'])
    tnumanswers.append(temp['num_answers'])
    tquestionkey.append(temp['question_key'])
    
f.close()
print "loading output data from file to check the result"
g = open("views/sample/output00.out",'r')
realy = []
for i in range(q):
    temp = json.loads(g.next().strip())
    realy.append(temp["__ans__"])
g.close()

realy = np.array(realy)
y = np.array(y)

anonymous = np.array(anonymous)
promotedto = np.array(promotedto)
numanswers = np.array(numanswers)
notopics = np.array(notopics)
contextfollowers = np.array(contextfollowers)
topicsfollowers = np.array(topicsfollowers)

tanonymous = np.array(tanonymous)
tpromotedto = np.array(tpromotedto)
tnumanswers = np.array(tnumanswers)
tnotopics = np.array(tnotopics)
tcontextfollowers = np.array(tcontextfollowers)
ttopicsfollowers = np.array(ttopicsfollowers)

print "extracting features"
quevectorizerTfid = TfidfVectorizer(sublinear_tf=True, max_df=0.5,stop_words='english')  
topvectorizerTfid = TfidfVectorizer(sublinear_tf=True, max_df=0.5,stop_words='english')
quevectorizerHash = HashingVectorizer(stop_words='english',non_negative=True, n_features=1000)
topvectorizerHash = HashingVectorizer(stop_words='english',non_negative=True, n_features=1000)

#quesparseHash = quevectorizerHash.transform(question)
topsparseHash = topvectorizerHash.transform(topics)
#tquesparseHash = quevectorizerHash.transform(tquestion)
ttopsparseHash = topvectorizerHash.transform(ttopics)

cfscaler = preprocessing.StandardScaler().fit(contextfollowers)
tfscaler = preprocessing.StandardScaler().fit(topicsfollowers)

cfscaled = cfscaler.transform(contextfollowers)
tfscaled = tfscaler.transform(topicsfollowers)
tcfscaled = cfscaler.transform(tcontextfollowers)
ttfscaled = tfscaler.transform(ttopicsfollowers)

def benchmark(clf, trainx, trainy, test, dataset):
    print 80 * '_'
    print "Training..."
    print clf
    t0 = time()
    clf.fit(trainx,trainy)
    train_time = time() - t0
    print "train time: %0.3fs" % train_time
    t0 = time()
    pred = clf.predict(test)
    test_time = time() - t0
    print "test time: %0.3fs" % test_time
    
    score = rmsle(realy,pred)
    print score
    clf_descr = str(clf).split('(')[0]
    return clf_descr, score, train_time, test_time, dataset

results = []


X_train1 = topsparseHash
y_train1 = y
X_test1 = ttopsparseHash

for clf, name in (
        (Ridge(tol=1e-2, solver="lsqr"), "Ridge Regression"),
        (LinearRegression, "Linear Regression"),
        (PassiveAggressiveRegressor(n_iter=50), "Passive-Aggressive"),
        (ARDRegression(),"ARD regressoin"),
        (BayesianRidge(),"Bayesian Ridge"),
        (SGDRegressor(), "SGDRegression")):
    print 80 * '='
    print name
    results.append(benchmark(clf,X_train1,y_train1,X_test1,1))


