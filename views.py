
"""
{u'__ans__': 2.089473684,
 u'anonymous': False,
 u'context_topic': {u'followers': 500022, u'name': u'Movies'},
 u'num_answers': 4,
 u'promoted_to': 0,
 u'question_key': u'AAEAAM9EY6LIJsEFvYiwKLfCe7d+hkbsXJ5qM7aSwTqemERp',
 u'question_text': u'What are some movies with mind boggling twists in it?',
 u'topics': [{u'followers': 500022, u'name': u'Movies'}]}
 """

import json

import numpy as np

from sklearn.linear_model import LinearRegression
from sklearn.linear_model import PassiveAggressiveRegressor
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn import metrics
from sklearn import preprocessing

import pylab as pl

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

def meanadj(arr):
    return (sum(arr)+100)/(len(arr)+1)
    
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

print "extracting features"
#quevectorizer = TfidfVectorizer(sublinear_tf=True, max_df=0.5,stop_words='english')  
#topvectorizer = TfidfVectorizer(sublinear_tf=True, max_df=0.5,stop_words='english')
quevectorizer = HashingVectorizer(stop_words='english',non_negative=True, n_features=10000)
topvectorizer = HashingVectorizer(stop_words='english',non_negative=True, n_features=10000)


cfscaler = preprocessing.StandardScaler().fit(contextfollowers)
tfscaler = preprocessing.StandardScaler().fit(topicsfollowers)


quesparse = quevectorizer.fit_transform(question)
topsparse = topvectorizer.fit_transform(topics)
cfscaled = cfscaler.transform(contextfollowers)
tfscaled = tfscaler.transform(topicsfollowers)

tquesparse = quevectorizer.transform(tquestion)
ttopsparse = topvectorizer.transform(ttopics)
tcfscaled = cfscaler.transform(tcontextfollowers)
ttfscaled = tfscaler.transform(ttopicsfollowers)



par = PassiveAggressiveRegressor()
par.fit(topsparse,y)
pred = par.predict(ttopsparse)
pred[pred<0] = 0


temp = pl.figure("train y")
temp = pl.subplot(2,1,1)
temp = pl.hist(y,1000)
temp = pl.subplot(2,1,2)
yy = y.copy()
yy[yy==0] = 1
temp = pl.hist(np.log10(yy),1000)

temp = pl.figure("test y")
temp = pl.subplot(4,1,1)
temp = pl.hist(pred,1000)
temp = pl.subplot(4,1,2)
yy = pred.copy()
yy[yy==0] = 1
temp = pl.hist(np.log10(yy),1000)
temp = pl.subplot(4,1,3)
temp = pl.hist(realy,1000)
temp = pl.subplot(4,1,4)
yy = realy.copy()
yy[yy==0] = 1
temp = pl.hist(np.log10(yy),1000)


pl.axis([0,200,0,1000])
pl.text(100,700,"train y")
pl.xlabel()
pl.ylabel()







#y = np.array(y)
#y[y==0] = 1
#ly = np.log10(y)

#par = PassiveAggressiveRegressor()
#par.fit(quesparse,ly)
#lpred = par.predict(tquesparse)
#pred = np.power(lpred,np.repeat(10,q))
#
#par1 = PassiveAggressiveRegressor()
#par1.fit(quesparse)


#topdense = topsparse.toarray()
#ttopdense = ttopsparse.toarray()
#par = PassiveAggressiveRegressor()
#par.fit(topdense,y)
#pred = par.predict(ttopdense)

#dims = [10000,15000,20000,30000,40000]
#for dim in dims:
#    print dim
#    vec =  HashingVectorizer(stop_words='english',non_negative=True,n_features=dim)
#    top1 = vec.transform(topics)
#    ttop1 = vec.transform(ttopics)
#    par = PassiveAggressiveRegressor()
#    par.fit(top1,y)
#    pred = par.predict(ttop1)
#    print dim
#    pred[pred<0] = 0
#    print rmsle(pred,realy)



# isomorphic regression by dividing the dataset into two parts and then doing
# linear regression seperately on the two parts seperately..
#y = np.array(y)
#less = y<50
#lessind = less.nonzero()[0]
#moreind = (less==False).nonzero()[0]
#lesstopsparse = topsparse[lessind,:]
#moretopsparse = topsparse[moreind,:]
#classlearner = PassiveAggressiveClassifier()
#lessregressor = PassiveAggressiveRegressor()
#moreregressor = PassiveAggressiveRegressor()
#classlearner.fit(topsparse,less)
#lessregressor.fit(lesstopsparse,y[lessind])
#moreregressor.fit(moretopsparse,y[moreind])
#testclass = classlearner.predict(ttopsparse)
#tlessind = testclass.nonzero()[0]
#tmoreind = (testclass==False).nonzero()[0]
#lesspred = lessregressor.predict(ttopsparse[tlessind,:])
#morepred = moreregressor.predict(ttopsparse[tmoreind,:])
#pred = np.zeros(q)
#pred[tlessind] = lesspred
#pred[tmoreind] = morepred
#pred[pred<0] = 0




#for i in range(q):
#    temp = dict()
#    temp['__ans__'] = pred[i]
#    temp['question_key'] = tquestionkey[i]
#    print """{"__ans__": %s, "question_key":"%s"}""" % (temp['__ans__'], temp["question_key"])