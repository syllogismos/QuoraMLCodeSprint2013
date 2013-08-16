"""
{u'__ans__': 23.33333,
 u'anonymous': False,
 u'context_topic': {u'followers': 7827, u'name': u'reddit'},
 u'question_key': u'AAEAACblijiZiOK3g4dUIQO8fr20tWJorKKpNh6WnEUUH1/K',
 u'question_text': u'What happened to cause reddit to go down for emergency maintenance at 9:40 pm, 10/01/12?',
 u'topics': [{u'followers': 7827, u'name': u'reddit'}]}
"""

import json

import numpy as np

from sklearn.linear_model import LinearRegression
from sklearn.linear_model import PassiveAggressiveRegressor
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn import metrics
from sklearn import preprocessing

import warnings
warnings.filterwarnings("ignore")

f = open("interest/sample/input00.in",'r')

n = int(f.next().strip())

y = []
anonymous = []
context = []
contextfollowers = []
topics = []
topicsfollowers = []
question = []
notopics = []

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
q = int(f.next().strip())
tanonymous = []
tcontext = []
tcontextfollowers = []
ttopics = []
ttopicsfollowers = []
tquestion = []
tnotopics = []
tquestion_key = []
print "loading test data"
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
    tquestion_key.append(temp['question_key'])
f.close()
print "loading output data from file to check the result"
g = open("interest/sample/output00.out",'r')
realy = []
for i in range(q):
    temp = json.loads(g.next().strip())
    realy.append(temp["__ans__"])
g.close()



print "extracting features"
quevectorizer = TfidfVectorizer(sublinear_tf=True, max_df=0.5,stop_words='english')  
topvectorizer = TfidfVectorizer(sublinear_tf=True, max_df=0.5,stop_words='english')
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

"""
ARDRegression
BayesianRidge
ElasticNet
ElasticNetCV
Hinge
Huber
Lars
LarsCV
Lasso
LassoCV
LassoLars
LassoLarsCV
LassoLarsIC
PassiveAggressiveRegressor
Ridge
SGDRegressor
LinearRegression
ModifiedHuber
MultiTaskElasticNet
"""

print "training using PassiveAggressiveRegressor"
par = PassiveAggressiveRegressor()
par.fit(quesparse,y)
pred = par.predict(tquesparse)
pred[pred<0] = 0
#for i in range(q):
#    temp = dict()
#    temp['__ans__'] = pred[i]
#    temp['question_key'] = tquestion_key[i]
#    print """{"__ans__": %s, "question_key":"%s"}""" % (temp['__ans__'], temp["question_key"])