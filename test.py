import requests
from sklearn.linear_model import LogisticRegression
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import GradientBoostingClassifier
l = requests.get('http://api.huobi.com/staticmarket/ltc_kline_001_json.js').json()
b = requests.get('http://api.huobi.com/staticmarket/btc_kline_001_json.js').json()

data=[]

for i in range(len(l)):
    if i >= 4 and i < (len(l) - 1):
        row=[
          1 if l[i][4]<l[i+1][4] else 0,
          l[i-4][4], l[i-3][4], l[i-2][4], l[i-1][4], l[i][4],
          l[i-4][5], l[i-3][5], l[i-2][5], l[i-1][5], l[i][5],
          b[i-4][4], b[i-3][4], b[i-2][4], b[i-1][4], b[i][4],
          b[i-4][5], b[i-3][5], b[i-2][5], b[i-1][5], b[i][5]
          ]
        data.append(row)

y=[item[0] for item in data]
x=[item[1:] for item in data]

train_size = 200
md = LogisticRegression()
md = md.fit(x[:train_size],y[:train_size])
p = md.predict(x[train_size:])

dt = tree.DecisionTreeClassifier()
dt = dt.fit(x[:train_size],y[:train_size])
pt = dt.predict(x[train_size:])

rf = RandomForestClassifier(n_estimators=10)
rf = rf.fit(x[:train_size],y[:train_size])
pr = rf.predict(x[train_size:])

sv = svm.SVC(kernel='sigmoid')
sv.fit(x[:train_size],y[:train_size])
ps = sv.predict(x[train_size:])

gnb = GaussianNB()
gnb = gnb.fit(x[:train_size],y[:train_size])
pg = gnb.predict(x[train_size:])

from sklearn.neighbors import KNeighborsClassifier
nbrs = KNeighborsClassifier()
nbrs = nbrs.fit(x[:train_size],y[:train_size])
pn = nbrs.predict(x[train_size:])

#gbrt =  GradientBoostingRegressor(n_estimators=100, max_depth=10)
#gbrt = gbrt.fit(x[:train_size],y[:train_size])
#pgt = gbrt.predict(x[train_size:])

gbdt =  GradientBoostingClassifier(n_estimators=100, max_depth=10)
gbdt = gbdt.fit(x[:train_size],y[:train_size])
pgt = gbdt.predict(x[train_size:])


def get_precision(test, predict):
    test_score = {}
    for i in range(0, len(predict)):
            t = test[i]
            p = predict[i]
            if t not in test_score:
                test_score.update({t:{'match':0, 'recall':0, 'precision':0}})
            if p not in test_score:
                test_score.update({p:{'match':0, 'recall':0, 'precision':0}})
            test_score[t]['recall'] += 1
            test_score[p]['precision'] += 1
            if t == p:
                test_score[t]['match'] += 1
    ret = {}
    for i in test_score:
        row = test_score[i]
        ret.update({i:{'pre':row['precision'] ,'match': row['match'], 'recall': row['match'] / max((row['recall'] + 0.0),1), 'precision' : row['match'] / max((row['precision'] + 0.0),1)}})
    return ret
print len(y[train_size:]), sum(y[train_size:])
print y[train_size:]
print 'LR:'
print list(p)
print get_precision(y[train_size:], p)
print 'DT:'
print list(pt)
print get_precision(y[train_size:], pt)
print 'RF:'
print list(pr)
print get_precision(y[train_size:], pr)
print 'SVM:'
print list(ps)
print get_precision(y[train_size:], ps)
print 'GaussianNB:'
print list(pg)
print get_precision(y[train_size:], pg)
print 'GBDT:'
print list(pgt)
print get_precision(y[train_size:], pgt)
print 'KN:'
print list(pn)
print get_precision(y[train_size:], pn)

