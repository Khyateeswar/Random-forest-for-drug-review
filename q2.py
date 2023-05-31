import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
from scipy import stats as st
from sklearn.feature_extraction.text import CountVectorizer
from nltk.corpus import stopwords
from scipy.sparse import hstack
from scipy import sparse
from lightgbm import LGBMClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import HalvingGridSearchCV
from xgboost import XGBClassifier
import statistics
from nltk.stem.snowball import SnowballStemmer

# train_path = "COL774_drug_review"
# test_path= "COL774_drug_review"
# val_path = "COL774_drug_review"

#train_path = str(sys.argv[1])
#test_path = str(sys.argv[2])
train_in = pd.read_csv("DrugsComTrain.csv")
test_in = pd.read_csv("DrugsComTest.csv")
val_in = pd.read_csv("DrugsComVal.csv")
stemmer = SnowballStemmer('english')

def process_train(l):
    month_num = {}
    month_num["January"]=0
    month_num["February"]=1
    month_num["March"]=2
    month_num["April"]=3
    month_num["May"]=4
    month_num["June"]=5
    month_num["July"]=6
    month_num["August"]=7
    month_num["September"]=8
    month_num["October"]=9
    month_num["November"]=10
    month_num["December"]=11
    date=[]
    for i in range(l.shape[0]):
      temp=[]
      ll = l[i,2].split()
      ll[1]=ll[1][:len(ll)-2]
      temp.append(int(ll[1]))
      temp.append(month_num[ll[0]])
      temp.append(int(ll[2]))
      date.append(temp)
    date=np.array(date)
    date = date.astype(np.float32)
    date = sparse.csr_matrix(date)

    vec_con = CountVectorizer(stop_words='english')
    x_con = vec_con.fit_transform(l[:,1].astype('U'))
    
    review = l[:,0]
    vec_rev = CountVectorizer(stop_words='english')
    corpus = []
    # count = 0
    for text in review:
        corpus.append(' '.join([stemmer.stem(word) for word in text.split()]))
        # if count%500==0:
        #     print(count)
        # 
  
    x_rev = vec_rev.fit_transform(corpus)

    return x_rev,x_con,date,vec_rev,vec_con


    
def process(l,vec_con,vec_rev):
    month_num = {}
    month_num["January"]=0
    month_num["February"]=1
    month_num["March"]=2
    month_num["April"]=3
    month_num["May"]=4
    month_num["June"]=5
    month_num["July"]=6
    month_num["August"]=7
    month_num["September"]=8
    month_num["October"]=9
    month_num["November"]=10
    month_num["December"]=11
    date=[]
    for i in range(l.shape[0]):
      temp=[]
      ll = l[i,2].split()
      ll[1]=ll[1][:len(ll)-2]
      temp.append(int(ll[1]))
      temp.append(month_num[ll[0]])
      temp.append(int(ll[2]))
      date.append(temp)
    date=np.array(date)
    date = date.astype(np.float32)
    date = sparse.csr_matrix(date)

    
    x_con = vec_con.transform(l[:,1].astype('U'))
    
    review = l[:,0]
    vec_rev = CountVectorizer(stop_words='english')
    corpus = []
    for text in review:
        corpus.append(' '.join([stemmer.stem(word) for word in text.split()]))
  
    x_rev = vec_rev.transform(corpus)

    return x_rev,x_con,date

def part_a():
    tr_d = (train_in[["review","condition","date","rating"]]).to_numpy()
    te_d = (test_in[["review","condition","date","rating"]]).to_numpy()
    va_d = (val_in[["review","condition","date","rating"]]).to_numpy()


    x_rev,x_con,x_d,vec_rev,vec_con = process_train(tr_d)
    xf = hstack([hstack([x_rev,x_con]),x_d])
    opt_clf = DecisionTreeClassifier()
    model = opt_clf.fit(xf,tr_d[:,3].astype(np.float32))
    #print(model.score(xf,tr_d[:,3].astype(np.float32)))

    x_rev,x_con,x_d = process(tr_d,vec_rev,vec_con)
    xf = hstack([hstack([x_rev,x_con]),x_d])
    train_pred = opt_clf.predict(xf)
    count= 0
    for i in range(len(train_pred)):
        if(train_pred[i]==float(tr_d[i,3])):
            count=count+1
    print(count/len(train_pred)*100)
    
    x_rev,x_con,x_d = process(te_d,vec_rev,vec_con)
    xf = hstack([hstack([x_rev,x_con]),x_d])
    train_pred = opt_clf.predict(xf)
    count= 0
    for i in range(len(train_pred)):
        if(train_pred[i]==float(te_d[i,3])):
            count=count+1
    print(count/len(train_pred)*100)
    

    x_rev,x_con,x_d = process(va_d,vec_rev,vec_con)
    xf = hstack([hstack([x_rev,x_con]),x_d])
    train_pred = opt_clf.predict(xf)
    count= 0
    for i in range(len(train_pred)):
        if(train_pred[i]==float(te_d[i,3])):
            count=count+1
    print(count/len(train_pred)*100)

    return

def part_b():
  tr_d = (train_in[["review","condition","date","rating"]]).to_numpy()
  te_d = (test_in[["review","condition","date","rating"]]).to_numpy()
  va_d = (val_in[["review","condition","date","rating"]]).to_numpy()
  x_rev,x_con,x_d,vec_rev,vec_con = process_train(tr_d)
  xf = hstack([hstack([x_rev,x_con]),x_d])


  acc = 0
  par = [1,1,1]
  for i in range(1,50):
      for j in range(2,50):
          for k in range(1,50):
              opt_clf = DecisionTreeClassifier(max_depth=i,min_samples_split=j,min_samples_leaf=k)
              model = opt_clf.fit(xf,tr_d[:,3].astype(np.float32))
              train_pred = opt_clf.predict(xf)
              count= 0
              for i in range(len(train_pred)):
                if(train_pred[i]==float(tr_d[i,3])):
                  count=count+1
              vacc=count/len(train_pred)*100
              if(vacc>acc):
                  acc=vacc
                  par[0]=i
                  par[1]=j
                  par[2]=k
  print(par)
  opt_clf = DecisionTreeClassifier(max_depth=par[0],min_samples_split=par[1],min_samples_leaf=par[2])
  model = opt_clf.fit(xf,tr_d[:,3].astype(np.float32))

  x_rev,x_con,x_d = process(tr_d,vec_rev,vec_con)
  xf = hstack([hstack([x_rev,x_con]),x_d])
  train_pred = opt_clf.predict(xf)
  count= 0
  for i in range(len(train_pred)):
      if(train_pred[i]==float(tr_d[i,3])):
          count=count+1
  print(count/len(train_pred)*100)
    
  x_rev,x_con,x_d = process(te_d,vec_rev,vec_con)
  xf = hstack([hstack([x_rev,x_con]),x_d])
  train_pred = opt_clf.predict(xf)
  count= 0
  for i in range(len(train_pred)):
      if(train_pred[i]==float(te_d[i,3])):
          count=count+1
  print(count/len(train_pred)*100)
    

  x_rev,x_con,x_d = process(va_d,vec_rev,vec_con)
  xf = hstack([hstack([x_rev,x_con]),x_d])
  train_pred = opt_clf.predict(xf)
  count= 0
  for i in range(len(train_pred)):
      if(train_pred[i]==float(te_d[i,3])):
          count=count+1
  print(count/len(train_pred)*100)

  return

def part_c():
    tr_d = (train_in[["Age","Shape","Margin","Density","Severity"]]).to_numpy()
    te_d = (test_in[["Age","Shape","Margin","Density","Severity"]]).to_numpy()
    va_d = (val_in[["Age","Shape","Margin","Density","Severity"]]).to_numpy()
    x_rev,x_con,x_d,vec_rev,vec_con = process_train(tr_d)
    xf = hstack([hstack([x_rev,x_con]),x_d])
    x_rev,x_con,x_d,vec_rev,vec_con = process(te_d)
    xf1 = hstack([hstack([x_rev,x_con]),x_d])
    x_rev,x_con,x_d,vec_rev,vec_con = process(va_d)
    xf2 = hstack([hstack([x_rev,x_con]),x_d])
    opt_clf = DecisionTreeClassifier()
    model = opt_clf.fit(xf,tr_d[:,3].astype(np.float32))
    
    alphas=opt_clf.cost_complexity_pruning_path(xf,tr_d[:,3].astype(np.float32))["ccp_alphas"]
    impur = opt_clf.cost_complexity_pruning_path(xf,tr_d[:,3].astype(np.float32))["impurities"]
    train_acc=[]
    test_acc=[]
    val_acc=[]
    nodes=[]
    depth=[]
    for i in range(len(alphas)):
        clf = DecisionTreeClassifier(ccp_alpha=alphas[i])
        clf.fit(xf,tr_d[:,3].astype(np.float32))
        train_pred = clf.predict(xf)
        count= 0
        for i in range(len(train_pred)):
            if(train_pred[i]==float(tr_d[i,3])):
                count=count+1
        train_acc.append(count/len(train_pred)*100)
        test_pred = clf.predict(xf1)
        count= 0
        for i in range(len(test_pred)):
            if(test_pred[i]==float(te_d[i,3])):
                count=count+1
        test_acc.append(count/len(test_pred)*100)
        val_pred = clf.predict(xf2)
        count= 0
        for i in range(len(val_pred)):
            if(val_pred[i]==float(va_d[i,3])):
                count=count+1
        val_acc.append(count/len(val_pred)*100)
        depth.append(clf.get_depth())
        nodes.append(clf.tree_.node_count)
    nodes = np.array(nodes)
    depth = np.array(depth)
    train_acc = np.array(train_acc)
    val_acc = np.array(val_acc)
    test_acc = np.array(test_acc)
    plt.plot(nodes,test_acc)
    plt.show()

    return

def part_d():
    tr_d = (train_in[["Age","Shape","Margin","Density","Severity"]]).to_numpy()
    te_d = (test_in[["Age","Shape","Margin","Density","Severity"]]).to_numpy()
    va_d = (val_in[["Age","Shape","Margin","Density","Severity"]]).to_numpy()
    x_rev,x_con,x_d,vec_rev,vec_con = process_train(tr_d)
    xf = hstack([hstack([x_rev,x_con]),x_d])
    x_rev,x_con,x_d,vec_rev,vec_con = process(te_d)
    xf1 = hstack([hstack([x_rev,x_con]),x_d])
    x_rev,x_con,x_d,vec_rev,vec_con = process(va_d)
    xf2 = hstack([hstack([x_rev,x_con]),x_d])
    n_estimators=[i for i in range(50,450,50)]
    maxfeatures=[i for i in range(0.4,0.8,0.1)]
    min_samples_split=[i for i in range(2,10,2)]
    grid={'n_estimators':n_estimators, 'maxfeatures':maxfeatures, 'min_samples_split':min_samples_split}
    tree=HalvingGridSearchCV(RandomForestClassifier(random_state=0,oob_score=True),grid)
    tree.fit(xf,tr_d[:,3].astype(np.float32))
    opt_clf=tree.best_estimator_
    train_pred = opt_clf.predict(xf)
    count= 0
    for i in range(len(train_pred)):
        if(train_pred[i]==float(tr_d[i,3])):
            count=count+1
    print((count/len(train_pred)*100))
    test_pred = opt_clf.predict(xf1)
    count= 0
    for i in range(len(test_pred)):
        if(test_pred[i]==float(te_d[i,3])):
            count=count+1
    print(count/len(test_pred)*100)
    val_pred = opt_clf.predict(xf2)
    count= 0
    for i in range(len(val_pred)):
        if(val_pred[i]==float(va_d[i,3])):
            count=count+1
    print(count/len(val_pred)*100)

    return

def part_f():
    tr_d = (train_in[["review","condition","date","rating"]]).to_numpy()
    te_d = (test_in[["review","condition","date","rating"]]).to_numpy()
    va_d = (val_in[["review","condition","date","rating"]]).to_numpy()


    x_rev,x_con,x_d,vec_rev,vec_con = process_train(tr_d)
    xf = hstack([hstack([x_rev,x_con]),x_d])
    opt_clf = LGBMClassifier()
    model = opt_clf.fit(xf,tr_d[:,3].astype(np.float32))
    #print(model.score(xf,tr_d[:,3].astype(np.float32)))

    x_rev,x_con,x_d = process(tr_d,vec_rev,vec_con)
    xf = hstack([hstack([x_rev,x_con]),x_d])
    train_pred = opt_clf.predict(xf)
    count= 0
    for i in range(len(train_pred)):
        if(train_pred[i]==float(tr_d[i,3])):
            count=count+1
    print(count/len(train_pred)*100)
    
    x_rev,x_con,x_d = process(te_d,vec_rev,vec_con)
    xf = hstack([hstack([x_rev,x_con]),x_d])
    train_pred = opt_clf.predict(xf)
    count= 0
    for i in range(len(train_pred)):
        if(train_pred[i]==float(te_d[i,3])):
            count=count+1
    print(count/len(train_pred)*100)
    

    x_rev,x_con,x_d = process(va_d,vec_rev,vec_con)
    xf = hstack([hstack([x_rev,x_con]),x_d])
    train_pred = opt_clf.predict(xf)
    count= 0
    for i in range(len(train_pred)):
        if(train_pred[i]==float(te_d[i,3])):
            count=count+1
    print(count/len(train_pred)*100)

    return 
  