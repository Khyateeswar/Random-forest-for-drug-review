import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
from sklearn.model_selection import GridSearchCV
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import HalvingGridSearchCV
from xgboost import XGBClassifier
import statistics

# train_path = str(sys.argv[1])
# test_path = str(sys.argv[2])
# val_path = str(sys.argv[3])

train_in = pd.read_csv('data1'+"/"+'train.csv')
test_in = pd.read_csv('data1'+"/"+'test.csv')
val_in = pd.read_csv('data1'+"/"+'val.csv')

def del_q(l):
    ans=[]
    t=[]
    for row in l:
        temp=[]
        if '?' not in row:
            temp.append(int(row[0]))
            temp.append(int(row[1]))
            temp.append(int(row[2]))
            temp.append(int(row[3]))
            t.append(int(row[4]))
            ans.append(temp)
    return [np.array(ans),np.array(t)]

def set_med(l):# changes to be done
    ans=[]
    t=[]
    for row in l:
        temp=[]
        if '?' not in row:
            temp.append(int(row[0]))
            temp.append(int(row[1]))
            temp.append(int(row[2]))
            temp.append(int(row[3]))
            t.append(int(row[4]))
        else :
            ls = []
            ind=[]
            for j in range(4):
                if(row[j]!="?"):
                    ls.append(row[j])
                else:
                    ind.append(j)
                temp.append(row[j])
            for k in range(len(ind)):
                temp[ind]=statistics.median(ls)
        ans.append(temp)
        t.append(row[4])
    return [np.array(ans),np.array(t)]
    
def set_mod(l):# changes to be done
    ans=[]
    t=[]
    for row in l:
        temp=[]
        if '?' not in row:
            temp.append(int(row[0]))
            temp.append(int(row[1]))
            temp.append(int(row[2]))
            temp.append(int(row[3]))
            t.append(int(row[4]))
        else :
            ls = []
            ind=[]
            for j in range(4):
                if(row[j]!="?"):
                    ls.append(row[j])
                else:
                    ind.append(j)
                temp.append(row[j])
            for k in range(len(ind)):
                temp[ind]=statistics.mode(ls)
        ans.append(temp)
        t.append(row[4])
    return [np.array(ans),np.array(t)]

def set_nan(l):
    ans=[]
    t=[]
    for row in l:
        temp=[]
        for j in range(4):
            if(row[j]=="?"):
                temp.append(np.nan)
            else:
                temp.append(float(row[j]))
        ans.append(temp)
        if(row[4]=="?"):
            t.append(np.nan)
        else:
            t.append(float(row[4])) 
    return [np.array(ans),np.array(t)]
    
def part_a():
    tr_d = (train_in[["Age","Shape","Margin","Density","Severity"]]).to_numpy()
    te_d = (test_in[["Age","Shape","Margin","Density","Severity"]]).to_numpy()
    va_d = (val_in[["Age","Shape","Margin","Density","Severity"]]).to_numpy()
    train_d =del_q(tr_d)[0]
    train_t = del_q(tr_d)[1]
    test_d = del_q(te_d)[0]
    test_t = del_q(te_d)[1]
    val_d = del_q(va_d)[0]
    val_t = del_q(va_d)[1]
    dectree_clf = DecisionTreeClassifier()
    dectree_clf.fit(train_d,train_t)
    train_pred = dectree_clf.predict(train_d)
    count= 0
    for i in range(len(train_pred)):
        if(train_pred[i]==train_t[i]):
            count=count+1
    print(count/len(train_pred)*100)
    test_pred = dectree_clf.predict(test_d)
    count= 0
    for i in range(len(test_pred)):
        if(test_pred[i]==test_t[i]):
            count=count+1
    print(count/len(test_pred)*100)
    val_pred = dectree_clf.predict(val_d)
    count= 0
    for i in range(len(val_pred)):
        if(val_pred[i]==val_t[i]):
            count=count+1
    print(count/len(val_pred)*100)
    
    fig,x = plt.subplots(figsize=(30,30))
    tree.plot_tree(dectree_clf,feature_names=["Age","Shape","Margin","Density"],class_names=['0','1'],filled=True)
    fig.savefig('1a.png')
    return 
    
def part_b():
    tr_d = (train_in[["Age","Shape","Margin","Density","Severity"]]).to_numpy()
    te_d = (test_in[["Age","Shape","Margin","Density","Severity"]]).to_numpy()
    va_d = (val_in[["Age","Shape","Margin","Density","Severity"]]).to_numpy()
    train_d =del_q(tr_d)[0]
    train_t = del_q(tr_d)[1]
    test_d = del_q(te_d)[0]
    test_t = del_q(te_d)[1]
    val_d = del_q(va_d)[0]
    val_t = del_q(va_d)[1]
    acc = 0
    par = [1,1,1]
    for i in range(1,50):
        for j in range(2,50):
            for k in range(1,50):
                clf = DecisionTreeClassifier(max_depth=i,min_samples_split=j,min_samples_leaf=k)
                clf.fit(train_d,train_t)
                train_pred = clf.predict(train_d)
                count= 0
                for y in range(len(train_pred)):
                    if(train_pred[y]==train_t[y]):
                        count=count+1
                vacc=count/len(train_pred)*100
                if(vacc>acc):
                    acc=vacc
                    par[0]=i
                    par[1]=j
                    par[2]=k
    print(par)
    opt_clf = DecisionTreeClassifier(max_depth=par[0],min_samples_split=par[1],min_samples_leaf=par[2])
    opt_clf.fit(train_d,train_t)
    train_pred = opt_clf.predict(train_d)
    count= 0
    for i in range(len(train_pred)):
        if(train_pred[i]==train_t[i]):
            count=count+1
    print(count/len(train_pred)*100)
    test_pred = opt_clf.predict(test_d)
    count= 0
    for i in range(len(test_pred)):
        if(test_pred[i]==test_t[i]):
            count=count+1
    print(count/len(test_pred)*100)
    val_pred = opt_clf.predict(val_d)
    count= 0
    for i in range(len(val_pred)):
        if(val_pred[i]==val_t[i]):
            count=count+1
    print(count/len(val_pred)*100)
    
    fig,x = plt.subplots(figsize=(30,30))
    tree.plot_tree(opt_clf,feature_names=["Age","Shape","Margin","Density"],class_names=['0','1'],filled=True)
    fig.savefig('1b.png')
    return
    
def part_c():
    tr_d = (train_in[["Age","Shape","Margin","Density","Severity"]]).to_numpy()
    te_d = (test_in[["Age","Shape","Margin","Density","Severity"]]).to_numpy()
    va_d = (val_in[["Age","Shape","Margin","Density","Severity"]]).to_numpy()
    train_d =del_q(tr_d)[0]
    train_t = del_q(tr_d)[1]
    test_d = del_q(te_d)[0]
    test_t = del_q(te_d)[1]
    val_d = del_q(va_d)[0]
    val_t = del_q(va_d)[1]
    dectree_clf = DecisionTreeClassifier()
    dectree_clf.fit(train_d,train_t)
    alphas=dectree_clf.cost_complexity_pruning_path(train_d,train_t)["ccp_alphas"]
    impur = dectree_clf.cost_complexity_pruning_path(train_d,train_t)["impurities"]
    print(len(alphas))
    print(len(impur))
    train_acc=[]
    test_acc=[]
    val_acc=[]
    nodes=[]
    depth=[]
    final_alpha=0
    val_max = 0
    for j in range(len(alphas)):
        clf = DecisionTreeClassifier(ccp_alpha=alphas[j])
        clf.fit(train_d,train_t)
        train_pred = clf.predict(train_d)
        count= 0
        for i in range(len(train_pred)):
            if(train_pred[i]==train_t[i]):
                count=count+1
        train_acc.append(count/len(train_pred)*100)
        test_pred = clf.predict(test_d)
        count= 0
        for i in range(len(test_pred)):
            if(test_pred[i]==test_t[i]):
                count=count+1
        test_acc.append(count/len(test_pred)*100)
        val_pred = clf.predict(val_d)
        count= 0
        for i in range(len(val_pred)):
            if(val_pred[i]==val_t[i]):
                count=count+1
        val_acc.append(count/len(val_pred)*100)
        if(count/len(val_pred)*100>val_max):
            val_max = count/len(val_pred)
            final_alpha = alphas[j]
        depth.append(clf.get_depth())
        nodes.append(clf.tree_.node_count)
    nodes = np.array(nodes)
    depth = np.array(depth)
    train_acc = np.array(train_acc)
    val_acc = np.array(val_acc)
    test_acc = np.array(test_acc)
    alphas = np.array(alphas)
    impur = np.array(impur)
    clf = DecisionTreeClassifier(ccp_alpha=final_alpha)
    clf.fit(train_d,train_t)
    train_pred = clf.predict(train_d)
    count= 0
    for i in range(len(train_pred)):
        if(train_pred[i]==train_t[i]):
            count=count+1
    print(count/len(train_pred)*100)
    test_pred = clf.predict(test_d)
    count= 0
    for i in range(len(test_pred)):
        if(test_pred[i]==test_t[i]):
            count=count+1
    print(count/len(test_pred)*100)
    val_pred = clf.predict(val_d)
    count= 0
    for i in range(len(val_pred)):
        if(val_pred[i]==val_t[i]):
            count=count+1
    print(count/len(val_pred)*100)
#     fig,x = plt.subplots(figsize=(30,30))
#     tree.plot_tree(clf,feature_names=["Age","Shape","Margin","Density"],class_names=['0','1'],filled=True)
#     fig.savefig('1c_tree.png')
    plt.plot(alphas,nodes)
    plt.ylabel("nodes")
    plt.xlabel("alphas")
    plt.savefig("1c_nodes_alpha.png")
    plt.clf()
    plt.plot(alphas,nodes)
    plt.ylabel("depth")
    plt.xlabel("alphas")
    plt.savefig("1c_depth_alpha.png")
    plt.clf()
    plt.plot(alphas,train_acc)
    plt.ylabel("Train accuracy")
    plt.xlabel("alphas")
    plt.savefig("1c_train_alpha.png")
    plt.clf()
    plt.plot(alphas,test_acc)
    plt.ylabel("Test accuracy")
    plt.xlabel("alphas")
    plt.savefig("1c_test_alpha.png")
    plt.clf()
    plt.plot(alphas,val_acc)
    plt.ylabel("Validation accuracy")
    plt.xlabel("alphas")
    plt.savefig("1c_val_alpha.png")
    plt.clf()
    plt.plot(alphas,impur)
    plt.xlabel("alphas")
    plt.ylabel("Impurity of Leaves")
    plt.savefig("1c_impur_alp.png")
    plt.clf()
    return

def part_d():
    tr_d = (train_in[["Age","Shape","Margin","Density","Severity"]]).to_numpy()
    te_d = (test_in[["Age","Shape","Margin","Density","Severity"]]).to_numpy()
    va_d = (val_in[["Age","Shape","Margin","Density","Severity"]]).to_numpy()
    train_d =del_q(tr_d)[0]
    train_t = del_q(tr_d)[1]
    test_d = del_q(te_d)[0]
    test_t = del_q(te_d)[1]
    val_d = del_q(va_d)[0]
    val_t = del_q(va_d)[1]
    n_estimators=[i for i in range(1,100,10)]
    max_features=[i for i in range(1,13)]
    min_samples_split=[i for i in range(2,13)]
    grid={'n_estimators':n_estimators, 'max_features':max_features, 'min_samples_split':min_samples_split}
    tree=HalvingGridSearchCV(RandomForestClassifier(random_state=0,oob_score=True),grid)
    tree.fit(train_d,train_t)
    opt_clf=tree.best_estimator_
    train_pred = opt_clf.predict(train_d)
    count= 0
    for i in range(len(train_pred)):
        if(train_pred[i]==train_t[i]):
            count=count+1
    print(count/len(train_pred)*100)
    test_pred = opt_clf.predict(test_d)
    count= 0
    for i in range(len(test_pred)):
        if(test_pred[i]==test_t[i]):
            count=count+1
    print(count/len(test_pred)*100)
    val_pred = opt_clf.predict(val_d)
    count= 0
    for i in range(len(val_pred)):
        if(val_pred[i]==val_t[i]):
            count=count+1
    print(count/len(val_pred)*100)
    
    return

def part_e():
    #median
    tr_d = (train_in[["Age","Shape","Margin","Density","Severity"]]).to_numpy()
    te_d = (test_in[["Age","Shape","Margin","Density","Severity"]]).to_numpy()
    va_d = (val_in[["Age","Shape","Margin","Density","Severity"]]).to_numpy()
    train_d =set_med(tr_d)[0]
    train_t = set_med(tr_d)[1]
    test_d = set_med(te_d)[0]
    test_t = set_med(te_d)[1]
    val_d = set_med(va_d)[0]
    val_t = set_med(va_d)[1]
    
    
    dectree_clf = DecisionTreeClassifier()
    dectree_clf.fit(train_d,train_t)
    train_pred = dectree_clf.predict(train_d)
    count= 0
    for i in range(len(train_pred)):
        if(train_pred[i]==train_t[i]):
            count=count+1
    print(count/len(train_pred)*100)
    test_pred = dectree_clf.predict(test_d)
    count= 0
    for i in range(len(test_pred)):
        if(test_pred[i]==test_t[i]):
            count=count+1
    print(count/len(test_pred)*100)
    val_pred = dectree_clf.predict(val_d)
    count= 0
    for i in range(len(val_pred)):
        if(val_pred[i]==val_t[i]):
            count=count+1
    print(count/len(val_pred)*100)
    
    fig,x = plt.subplots(figsize=(30,30))
    tree.plot_tree(dectree_clf,feature_names=["Age","Shape","Margin","Density"],class_names=['0','1'],filled=True)
    fig.savefig('1e_med_a.png')
    
    
    
    
    acc = 0
    par = [1,1,1]
    for i in range(1,50):
        for j in range(2,50):
            for k in range(1,50):
                clf = DecisionTreeClassifier(max_depth=i,min_samples_split=j,min_samples_leaf=k)
                clf.fit(train_d,train_t)
                train_pred = clf.predict(train_d)
                count= 0
                for y in range(len(train_pred)):
                    if(train_pred[y]==train_t[y]):
                        count=count+1
                vacc=count/len(train_pred)*100
                if(vacc>acc):
                    acc=vacc
                    par[0]=i
                    par[1]=j
                    par[2]=k
    print(par)
    opt_clf = DecisionTreeClassifier(max_depth=par[0],min_samples_split=par[1],min_samples_leaf=par[2])
    opt_clf.fit(train_d,train_t)
    train_pred = opt_clf.predict(train_d)
    count= 0
    for i in range(len(train_pred)):
        if(train_pred[i]==train_t[i]):
            count=count+1
    print(count/len(train_pred)*100)
    test_pred = opt_clf.predict(test_d)
    count= 0
    for i in range(len(test_pred)):
        if(test_pred[i]==test_t[i]):
            count=count+1
    print(count/len(test_pred)*100)
    val_pred = opt_clf.predict(val_d)
    count= 0
    for i in range(len(val_pred)):
        if(val_pred[i]==val_t[i]):
            count=count+1
    print(count/len(val_pred)*100)
    
    fig,x = plt.subplots(figsize=(30,30))
    tree.plot_tree(opt_clf,feature_names=["Age","Shape","Margin","Density"],class_names=['0','1'],filled=True)
    fig.savefig('1e_med_b.png')
    
    
    
    dectree_clf = DecisionTreeClassifier()
    dectree_clf.fit(train_d,train_t)
    alphas=dectree_clf.cost_complexity_pruning_path(train_d,train_t)["ccp_alphas"]
    impur = dectree_clf.cost_complexity_pruning_path(train_d,train_t)["impurities"]
    print(len(alphas))
    print(len(impur))
    train_acc=[]
    test_acc=[]
    val_acc=[]
    nodes=[]
    depth=[]
    final_alpha=0
    val_max = 0
    for j in range(len(alphas)):
        clf = DecisionTreeClassifier(ccp_alpha=alphas[j])
        clf.fit(train_d,train_t)
        train_pred = clf.predict(train_d)
        count= 0
        for i in range(len(train_pred)):
            if(train_pred[i]==train_t[i]):
                count=count+1
        train_acc.append(count/len(train_pred)*100)
        test_pred = clf.predict(test_d)
        count= 0
        for i in range(len(test_pred)):
            if(test_pred[i]==test_t[i]):
                count=count+1
        test_acc.append(count/len(test_pred)*100)
        val_pred = clf.predict(val_d)
        count= 0
        for i in range(len(val_pred)):
            if(val_pred[i]==val_t[i]):
                count=count+1
        val_acc.append(count/len(val_pred)*100)
        if(count/len(val_pred)*100>val_max):
            val_max = count/len(val_pred)
            final_alpha = alphas[j]
        depth.append(clf.get_depth())
        nodes.append(clf.tree_.node_count)
    nodes = np.array(nodes)
    depth = np.array(depth)
    train_acc = np.array(train_acc)
    val_acc = np.array(val_acc)
    test_acc = np.array(test_acc)
    alphas = np.array(alphas)
    impur = np.array(impur)
    clf = DecisionTreeClassifier(ccp_alpha=final_alpha)
    clf.fit(train_d,train_t)
    train_pred = clf.predict(train_d)
    count= 0
    for i in range(len(train_pred)):
        if(train_pred[i]==train_t[i]):
            count=count+1
    print(count/len(train_pred)*100)
    test_pred = clf.predict(test_d)
    count= 0
    for i in range(len(test_pred)):
        if(test_pred[i]==test_t[i]):
            count=count+1
    print(count/len(test_pred)*100)
    val_pred = clf.predict(val_d)
    count= 0
    for i in range(len(val_pred)):
        if(val_pred[i]==val_t[i]):
            count=count+1
    print(count/len(val_pred)*100)
#     fig,x = plt.subplots(figsize=(30,30))
#     tree.plot_tree(clf,feature_names=["Age","Shape","Margin","Density"],class_names=['0','1'],filled=True)
#     fig.savefig('1c_tree.png')
    plt.plot(alphas,nodes)
    plt.ylabel("nodes")
    plt.xlabel("alphas")
    plt.savefig("1e_med_nodes_alpha.png")
    plt.clf()
    plt.plot(alphas,nodes)
    plt.ylabel("depth")
    plt.xlabel("alphas")
    plt.savefig("1e_med_depth_alpha.png")
    plt.clf()
    plt.plot(alphas,train_acc)
    plt.ylabel("Train accuracy")
    plt.xlabel("alphas")
    plt.savefig("1e_med_train_alpha.png")
    plt.clf()
    plt.plot(alphas,test_acc)
    plt.ylabel("Test accuracy")
    plt.xlabel("alphas")
    plt.savefig("1e_med_test_alpha.png")
    plt.clf()
    plt.plot(alphas,val_acc)
    plt.ylabel("Validation accuracy")
    plt.xlabel("alphas")
    plt.savefig("1e_med_val_alpha.png")
    plt.clf()
    plt.plot(alphas,impur)
    plt.xlabel("alphas")
    plt.ylabel("Impurity of Leaves")
    plt.savefig("1e_med_impur_alp.png")
    plt.clf()
    
    
    n_estimators=[i for i in range(1,100,10)]
    max_features=[i for i in range(1,13)]
    min_samples_split=[i for i in range(2,13)]
    grid={'n_estimators':n_estimators, 'max_features':max_features, 'min_samples_split':min_samples_split}
    tree=HalvingGridSearchCV(RandomForestClassifier(random_state=0,oob_score=True),grid)
    tree.fit(train_d,train_t)
    opt_clf=tree.best_estimator_
    train_pred = opt_clf.predict(train_d)
    count= 0
    for i in range(len(train_pred)):
        if(train_pred[i]==train_t[i]):
            count=count+1
    print(count/len(train_pred)*100)
    test_pred = opt_clf.predict(test_d)
    count= 0
    for i in range(len(test_pred)):
        if(test_pred[i]==test_t[i]):
            count=count+1
    print(count/len(test_pred)*100)
    val_pred = opt_clf.predict(val_d)
    count= 0
    for i in range(len(val_pred)):
        if(val_pred[i]==val_t[i]):
            count=count+1
    print(count/len(val_pred)*100)
    
    
    #mode
    tr_d = (train_in[["Age","Shape","Margin","Density","Severity"]]).to_numpy()
    te_d = (test_in[["Age","Shape","Margin","Density","Severity"]]).to_numpy()
    va_d = (val_in[["Age","Shape","Margin","Density","Severity"]]).to_numpy()
    train_d =set_mod(tr_d)[0]
    train_t = set_mod(tr_d)[1]
    test_d = set_mod(te_d)[0]
    test_t = set_mod(te_d)[1]
    val_d = set_mod(va_d)[0]
    val_t = set_mod(va_d)[1]
    
    dectree_clf = DecisionTreeClassifier()
    dectree_clf.fit(train_d,train_t)
    train_pred = dectree_clf.predict(train_d)
    count= 0
    for i in range(len(train_pred)):
        if(train_pred[i]==train_t[i]):
            count=count+1
    print(count/len(train_pred)*100)
    test_pred = dectree_clf.predict(test_d)
    count= 0
    for i in range(len(test_pred)):
        if(test_pred[i]==test_t[i]):
            count=count+1
    print(count/len(test_pred)*100)
    val_pred = dectree_clf.predict(val_d)
    count= 0
    for i in range(len(val_pred)):
        if(val_pred[i]==val_t[i]):
            count=count+1
    print(count/len(val_pred)*100)
    
    fig,x = plt.subplots(figsize=(30,30))
    tree.plot_tree(dectree_clf,feature_names=["Age","Shape","Margin","Density"],class_names=['0','1'],filled=True)
    fig.savefig('1e_mod_a.png')
    
    
    
    
    acc = 0
    par = [1,1,1]
    for i in range(1,50):
        for j in range(2,50):
            for k in range(1,50):
                clf = DecisionTreeClassifier(max_depth=i,min_samples_split=j,min_samples_leaf=k)
                clf.fit(train_d,train_t)
                train_pred = clf.predict(train_d)
                count= 0
                for y in range(len(train_pred)):
                    if(train_pred[y]==train_t[y]):
                        count=count+1
                vacc=count/len(train_pred)*100
                if(vacc>acc):
                    acc=vacc
                    par[0]=i
                    par[1]=j
                    par[2]=k
    print(par)
    opt_clf = DecisionTreeClassifier(max_depth=par[0],min_samples_split=par[1],min_samples_leaf=par[2])
    opt_clf.fit(train_d,train_t)
    train_pred = opt_clf.predict(train_d)
    count= 0
    for i in range(len(train_pred)):
        if(train_pred[i]==train_t[i]):
            count=count+1
    print(count/len(train_pred)*100)
    test_pred = opt_clf.predict(test_d)
    count= 0
    for i in range(len(test_pred)):
        if(test_pred[i]==test_t[i]):
            count=count+1
    print(count/len(test_pred)*100)
    val_pred = opt_clf.predict(val_d)
    count= 0
    for i in range(len(val_pred)):
        if(val_pred[i]==val_t[i]):
            count=count+1
    print(count/len(val_pred)*100)
    
    fig,x = plt.subplots(figsize=(30,30))
    tree.plot_tree(opt_clf,feature_names=["Age","Shape","Margin","Density"],class_names=['0','1'],filled=True)
    fig.savefig('1e_mod_b.png')
    
    
    
    dectree_clf = DecisionTreeClassifier()
    dectree_clf.fit(train_d,train_t)
    alphas=dectree_clf.cost_complexity_pruning_path(train_d,train_t)["ccp_alphas"]
    impur = dectree_clf.cost_complexity_pruning_path(train_d,train_t)["impurities"]
    print(len(alphas))
    print(len(impur))
    train_acc=[]
    test_acc=[]
    val_acc=[]
    nodes=[]
    depth=[]
    final_alpha=0
    val_max = 0
    for j in range(len(alphas)):
        clf = DecisionTreeClassifier(ccp_alpha=alphas[j])
        clf.fit(train_d,train_t)
        train_pred = clf.predict(train_d)
        count= 0
        for i in range(len(train_pred)):
            if(train_pred[i]==train_t[i]):
                count=count+1
        train_acc.append(count/len(train_pred)*100)
        test_pred = clf.predict(test_d)
        count= 0
        for i in range(len(test_pred)):
            if(test_pred[i]==test_t[i]):
                count=count+1
        test_acc.append(count/len(test_pred)*100)
        val_pred = clf.predict(val_d)
        count= 0
        for i in range(len(val_pred)):
            if(val_pred[i]==val_t[i]):
                count=count+1
        val_acc.append(count/len(val_pred)*100)
        if(count/len(val_pred)*100>val_max):
            val_max = count/len(val_pred)
            final_alpha = alphas[j]
        depth.append(clf.get_depth())
        nodes.append(clf.tree_.node_count)
    nodes = np.array(nodes)
    depth = np.array(depth)
    train_acc = np.array(train_acc)
    val_acc = np.array(val_acc)
    test_acc = np.array(test_acc)
    alphas = np.array(alphas)
    impur = np.array(impur)
    clf = DecisionTreeClassifier(ccp_alpha=final_alpha)
    clf.fit(train_d,train_t)
    train_pred = clf.predict(train_d)
    count= 0
    for i in range(len(train_pred)):
        if(train_pred[i]==train_t[i]):
            count=count+1
    print(count/len(train_pred)*100)
    test_pred = clf.predict(test_d)
    count= 0
    for i in range(len(test_pred)):
        if(test_pred[i]==test_t[i]):
            count=count+1
    print(count/len(test_pred)*100)
    val_pred = clf.predict(val_d)
    count= 0
    for i in range(len(val_pred)):
        if(val_pred[i]==val_t[i]):
            count=count+1
    print(count/len(val_pred)*100)
#     fig,x = plt.subplots(figsize=(30,30))
#     tree.plot_tree(clf,feature_names=["Age","Shape","Margin","Density"],class_names=['0','1'],filled=True)
#     fig.savefig('1c_tree.png')
    plt.plot(alphas,nodes)
    plt.ylabel("nodes")
    plt.xlabel("alphas")
    plt.savefig("1e_mod_nodes_alpha.png")
    plt.clf()
    plt.plot(alphas,nodes)
    plt.ylabel("depth")
    plt.xlabel("alphas")
    plt.savefig("1e_mod_depth_alpha.png")
    plt.clf()
    plt.plot(alphas,train_acc)
    plt.ylabel("Train accuracy")
    plt.xlabel("alphas")
    plt.savefig("1e_mod_train_alpha.png")
    plt.clf()
    plt.plot(alphas,test_acc)
    plt.ylabel("Test accuracy")
    plt.xlabel("alphas")
    plt.savefig("1e_mod_test_alpha.png")
    plt.clf()
    plt.plot(alphas,val_acc)
    plt.ylabel("Validation accuracy")
    plt.xlabel("alphas")
    plt.savefig("1e_mod_val_alpha.png")
    plt.clf()
    plt.plot(alphas,impur)
    plt.xlabel("alphas")
    plt.ylabel("Impurity of Leaves")
    plt.savefig("1e_mod_impur_alp.png")
    plt.clf()
    
    
    n_estimators=[i for i in range(1,100,10)]
    max_features=[i for i in range(1,13)]
    min_samples_split=[i for i in range(2,13)]
    grid={'n_estimators':n_estimators, 'max_features':max_features, 'min_samples_split':min_samples_split}
    tree=HalvingGridSearchCV(RandomForestClassifier(random_state=0,oob_score=True),grid)
    tree.fit(train_d,train_t)
    opt_clf=tree.best_estimator_
    train_pred = opt_clf.predict(train_d)
    count= 0
    for i in range(len(train_pred)):
        if(train_pred[i]==train_t[i]):
            count=count+1
    print(count/len(train_pred)*100)
    test_pred = opt_clf.predict(test_d)
    count= 0
    for i in range(len(test_pred)):
        if(test_pred[i]==test_t[i]):
            count=count+1
    print(count/len(test_pred)*100)
    val_pred = opt_clf.predict(val_d)
    count= 0
    for i in range(len(val_pred)):
        if(val_pred[i]==val_t[i]):
            count=count+1
    print(count/len(val_pred)*100)
    return


def part_f():
    tr_d = (train_in[["Age","Shape","Margin","Density","Severity"]]).to_numpy()
    te_d = (test_in[["Age","Shape","Margin","Density","Severity"]]).to_numpy()
    va_d = (val_in[["Age","Shape","Margin","Density","Severity"]]).to_numpy()
    trs=set_nan(tr_d)
    tes=set_nan(te_d)
    vals = set_nan(va_d)
    train_d =trs[0]
    train_t = trs[1]
    test_d = tes[0]
    test_t = tes[1]
    val_d = vals[0]
    val_t = vals[1]
    
    
    subsample=[0.1,0.2,0.3,0.4,0.5]
    n_estimators=[i for i in range(10,50,10)]
    max_depth=[i for i in range(3,10)]

    grid={'subsample':subsample, 'n_estimators':n_estimators, 'max_depth':max_depth}
    tree=GridSearchCV(XGBClassifier(),grid)
    tree.fit(train_d,train_t)

    opt_clf=tree.best_estimator_
    train_pred = opt_clf.predict(train_d)
    count= 0
    for i in range(len(train_pred)):
        if(train_pred[i]==train_t[i]):
            count=count+1
    print(count/len(train_pred)*100)
    test_pred = opt_clf.predict(test_d)
    count= 0
    for i in range(len(test_pred)):
        if(test_pred[i]==test_t[i]):
            count=count+1
    print(count/len(test_pred)*100)
    val_pred = opt_clf.predict(val_d)
    count= 0
    for i in range(len(val_pred)):
        if(val_pred[i]==val_t[i]):
            count=count+1
    print(count/len(val_pred)*100)
    return




   