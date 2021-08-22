
import numpy as np
from sklearn import svm

def get_date(s):
    arr = s.split('-')
    d = arr[0]
    m = arr[1]
    y = arr[2]
    day = int(d)
    month = int(m)
    year = int(y)
    k = year-2015
    l = month-1
    p = day-1
    return k*365+l*30+p+190

def num_zeros():
    cl = 0
    for i in range(882):
        id1 = int(of_test[i][0])
        if((int(dict_of_ids[id1])) == 0):
                cl+=1
    return

def save_prediction(s):
    l = [['id','left']]
    for i in range(882):
        id1 = int(of_test[i][0])
        l.append([of_test[i][0],str(int(dict_of_ids[id1]))])
    l = np.array(l)
    np.savetxt(s+'.csv', l, delimiter=',',fmt = '%s')
    
def get_error_rate(pred, actual) :
    n = pred.shape[0]
    x = 0
    for i in range(n):
        if pred[i] == 1 and actual[i] == 1:
            x+=5
        elif pred[i] == -1 and actual[i] == -1:
            x+=1
    y = 0
    for i in range(n):
        if actual[i] == 1:
            y+=5
        elif actual[i] == -1:
            y+=1
    accuracy = 1
    if(y!=0):
        accuracy = x/y
    return (1-accuracy)

def split_train_valid_data(X_train,Y_train,p):
    if(len(X_train) == 0):
        A = np.array([])
        return [A,A,A,A]
    X = np.array(X_train)
    Y = np.array(Y_train,dtype = 'i8')
    X = X.reshape(-1,1)
    a = X
    b = Y
    #print(a.shape)
    #print(b.shape)
    c = np.c_[a.reshape(len(a), -1), b.reshape(len(b), -1)]
    np.random.shuffle(c)
    a_new = []
    z = c.T
    for i in range(X.shape[1]):
        a_new.append(z[i,:])
    a_new = np.array(a_new)
    a_new = a_new.T
    b_new = z[X.shape[1],:]
    X_new = a_new
    Y_new = []
    for i in b_new:
        Y_new.append(int(i))
    Y_new = np.array(Y_new)
    n = X_new.shape[0]
    a = int(p*n)
    valid_X,train_X = X_new[:a,0], X_new[a:,0]
    valid_Y,train_Y = Y_new[:a], Y_new[a:]
    return [train_X,train_Y,valid_X,valid_Y]

def return_class(identifier):
    x2 = 0
    if identifier in dict_of_remarks:
        x2 = 1
    x3 = 1
    if identifier in dict_of_remarks_supp_opp_id:
        x3 = 3
    return (x2+x3)

def solve_for_class(class_val):
    X_train = dict_of_class_with_data[class_val][0]
    Y_train = dict_of_class_with_data[class_val][1]
    X_test = dict_of_class_with_data[class_val][2]
    id_list = dict_of_class_with_data[class_val][3]
    test_data = dict_of_class_with_data[class_val][4]
    test_actual = dict_of_class_with_data[class_val][5]
    id_list2 = dict_of_class_with_data[class_val][6]
    #test_pred = return_result(X_train,Y_train,test_data,class_val)
    Y_test = return_result(X_train,Y_train,X_test,class_val)
    for i in range(len(id_list)):
        dict_of_ids[id_list[i]] = (Y_test[i]+1)/2


def return_result(X_train,Y_train,X_test,class_no):
    ret_val = np.zeros(len(X_test))
    for i in range(3):
        l = split_train_valid_data(X_train,Y_train,0.2)
        ret_val = ret_val + ret_pred(l[0],l[1],l[2],l[3],X_test,class_no)
    Y = np.zeros(len(X_test))
    for i in range(len(X_test)):
        if(ret_val[i]>=0):
            Y[i] = 1
        else:
            Y[i] = -1
    return Y

def ret_pred(a,b,c,d,e,f):
    X_train = get_attributes(a,f)
    Y_train = np.array(b)
    X_valid = get_attributes(c,f)
    X_test = get_attributes(e,f)
    Y_valid = np.array(d)
    return prediction_set(X_train,Y_train,X_valid,Y_valid,X_test)


def get_test_accuracies():
    n = len(id2_list)
    id_list = np.array(id2_list)
    half_num = int(n/2)
    arr = np.random.choice(range(n), half_num, replace=False)
    arr2 = list(set(range(n)) - set(arr))
    first_half = id_list[arr]
    second_half = id_list[arr2]
    print(str(get_accuracy(first_half)))
    print(str(get_accuracy(second_half)))
    
def get_accuracy(arr):
    a = 0
    c = 0
    for id2 in arr:
        l = dict_of_ids2[id2]
        if(l[1]==1):
            c+=5
        else:
            c+=1
        if(l[1] == l[0]):
            if(l[0] == 1):
                a+=5
            else:
                a+=1
    return a/c


def prediction_set(X_train,Y_train,X_valid,Y_valid,X_test):
    #ADD SVM HERE
    min_error = 1
    min_sum = 0
    X = X_train
    Y = Y_train
    min_rf = 0
    min_criterion = ''
    for num_trees in [40,60,80]:
        for criterion in ['accuracy','entropy']:
            for a in [0.5,0.75]:
                for b in [0.5]:
                    for num_nodes_stop in [10,30,50]:
                        rf =  train_random_forest(X, Y, num_trees, num_nodes_stop,criterion, a, b)
                        valid_pred = eval_rf(rf,X_valid)
                        error = get_error_rate(valid_pred,Y_valid)
                        if (error < min_error):
                            min_error = error
                            min_sum = np.sum(valid_pred)
                            min_rf = rf
                            min_criterion = criterion
                        if(error == 1 and min_error == 1):
                            min_rf = rf
    #print(str(min_error)+" "+str(min_sum)+" "+str(len(Y_valid))+" "+str(np.sum(Y_valid)))
    valid_Y = Y_valid
    train_X = X
    train_Y = Y
    valid_X = X_valid
    flag = 0
    min_C = 0
    min_gamma = 0
    min_clf = 0
    min_pred =  np.zeros(len(valid_Y))
    for C1 in [1e-4,0.001,0.01,0.1,1,10,100,1000]:
        for gamma1 in [1e-5,1e-4,0.001,0.01,0.1,1,10,100,1000]:
            classif_algo =  svm.SVC(C=C1, gamma = gamma1,class_weight={1: 5, -1:1})
            X = train_X
            Y = train_Y
            classifier = classif_algo.fit(X,Y)
            #print(valid_Y)
            valid_pred = ret_val(valid_X,classifier)
            error = get_error_rate(valid_pred,valid_Y)
            #print(train_Y)
            train_pred = ret_val(train_X,classifier)
            #print(str(error)+" "+str(C1)+" "+str(gamma1)+" "+str(np.sum(valid_pred))+" "+str(np.sum(valid_Y))+" ")
            #print(str(np.sum(train_pred))+" "+str(np.sum(train_Y)))
            if (error<=min_error):
                flag = 1
                min_error = error
                min_sum = np.sum(valid_pred)
                min_C = C1
                min_gamma = gamma1
                min_clf = classifier
                min_pred = valid_pred
            if(error == 1 and min_error == 1):
                min_clf = classifier
    
    #print(str(min_error)+" "+str(min_sum)+" "+str(len(Y_valid))+" "+str(np.sum(Y_valid))+" "+str(flag))
    if(flag == 0):
        return eval_rf(min_rf,X_test)
    else:
        return ret_val(X_test,min_clf)

def get_attr(identifier,class_no):
    #FROM RATINGS
    ret_l = []
    l = dict_of_ratings[identifier]
    length = len(l)
    first_date = l[0][0]
    last_date = l[length-1][0]
    first_rating = l[0][1]
    last_rating = l[length-1][1]
    avg_rating = 0
    for i in l:
        avg_rating = avg_rating+(i[1]/length)
    ret_l.extend([length,first_date,last_date,first_rating,last_rating,avg_rating,last_date-first_date,last_date-first_date/length])
    if(class_no == 2 or class_no == 4):
        #FROM REMARKS
        l = dict_of_remarks[identifier]
        length = len(l)
        first_date = l[length-1][0]
        last_date = l[0][0]
        first_length = l[length-1][2]
        last_length = l[0][2]
        avg_rating = 0
        for i in l:
            avg_rating = avg_rating+(i[2]/length)
        ret_l.extend([length,first_date,last_date,first_length,last_length,avg_rating,last_date-first_date,last_date-first_date/length])
    if(class_no == 3 or class_no == 4):
        #FROM REMARKS_SUPP_OPP
        l = dict_of_remarks_supp_opp_id[identifier]
        ret_l.extend([len(l[0]),len(l[1])])
    return np.array(ret_l)


def get_attributes(X,class_no):
    l = []
    for i in X:
        l.append(get_attr(i,class_no))
    return np.array(l)


# CodeWrite cell
# Write Random Forest classifier. 
def h(a):
    if(a == 1 or a==0):
        return 0
    b = 1-a
    return (5*a*np.log2(1/a)+b*np.log2(1/b))
def entropy(pl,ql,pr,qr):
    return -(pl*h(ql)+pr*h(qr))

def value_of_criterion(criterion,total_left_points,total_right_points,pos_left_points,pos_right_points,left_val):
    if criterion == 'accuracy':
        z = (pos_left_points)+(pos_right_points)
        if left_val == -1:
            return ((total_left_points)-(pos_left_points)+4*(pos_right_points))/(4*z+total_left_points+total_right_points)
        else:
            return ((total_right_points)+4*(pos_left_points)-(pos_right_points))/(4*z+total_left_points+total_right_points)
    else:
        total_points = total_left_points+total_right_points
        if(total_left_points == 0):
            return entropy(total_left_points/total_points,1,total_right_points/total_points,pos_right_points/total_right_points)
        if(total_right_points == 0):
            return entropy(total_left_points/total_points,pos_left_points/total_left_points,total_right_points/total_points,1)
        return entropy(total_left_points/total_points,pos_left_points/total_left_points,
                       total_right_points/total_points,pos_right_points/total_right_points)   

def max_min(X_train,d):
    maxi = -5000
    mini = 5000
    for i in range(X_train.shape[0]):
        if(maxi<X_train[i][d]):
            maxi = X_train[i][d]
        if(mini>X_train[i][d]):
            mini = X_train[i][d]
    return (maxi,mini)


def args_to_criterion(X_train,Y_train,d,value):
    total_left_points = 0
    total_right_points = 0
    pos_left_points = 0
    pos_right_points = 0
    for i in range(X_train.shape[0]):
        if(X_train[i][d]<value):
            total_left_points+=1
            if(Y_train[i] == 1):
                pos_left_points+=1
        else:
            total_right_points+=1
            if(Y_train[i] == 1):
                pos_right_points+=1
    return (total_left_points,total_right_points,pos_left_points,pos_right_points)


def get_tree_less_attr(X, Y,node_val,node_pos,num_nodes_stop,criterion,d2):
    n = X.shape[0]
    d = X.shape[1]
    a = []
    #print(n)
    if(sum(Y)==n):
        t = [node_pos,0,-1,-1,1]
        a.append(np.array(t))
        return np.array(a)
    elif(sum(Y)==-n):
        t = [node_pos,0,-1,-1,-1]
        a.append(np.array(t))
        return np.array(a)
    elif(n<=num_nodes_stop):
        t = [node_pos,0,-1,-1,node_val]
        a.append(np.array(t))
        return np.array(a)
    else:
        to_comp_val_of_cr = -5000
        req_d1 = 0
        req_split_val = 0
        req_left_val = 0
        req_k = 0
        l = int(d*d2)
        co_or_check = np.random.choice(np.arange(d),l)
        #print(co_or_check)
        for d1 in co_or_check:
            maxi,mini = max_min(X,d1)
            diff = maxi-mini
            for k in range(12):
                val = mini+k*diff/10
                a,b,c,d3 = args_to_criterion(X,Y,d1,val)
                #print(d3)
                #print(str(a)+" "+str(b)+" "+str(c)+" "+str(d3)+" "+str(k)+" "+str(d1))
                x1 = value_of_criterion(criterion,a,b,c,d3,-1)
                x2 = value_of_criterion(criterion,a,b,c,d3,1)
                y1 = value_of_criterion('accuracy',a,b,c,d3,-1)
                y2 = value_of_criterion('accuracy',a,b,c,d3,1)
                if(x1<x2):
                    to_comp = x2
                    #print(x2)
                    if(to_comp_val_of_cr<to_comp):
                        to_comp_val_of_cr=to_comp
                        req_left_val = 1
                        req_split_val = val
                        req_d1 = d1
                        req_k = k
                elif(x1>x2):
                    to_comp = x1
                    #print(x1)
                    if(to_comp_val_of_cr<to_comp):
                        to_comp_val_of_cr=to_comp
                        req_left_val = -1
                        req_split_val = val
                        req_d1 = d1
                        req_k = k
                else:
                    to_comp = x1
                    #print(x1)
                    if(to_comp_val_of_cr<=to_comp):
                        to_comp_val_of_cr=to_comp
                        if(y1>=y2):
                            req_left_val = -1
                        else:
                            req_left_val = 1
                        req_split_val = val
                        req_d1 = d1
                        req_k = k
        #print(str(a)+" "+str(b)+" "+str(c)+" "+str(d3)+" ")
        #print(str(to_comp_val_of_cr)+" +nodepos:"+str(node_pos)+" k="+str(req_k))
        if(req_k == 0 or req_k == 11):
            a = []
            t = [node_pos,0,-1,-1,node_val]
            a.append(np.array(t))
            return np.array(a)
        X1= []
        Y1 = []
        X2=[]
        Y2 = []
        for i in range(n):
            if(X[i][req_d1]<req_split_val):
                X1.append(X[i])
                Y1.append(Y[i])
            else:
                X2.append(X[i])
                Y2.append(Y[i])
        X1 = np.array(X1)
        X2 = np.array(X2)
        Y1 = np.array(Y1)
        Y2 = np.array(Y2)
        a = 2*node_pos
        child1 = get_tree_less_attr(X1, Y1,req_left_val,a,num_nodes_stop, criterion,d2)
        child2 = get_tree_less_attr(X2, Y2,0-req_left_val,a+1,num_nodes_stop, criterion,d2)
        a = []
        t = [node_pos,1,req_d1,req_split_val,node_val]
        a.append(np.array(t))
        a.extend(child1)
        a.extend(child2)
        return np.array(a)    

def train_decision_tree_new(X, Y,b,num_nodes_stop, criterion):
    n = X.shape[0]
    d = X.shape[1]
    a = []
    n1 = 0
    n0 = 0
    for i in range(n):
        if(Y[i] == -1):
            n0+=1
        else:
            n1+=1
    if(n1>n0):
        return get_tree_less_attr(X, Y,1,1,num_nodes_stop, criterion,b)
    else:
        return get_tree_less_attr(X, Y,-1,1,num_nodes_stop, criterion,b)

    
def train_random_forest(X, Y, num_trees=10, num_nodes_stop=1, 
                        criterion='accuracy', a=0.5, b=0.5):
    R_F = []
    for ac in range(num_trees):
        #print(ac)
        n = X.shape[0]
        d = X.shape[1]
        k = int(n*a)
        train_data = np.random.choice(np.arange(n),k)
        tree = train_decision_tree_new(X[train_data], Y[train_data],b, num_nodes_stop, criterion)
        R_F.append(tree)
    return np.array(R_F)

""" Returns a random forest trained on X and Y. 
Trains num_trees.
Stops splitting nodes in each tree when a node has hit a size of "num_nodes_stop" or lower.
Split criterion can be either 'accuracy' or 'entropy'.
Fraction of data used per tree = a
Fraction of features used in each node = b
Returns a random forest (In whatever format that you find appropriate)
"""

def eval_decision_single_tree(tree, test_X):
    """ Takes in a tree, and a bunch of instances X and 
    returns the tree predicted values at those instances.
    """
    a = tree
    #print(tree)
    a = a[a[:,0].argsort(kind='mergesort')]
    this_dict = {a[i][0]: a[i] for i in range(a.shape[0])}
    #print(this_dict)
    Y = []
    #print(a)
    for elt in test_X:
        t = 1
        while this_dict[t][1] > 0:
            l = this_dict[t]
            #print(l)
            if(elt[int(l[2])]<l[3]):
                pos = l[0]
                t = int(2*pos)
            else:
                pos = l[0]
                t = int(2*pos+1)
            #print(this_dict[t][1])
        Y.append(this_dict[t][4])
    return np.array(Y)


def eval_rf(random_forest, test_X):
    num_trees = random_forest.shape[0]
    Y = np.zeros(test_X.shape[0])
    for tree in random_forest:
        Y = Y + eval_decision_single_tree(tree, test_X)
    Y = Y/num_trees
    for i in range(Y.shape[0]):
        if(Y[i]>0):
            Y[i] = 1
        else:
            Y[i] = -1
    return Y

    """ Takes in a tree, and a bunch of instances X and 
returns the tree predicted values at those instances."""




def get_SVM(train_X,train_Y,valid_X,valid_Y,X_test):
    min_error = 1
    min_sum = 0
    min_C = 0
    min_gamma = 0
    min_clf = 0
    min_pred =  np.zeros(len(valid_Y))
    for C1 in [0.001,0.01,0.1,1,10,100,1000]:
        for gamma1 in [0.001,0.01,0.1,1,10,100,1000]:
            classif_algo =  svm.SVC(C=C1, gamma = gamma1,class_weight={1: 5, -1:1})
            X = train_X
            Y = train_Y
            classifier = classif_algo.fit(X,Y)
            #print(valid_Y)
            valid_pred = ret_val(valid_X,classifier)
            error = get_error_rate(valid_pred,valid_Y)
            #print(train_Y)
            train_pred = ret_val(train_X,classifier)
            #print(str(error)+" "+str(C1)+" "+str(gamma1)+" "+str(np.sum(valid_pred))+" "+str(np.sum(valid_Y))+" ")
            #print(str(np.sum(train_pred))+" "+str(np.sum(train_Y)))
            if (error< min_error):
                min_error = error
                min_sum = np.sum(valid_pred)
                min_C = C1
                min_gamma = gamma1
                min_clf = classifier
                min_pred = valid_pred
            if(error == 1 and min_error == 1):
                min_clf = classifier
    Y_pred = ret_val(X_test,min_clf)
    return Y_pred
    
def ret_val(valid_X,classifier):
    Y_test_pred = classifier.predict(valid_X)
    #print(Y_test_pred)
    Y = []
    for i in range(len(valid_X)):
        if(Y_test_pred[i] >= 0):
            Y.append(1)
        else:
            Y.append(-1)
    return np.array(Y)



np.random.seed(10)



types = [np.dtype('U50'),'i8',np.dtype('U50'),np.dtype('U50'),'i8']
of_train = np.genfromtxt('train.csv', delimiter=',', dtype=types,comments = '@')
x = np.arange(1,of_train.shape[0])
of_train = of_train[x]
x = []
for i in range(of_train.shape[0]):
    if(of_train[i][1] >= 0):
        x.append(i)
of_train = of_train[x]

types = ['i8',np.dtype('U50'),np.dtype('U50'),'i8']
of_ratings = np.genfromtxt('ratings.csv', delimiter=',', dtype=types,comments = '@')
x = np.arange(1,of_ratings.shape[0])
of_ratings = of_ratings[x]
x = []
for i in range(of_ratings.shape[0]):
    if(of_ratings[i][0] >= 0):
        x.append(i)
of_ratings = of_ratings[x]
a = np.array(of_ratings)

types = ['i8',np.dtype('U50'),np.dtype('U50'),np.dtype('U5000'),np.dtype('U50')]
of_remarks = np.genfromtxt('remarks.csv', delimiter=',', dtype=types,comments = '@')
x = np.arange(1,of_remarks.shape[0]-1)
of_remarks = of_remarks[x]
x = []
for i in range(of_remarks.shape[0]):
    if(of_remarks[i][0] >= 0):
        x.append(i)
of_remarks = of_remarks[x]

types = ['i8',np.dtype('U50'),np.dtype('U50'),np.dtype('U50'),np.dtype('U50')]
of_remarks_supp_opp = np.genfromtxt('remarks_supp_opp.csv', delimiter=',', dtype=types,comments = '@')
x = np.arange(1,of_remarks_supp_opp.shape[0])
of_remarks_supp_opp = of_remarks_supp_opp[x]
x = []
for i in range(of_remarks_supp_opp.shape[0]):
    if(of_remarks_supp_opp[i][0] > 0):
        x.append(i)
of_remarks_supp_opp = of_remarks_supp_opp[x]
types = [np.dtype('U50'),'i8',np.dtype('U50'),np.dtype('U50')]
of_test = np.genfromtxt('test.csv', delimiter=',', dtype=types,comments = '@')
x = np.arange(1,of_test.shape[0])
of_test = of_test[x]
x = []
for i in range(of_test.shape[0]):
    if(of_test[i][1] >= 0):
        x.append(i)
of_test = of_test[x]

list_of_companies = []
for i in of_train:
    if(not(i[2] in list_of_companies)):
        list_of_companies.append(i[2])



dict_of_comp_employees = {i[1]: [] for i in of_ratings}
for i in of_ratings:
    if( not ((i[1]+str(i[0])) in dict_of_comp_employees[i[1]])):
        dict_of_comp_employees[i[1]].append((i[1]+str(i[0])))

dict_of_ratings = {(i[1]+str(i[0])): [] for i in of_ratings}
for i in of_ratings:
    l = [get_date(i[2])]
    l.append(i[3]/4)
    dict_of_ratings[(i[1]+str(i[0]))].append(l)

dict_of_remarks = {(i[1]+str(i[0])): [] for i in of_remarks}
dict_of_remarkids = {i[2]:[i[1]+str(i[0]),get_date(i[4]),len(i[3])] for i in of_remarks}
for i in of_remarks:
    l = [get_date(i[4])]
    l.append(i[2])
    l.append(len(i[3]))
    if i[3] == "#NAME?":
        l.append(0)
    else:
        l.append(len(i[3]))
    if l in dict_of_remarks[(i[1]+str(i[0]))]:
        a=1
    else:
        dict_of_remarks[(i[1]+str(i[0]))].append(l)

dict_of_remarks_supp_opp = {i[4]: [[],[]] for i in of_remarks_supp_opp}
for i in of_remarks_supp_opp:
    string = i[1]+str(i[0])
    if(i[2] == 'True'):
        dict_of_remarks_supp_opp[i[4]][0].append(string)
    else:
        dict_of_remarks_supp_opp[i[4]][1].append(string)

dict_of_train = {i[2]+str(i[1]): i[4] for i in of_train}

dict_of_remarks_supp_opp_id = {i[1]+str(i[0]): [[],[]] for i in of_remarks_supp_opp}

for i in of_remarks_supp_opp:
    string = i[4]
    if(i[2] == 'True'):
        dict_of_remarks_supp_opp_id[i[1]+str(i[0])][0].append(string)
    else:
        dict_of_remarks_supp_opp_id[i[1]+str(i[0])][1].append(string)



dict_of_class_with_data = {}
dict_of_ids = {}
dict_of_ids2 = {}
id2_list = []
dict_of_company_no = {}
for i in range(len(list_of_companies)):
    dict_of_company_no[list_of_companies[i]] = i
for i in range(1,5):
    dict_of_class_with_data[i] = [[],[],[],[],[],[],[]]
ci=0
for i in of_train:
    ci+=1
    identifier = i[2]+str(i[1])
    class_val = return_class(identifier)
    x = np.random.choice([1,2,3,4,0],1)
    if(ci%5!=100):
        dict_of_class_with_data[class_val][0].append(identifier)
        dict_of_class_with_data[class_val][1].append(int(2*i[4]-1))
    else:
        dict_of_class_with_data[class_val][4].append(identifier)
        dict_of_class_with_data[class_val][5].append(int(2*i[4]-1))
        dict_of_class_with_data[class_val][6].append(int(i[0]))
        dict_of_ids2[int(i[0])] = [0,0]
        id2_list.append(int(i[0]))
for i in of_test:
    identifier = i[2]+str(i[1])
    class_val = return_class(identifier)
    if(not(identifier in dict_of_train)):
        dict_of_class_with_data[class_val][2].append(identifier)
        dict_of_class_with_data[class_val][3].append(int(i[0]))
        dict_of_ids[int(i[0])] = 0
    else:
        dict_of_ids[int(i[0])] = dict_of_train[identifier]
        
for i in [1,2,3,4]:
    solve_for_class(i)


num_zeros()

save_prediction('Chaos_cs17b004_cs17b031')
