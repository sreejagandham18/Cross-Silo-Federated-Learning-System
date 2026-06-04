from django.shortcuts import render
from django.template import RequestContext
from django.contrib import messages
import pymysql
from django.http import HttpResponse
from django.core.files.storage import FileSystemStorage
import os
import random
import io
import base64
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split 
from sklearn import svm
import pandas as pd
from sklearn.metrics import f1_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
import numpy as np
from sklearn.metrics import confusion_matrix

from nltk.corpus import stopwords
import nltk
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.feature_extraction.text import TfidfVectorizer #loading tfidf vector
from sklearn.ensemble import VotingClassifier
import pickle
import seaborn as sns

global username, X_train, X_test, y_train, y_test, X, Y, ensemble_classifier, img_b64
global accuracy, precision, recall, fscore, vectorizer
stop_words = set(stopwords.words('english'))

def PredictAction(request):
    if request.method == 'POST':
        global ensemble_classifier, vectorizer
        myfile = request.FILES['t1']
        name = request.FILES['t1'].name
        if os.path.exists("VulnerApp/static/testData.csv"):
            os.remove("VulnerApp/static/testData.csv")
        fs = FileSystemStorage()
        filename = fs.save('VulnerApp/static/testData.csv', myfile)
        df = pd.read_csv('VulnerApp/static/testData.csv')
        temp = df.values
        X = vectorizer.transform(df['Test_data'].astype('U')).toarray()
        predict = ensemble_classifier.predict(X)
        output = '<table border="1" align="center" width="100%" ><tr><th><font size="" color="black">Test Data</th>'
        output += '<th><font size="" color="black">Predicted Vulnerability</th></tr>'
        for i in range(len(predict)):
            if predict[i] == 0:
                status = "No Vulnerability"
            if predict[i] == 1:
                status = "SQL Injection"
            if predict[i] == 2:
                status = "Cross Site Scripting/RFI"
            out = str(temp[i,0])
            out = out.replace("<","")
            out = out.replace(">","")
            output+='<tr><td><font size="" color="black">'+out+'</td>'
            output+='<td><font size="" color="black">'+status+'</td></tr>'
        output+="</table><br/><br/><br/><br/><br/><br/>"
        context= {'data':output}
        return render(request, 'UserScreen.html', context)        

def UploadAction(request):
    if request.method == 'POST':
        global X_train, X_test, y_train, y_test, X, Y, vectorizer
        myfile = request.FILES['t1']
        name = request.FILES['t1'].name
        if os.path.exists("VulnerApp/static/Data.csv"):
            os.remove("VulnerApp/static/Data.csv")
        fs = FileSystemStorage()
        filename = fs.save('VulnerApp/static/Data.csv', myfile)
        df = pd.read_csv('VulnerApp/static/Data.csv')
        df['Label'] = df['Label'].astype(str).astype(int)
        vectorizer = TfidfVectorizer(stop_words=stop_words, use_idf=True, smooth_idf=False, norm=None, decode_error='replace', max_features=300)
        X = vectorizer.fit_transform(df['Sentence'].astype('U')).toarray()
        temp = pd.DataFrame(X, columns=vectorizer.get_feature_names())
        Y = df['Label'].ravel()
        indices = np.arange(X.shape[0])
        np.random.shuffle(indices)
        X = X[indices]
        Y = Y[indices]
        X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.1)
        output = "Dataset Loading & Processing Completed<br/>"
        output += "Dataset Length : "+str(len(X))+"<br/>"
        output += "Splitted Training Length : "+str(len(X_train))+"<br/>"
        output += "Splitted Test Length : "+str(len(X_test))+"<br/>"
        output += "<br/><br/>Features Vector<br/><br/>"+str(temp)+"<br/><br/>"
        context= {'data': output}
        return render(request, 'Upload.html', context)

def calculateMetrics(algorithm, predict, y_test):
    global accuracy, precision, recall, fscore, img_b64
    a = accuracy_score(y_test,predict)*100
    p = precision_score(y_test, predict,average='macro') * 100
    r = recall_score(y_test, predict,average='macro') * 100
    f = f1_score(y_test, predict,average='macro') * 100
    accuracy.append(a)
    precision.append(p)
    recall.append(r)
    fscore.append(f)
    labels = ['No Vulnerability', 'SQL Injection', 'Cross Site Scripting/RFI']
    conf_matrix = confusion_matrix(y_test, predict) 
    plt.figure(figsize =(6, 3)) 
    ax = sns.heatmap(conf_matrix, xticklabels = labels, yticklabels = labels, annot = True, cmap="viridis" ,fmt ="g");
    ax.set_ylim([0,len(labels)])
    plt.title(algorithm+" Confusion matrix") 
    plt.ylabel('True class') 
    plt.xlabel('Predicted class')
    #plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight')
    plt.close()
    img_b64 = base64.b64encode(buf.getvalue()).decode()    

def RunEnsemble(request):
    if request.method == 'GET':
        global accuracy, precision, recall, fscore, X_train, X_test, y_train, y_test, ensemble_classifier
        accuracy = []
        precision = []
        recall = []
        fscore = []
        nb_cls = GaussianNB()
        svm_cls = svm.SVC()
        knn_cls = KNeighborsClassifier(n_neighbors=2)
        if os.path.exists("model/ensemble.pckl"):
            f = open('model/ensemble.pckl', 'rb')
            ensemble_classifier = pickle.load(f)
            f.close()    
        else:
            #creating estimator with list of classifiers object such as KNN< SVM and Naive Bayes
            estimators = [('nb', nb_cls), ('svm', svm_cls), ('knn', knn_cls)]
            #training voting classifier as stack classifier by giving stack of 3 different classifiers
            ensemble_classifier = VotingClassifier(estimators = estimators)
            #now trained classifier on training data
            ensemble_classifier.fit(X_train, y_train)
            f = open('model/ensemble.pckl', 'wb')
            pickle.dump(ensemble_classifier, f)
            f.close()
        X_test = X_test[0:2000]
        y_test = y_test[0:2000]
        #performing prediction on test data using ensemble classifier
        predict = ensemble_classifier.predict(X_test)
        calculateMetrics("Ensemble Classifier", predict, y_test)
        algorithms = ['Ensemble Classifier']
        output = '<table border="1" align="center" width="100%" ><tr><th><font size="" color="black">Algorithm Name</th>'
        output += '<th><font size="" color="black">Accuracy</th><th><font size="" color="black">Precision</th>'
        output += '<th><font size="" color="black">Recall</th><th><font size="" color="black">FScore</th></tr>'
        for i in range(len(algorithms)):
            output+='<tr><td><font size="" color="black">'+algorithms[i]+'</td>'
            output+='<td><font size="" color="black">'+str(accuracy[i])+'</td>'
            output+='<td><font size="" color="black">'+str(precision[i])+'</td>'
            output+='<td><font size="" color="black">'+str(recall[i])+'</td>'
            output+='<td><font size="" color="black">'+str(fscore[i])+'</td></tr>'
        output+="</table><br/><br/><br/><br/><br/><br/>"
        context= {'data':output}
        return render(request, 'UserScreen.html', context)    


def Graph(request):
    if request.method == 'GET':
        global img_b64     
        context= {'data': img_b64}
        return render(request, 'ViewGraph.html', context)   

def Predict(request):
    if request.method == 'GET':
        return render(request, 'Predict.html', {})

def Upload(request):
    if request.method == 'GET':
        return render(request, 'Upload.html', {})

def index(request):
    if request.method == 'GET':
        return render(request, 'index.html', {})

def Login(request):
    if request.method == 'GET':
        return render(request, 'Login.html', {})

def Register(request):
    if request.method == 'GET':
        return render(request, 'Register.html', {})

def UserLogin(request):
    if request.method == 'POST':
        global username
        username = request.POST.get('username', False)
        password = request.POST.get('password', False)
        status = 'none'
        con = pymysql.connect(host='127.0.0.1',port = 3306,user = 'root', password = 'root', database = 'VulnerDB',charset='utf8')
        with con:
            cur = con.cursor()
            cur.execute("select * FROM register")
            rows = cur.fetchall()
            for row in rows:
                if row[0] == username and row[1] == password:
                    status = 'success'
                    break
        if status == 'success':
            context= {'data':'Welcome '+username}
            return render(request, 'UserScreen.html', context)            
        else:
            context= {'data':'Invalid login details'}
            return render(request, 'Login.html', context)

def Signup(request):
    if request.method == 'POST':
      username = request.POST.get('username', False)
      password = request.POST.get('password', False)
      contact = request.POST.get('contact', False)
      email = request.POST.get('email', False)
      address = request.POST.get('address', False)
      status = "none"
      con = pymysql.connect(host='127.0.0.1',port = 3306,user = 'root', password = 'root', database = 'VulnerDB',charset='utf8')
      with con:
            cur = con.cursor()
            cur.execute("select username FROM register where username='"+username+"'")
            rows = cur.fetchall()
            if len(rows) > 0:
                status = username+" already exists"
      if status == "none":
          db_connection = pymysql.connect(host='127.0.0.1',port = 3306,user = 'root', password = 'root', database = 'VulnerDB',charset='utf8')
          db_cursor = db_connection.cursor()
          student_sql_query = "INSERT INTO register(username,password,contact,email,address) VALUES('"+username+"','"+password+"','"+contact+"','"+email+"','"+address+"')"
          db_cursor.execute(student_sql_query)
          db_connection.commit()
          print(db_cursor.rowcount, "Record Inserted")
          if db_cursor.rowcount == 1:
               context= {'data':'Signup Process Completed'}
               return render(request, 'Register.html', context)
          else:
               context= {'data':'Error in signup process'}
               return render(request, 'Register.html', context)
      else:
          context= {'data': status}
          return render(request, 'Register.html', context)
