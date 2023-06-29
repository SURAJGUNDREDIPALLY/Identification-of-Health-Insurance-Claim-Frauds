from django.db.models import Count
from django.db.models import Q
from django.shortcuts import render, redirect, get_object_or_404

import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import VotingClassifier
# Create your views here.
from Remote_User.models import ClientRegister_Model,insurance_claim_frauds,detection_ratio,detection_accuracy

def login(request):


    if request.method == "POST" and 'submit1' in request.POST:

        username = request.POST.get('username')
        password = request.POST.get('password')
        try:
            enter = ClientRegister_Model.objects.get(username=username,password=password)
            request.session["userid"] = enter.id

            return redirect('ViewYourProfile')
        except:
            pass

    return render(request,'RUser/login.html')

def index(request):
    return render(request, 'RUser/index.html')

def Add_DataSet_Details(request):

    return render(request, 'RUser/Add_DataSet_Details.html', {"excel_data": ''})


def Register1(request):

    if request.method == "POST":
        username = request.POST.get('username')
        email = request.POST.get('email')
        password = request.POST.get('password')
        phoneno = request.POST.get('phoneno')
        country = request.POST.get('country')
        state = request.POST.get('state')
        city = request.POST.get('city')
        address = request.POST.get('address')
        gender = request.POST.get('gender')
        ClientRegister_Model.objects.create(username=username, email=email, password=password, phoneno=phoneno,
                                            country=country, state=state, city=city,address=address,gender=gender)

        obj = "Registered Successfully"
        return render(request, 'RUser/Register1.html',{'object':obj})
    else:
        return render(request,'RUser/Register1.html')

def ViewYourProfile(request):
    userid = request.session['userid']
    obj = ClientRegister_Model.objects.get(id= userid)
    return render(request,'RUser/ViewYourProfile.html',{'object':obj})


def Predict_Health_Insurance_Claim_Fraud(request):
    if request.method == "POST":

        if request.method == "POST":

            RID= request.POST.get('RID')
            Sum_Insured= request.POST.get('Sum_Insured')
            age= request.POST.get('age')
            sex= request.POST.get('sex')
            weight= request.POST.get('weight')
            bmi= request.POST.get('bmi')
            hereditary_diseases= request.POST.get('hereditary_diseases')
            no_of_dependents= request.POST.get('no_of_dependents')
            smoker= request.POST.get('smoker')
            city= request.POST.get('city')
            bloodpressure= request.POST.get('bloodpressure')
            diabetes= request.POST.get('diabetes')
            regular_ex= request.POST.get('regular_ex')
            job_title= request.POST.get('job_title')
            claim= request.POST.get('claim')


        df = pd.read_csv('Healthcare_Insurance.csv')

        def apply_response(Label):
            if (Label == 0):
                return 0  # Bad
            elif (Label == 1):
                return 1  # Average

        df['Results'] = df['Label'].apply(apply_response)

        cv = CountVectorizer()
        X = df['RID'].apply(str)
        y = df['Results']

        print("RID")
        print(X)
        print("Results")
        print(y)

        X = cv.fit_transform(X)

        models = []
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)
        X_train.shape, X_test.shape, y_train.shape


        # SVM Model
        print("SVM")
        from sklearn import svm

        lin_clf = svm.LinearSVC()
        lin_clf.fit(X_train, y_train)
        predict_svm = lin_clf.predict(X_test)
        svm_acc = accuracy_score(y_test, predict_svm) * 100
        print("ACCURACY")
        print(svm_acc)
        print("CLASSIFICATION REPORT")
        print(classification_report(y_test, predict_svm))
        print("CONFUSION MATRIX")
        print(confusion_matrix(y_test, predict_svm))
        models.append(('svm', lin_clf))

        print("Logistic Regression")

        from sklearn.linear_model import LogisticRegression

        reg = LogisticRegression(random_state=0, solver='lbfgs').fit(X_train, y_train)
        y_pred = reg.predict(X_test)
        print("ACCURACY")
        print(accuracy_score(y_test, y_pred) * 100)
        print("CLASSIFICATION REPORT")
        print(classification_report(y_test, y_pred))
        print("CONFUSION MATRIX")
        print(confusion_matrix(y_test, y_pred))
        models.append(('logistic', reg))

        print("Decision Tree Classifier")
        dtc = DecisionTreeClassifier()
        dtc.fit(X_train, y_train)
        dtcpredict = dtc.predict(X_test)
        print("ACCURACY")
        print(accuracy_score(y_test, dtcpredict) * 100)
        print("CLASSIFICATION REPORT")
        print(classification_report(y_test, dtcpredict))
        print("CONFUSION MATRIX")
        print(confusion_matrix(y_test, dtcpredict))
        models.append(('DecisionTreeClassifier', dtc))

        classifier = VotingClassifier(models)
        classifier.fit(X_train, y_train)
        y_pred = classifier.predict(X_test)

        RID1 = [RID]
        vector1 = cv.transform(RID1).toarray()
        predict_text = classifier.predict(vector1)

        pred = str(predict_text).replace("[", "")
        pred1 = pred.replace("]", "")

        prediction = int(pred1)

        if (prediction == 0):
            val = 'No Fraud Detection'
        elif (prediction == 1):
            val = 'Fraud Detection'


        print(val)
        print(pred1)

        insurance_claim_frauds.objects.create(
        RID=RID,
        Sum_Insured=Sum_Insured,
        age=age,
        sex=sex,
        weight=weight,
        bmi=bmi,
        hereditary_diseases=hereditary_diseases,
        no_of_dependents=no_of_dependents,
        smoker=smoker,
        city=city,
        bloodpressure=bloodpressure,
        diabetes=diabetes,
        regular_ex=regular_ex,
        job_title=job_title,
        claim=claim,
        Prediction=val)

        return render(request, 'RUser/Predict_Health_Insurance_Claim_Fraud.html',{'objs': val})
    return render(request, 'RUser/Predict_Health_Insurance_Claim_Fraud.html')



