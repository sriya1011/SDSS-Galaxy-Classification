import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, accuracy_score
from sklearn.ensemble import RandomForestClassifier
import warnings
warnings.filterwarnings("ignore")
df = pd.read_csv("sdss_100k_galaxy_form_burst.csv", header=1)
df.head()
df.shape
df.info()
df.isnull().sum()
df['subclass'].replace(['STARFORMING','STARBURST'],[0,1], inplace=True)
df.describe()
sub=df["subclass"].value_counts()
sub
# Replace invalid values -9999 with NaN
df = df.replace([-9999.0, -9999], np.nan)

# Fill missing numeric values with median
numeric_cols = df.select_dtypes(include=['float64','int64']).columns
df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())

# Fill missing categorical values with mode
categorical_cols = df.select_dtypes(include='object').columns
for col in categorical_cols:
    df[col] = df[col].fillna(df[col].mode()[0])

from sklearn.preprocessing import LabelEncoder

label_enc = LabelEncoder()

for col in categorical_cols:
    df[col] = label_enc.fit_transform(df[col])
sub.plot(kind="pie",subplots=True,autopct="%1.2f%%")
def func(col):
    sns.boxplot(x=col,data=df)
    plt.show()
for i in df.columns:
    func(i)
quant=df['u'].quantile(q=[0.75,0.25])
print(quant)
Q3=quant.loc[0.75]
print(Q3)
Q1=quant.loc[0.25]
print(Q1)
IQR=Q3-Q1
print(IQR)
maxwhisker=Q3+1.5*IQR
print(maxwhisker)
miniwhisker=Q1-1.5*IQR
print(miniwhisker)
df['u']=np.where(df['u']>22.054895, 22.054895,df['u'])
df['u']=np.where(df['u']<16.787095, 16.787095,df['u'])
sns.boxplot(y='u',data=df)
x=df.drop(['subclass',], axis=1)
y=df['subclass']
#i want to know top k best columns in the data frame using SelectkBest k = 10
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif
# Assuming x and y are your data and target variables
selector=SelectKBest (score_func=f_classif, k=10) # Select top 10 features
#selector SelectKBest(score_func=chi2, k=10) # For classification tasks with non-negative features
# Fit selector to the data
X_selected=selector.fit_transform(x, y)
#Get the names of the selected features
selected_features=x.columns[selector.get_support()]
# Print the selected features
print("Selected features:", selected_features)
# Assuming your target column is 'subclass' in your DataFrame 'df'
X = df.drop(['subclass', 'class'], axis=1)
y = df ['subclass']
# Initialize SMOTE
smote=SMOTE (random_state=42)
# Perform SMOTE oversampling
X_resampled, y_resampled = smote.fit_resample(x, y)
# Check the new value counts
print(pd.Series (y_resampled).value_counts())
df1=df[['i','z','modelFlux_z','petroRad_g','petroRad_r', 'petroFlux_z', 'petroR50_u', 'petroR50_g', 'petroR50_i', 'petroR50_r', 'subclass']]
from sklearn.model_selection import train_test_split
x =df1[['i','z', 'modelFlux_z', 'petroRad_g', 'petroRad_r', 'petroFlux_z','petroR50_u', 'petroR50_g', 'petroR50_i', 'petroR50_r']]
y = df1["subclass"]
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.20)
from sklearn.preprocessing import StandardScaler
# Create a scaler object
sc = StandardScaler()
# Transform your data
scaled_data = sc.fit_transform(x_train)
from sklearn.tree import DecisionTreeClassifier
clf=DecisionTreeClassifier()
#Train the classifier on the training data
clf.fit(x_train, y_train)
#Make predictions on the testing data
y_pred=clf.predict(x_test)
#Evaluate the classifier
report=classification_report (y_test, y_pred)
print("classification Report:\n", report)

print(accuracy_score(y_pred,y_test))
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, recall_score, precision_score, confusion_matrix, f1_score
lg=LogisticRegression()
log=lg.fit(x_train,y_train)
y_pred=lg.predict(x_test)
print("Confusion Matrix: \n", confusion_matrix(y_test,y_pred))
print("-------------------------------------------------")
print("Classification report:\n", classification_report(y_test, y_pred))


print(accuracy_score(y_pred,y_test))
from sklearn.ensemble import RandomForestClassifier
RF = RandomForestClassifier()
#Train the Random Forest classifier
RF = RandomForestClassifier()
RF.fit(x_train, y_train)
RFtrain =RF.predict(x_train)
RFtest =RF.predict(x_test)
# Print classification report
confusion_matrix
print(confusion_matrix (RFtrain,y_train))
print(confusion_matrix(RFtest,y_test))
print(classification_report (RFtrain,y_train))
print(classification_report(RFtest,y_test))

print(accuracy_score (RFtrain, y_train))
print(accuracy_score (RFtest,y_test))
import pickle
pickle.dump(RF,open("RF.pkl","wb"))
pred1=RF.predict([[16.946170,16.708910, 207.218700, 4.180779, 4.060687,194.731000, 2.141953, 2.149080,2.056686,2.055798]])
print(pred1)

pred2 = RF.predict([[17.675285, 17.52775, 104.25655, 3.397512, 3.424717,
                     90.717547, 1.613005, 1.632243, 1.548225, 1.596137]])
print(pred2)
from flask import Flask, request, render_template
import pickle
import numpy as np
import json
import requests
import pandas as pd
from flask import Flask, request, render_template_string
import pickle
import os

app = Flask(__name__)
model = pickle.load(open("RF.pkl", "rb"))

BASE_DIR = os.getcwd()

@app.route("/")
def home():
    return open(os.path.join(BASE_DIR, "index.html")).read()

@app.route("/predict", methods=["POST"])
def predict():
    features = [
        float(request.form['i']),
        float(request.form['z']),
        float(request.form['modelFlux_z']),
        float(request.form['petroRad_g']),
        float(request.form['petroRad_r']),
        float(request.form['petroFlux_z']),
        float(request.form['petroR50_u']),
        float(request.form['petroR50_g']),
        float(request.form['petroR50_i']),
        float(request.form['petroR50_r'])
    ]

    result = model.predict([features])[0]
    prediction = "starbursting" if result == 1 else "starforming"

    html = open(os.path.join(BASE_DIR, "result.html")).read()
    return render_template_string(html, prediction=prediction)

app.run(port=2222, debug=False, use_reloader=False)