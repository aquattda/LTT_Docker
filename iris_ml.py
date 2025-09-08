from sklearn import datasets
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

iris = datasets.load_iris()
x = iris.data[:, [2,3]]
y = iris.target


x_train, x_test, y_train, y_test  = train_test_split(x,y,test_size=0.2, random_state=1)


scaler = StandardScaler() # ?????
x_train = scaler.fit_transform(x_train)   #Fit vs Fittransfrom ????
x_test = scaler.transform(x_test)  #?????


model = RandomForestClassifier()
model.fit(x_train, y_train)
y_pred = model.predict(x_test)

for pred, label in zip(y_pred, y_test):
    print("Prediction: {}. Label: {}".format(pred, label))
print("Accuracy: %.2f" % accuracy_score(y_test, y_pred))