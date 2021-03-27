
# Naive Bayes classifier example
# Play Tennis

import pandas as pd
from sklearn.naive_bayes import CategoricalNB
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score

le = LabelEncoder()
df = pd.read_csv('play_tennis.csv', index_col=0)
print("data :\n", df)
df = df.apply(le.fit_transform)
print("converted \n", df)
X = df[['Outlook', 'Temperature', 'Humidity',  'Windy']].values
Y = df[['Play']].values
X_train = X[:10]
y_train = Y[:10].ravel()
X_test = X[10:]
y_test = Y[10:].ravel()
# fit model
model = CategoricalNB()
model.fit(X_train, y_train)
# print model
print("\n------\nmodel counts:")
print("class count \n", model.class_count_)
print("sample count \n", model.category_count_)
y_hat_test = model.predict(X_test)
y_hat_p=model.predict_proba(X_test)
print("\n------\ntest data output:")
print("predict probility",y_hat_p)
output = le.inverse_transform(y_hat_test)
print("predict result:", output)
accu = accuracy_score(y_test, y_hat_test)
print("accu:", accu)
