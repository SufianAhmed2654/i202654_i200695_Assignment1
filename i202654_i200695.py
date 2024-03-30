import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.datasets import load_iris

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

iris = load_iris()
iris_df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
iris_df['species'] = iris.target
iris_df['species'] = iris_df['species'].replace(0, 'setosa')
iris_df['species'] = iris_df['species'].replace(1, 'versicolor')
iris_df['species'] = iris_df['species'].replace(2, 'virginica')

X = iris_df.drop('species', axis=1)
y = iris_df['species']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

model = LogisticRegression(multi_class='multinomial', solver='lbfgs')
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

plt.figure()
sns.scatterplot(x=y_test, y=y_pred)
plt.xlabel('True Species')
plt.ylabel('Predicted Species')
plt.title('Logistic Regression Predictions')
plt.show()

print('Model Coefficients:')
print(model.coef_)
print('Model Intercept:')
print(model.intercept_)

print('Model Score:')
print(model.score(X_test, y_test))

print('Model Predictions:')
print(y_pred)

print('True Values:')
print(y_test)

print('Iris Dataset:')
print(iris_df)


import pickle
pickle.dump(model, open('model.pkl', 'wb'))
print('Model saved as model.pkl')