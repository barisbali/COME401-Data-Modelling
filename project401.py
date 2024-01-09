import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.impute import SimpleImputer


#Reading the data with pandas
titanic_df = pd.read_csv('Titanic-Dataset.csv')
#displaying dataset read by pandas
print(titanic_df.head())

label_encoder=LabelEncoder()

#Using label encoder to transform this categorical values to numerical values
titanic_df['Sex'] = label_encoder.fit_transform(titanic_df['Sex'])
titanic_df['Embarked'] = titanic_df['Embarked'].fillna('Unknown') #for filling missing values
titanic_df['Embarked'] = label_encoder.fit_transform(titanic_df['Embarked'])

#to display with numerical values
print(titanic_df.head())

#normalization with standar scaler module (all numeric values)
numeric_columns = ['Age', 'Fare', 'Sex', 'Embarked']
scaler =StandardScaler()
titanic_df[numeric_columns] = scaler.fit_transform(titanic_df[numeric_columns])
#Normalization df
print(titanic_df.head())

#KNN three time (3,7,10)
X = titanic_df[['Age', 'Fare', 'Sex', 'Embarked']] 
y = titanic_df['Survived'] 

# Adding a simple imputer because of NaN values in the dataset
imputer = SimpleImputer(strategy='mean') 
X_imputed = imputer.fit_transform(X)

#train test splits for data
X_train, X_test, y_train, y_test = train_test_split(X_imputed, y, test_size=0.2, random_state=42)
for k in [3, 7, 11]:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)
    
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    
    print(f"KNN (K={k})")
    print("Accuracy:", accuracy)
    print("Classification Report:\n", report)

#MLP
    configurations = [
    (32,),  # Single hidden layer with 32 neurons
    (32, 32),  # Two hidden layers with 32 neurons each
    (32, 32, 32),  # Three hidden layers with 32 neurons each
]

for config in configurations:
    mlp = MLPClassifier(hidden_layer_sizes=config)
    mlp.fit(X_train, y_train)
    y_pred = mlp.predict(X_test)
    
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    
    print(f"MLP (Config={config})")
    print("Accuracy:", accuracy)
    print("Classification Report:\n", report)

#NB
nb = GaussianNB()
nb.fit(X_train, y_train)
y_pred = nb.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

print("Naïve Bayes")
print("Accuracy:", accuracy)
print("Classification Report:\n", report)



