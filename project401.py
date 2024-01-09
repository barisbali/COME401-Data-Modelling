import pandas as pd
from sklearn.preprocessing import LabelEncoder

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
