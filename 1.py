import pandas as pd

train_df = pd.read_csv("D:\\Study\\Intership\\Task 2\\titanic\\train.csv")
test_df = pd.read_csv("D:\\Study\\Intership\\Task 2\\titanic\\test.csv")
gender_submission_df = pd.read_csv("D:\\Study\\Intership\\Task 2\\titanic\\gender_submission.csv")

missing_train = train_df.isnull().sum()
missing_test = test_df.isnull().sum()

print("Missing values in train dataset:\n", missing_train)
print("\nMissing values in test dataset:\n", missing_test)

train_df['Age'].fillna(train_df['Age'].median(), inplace=True)
test_df['Age'].fillna(test_df['Age'].median(), inplace=True)

train_df['Embarked'].fillna(train_df['Embarked'].mode()[0], inplace=True)

test_df['Fare'].fillna(test_df['Fare'].median(), inplace=True)

train_df.drop('Cabin', axis=1, inplace=True)
test_df.drop('Cabin', axis=1, inplace=True)

print("\nMissing values in train dataset after cleaning:\n", train_df.isnull().sum())
print("\nMissing values in test dataset after cleaning:\n", test_df.isnull().sum())

import seaborn as sns
import matplotlib.pyplot as plt

sns.countplot(x='Survived', data=train_df)
plt.title('Survival Count')
plt.show()

sns.barplot(x='Sex', y='Survived', data=train_df)
plt.title('Survival Rate by Gender')
plt.show()

sns.barplot(x='Pclass', y='Survived', data=train_df)
plt.title('Survival Rate by Passenger Class')
plt.show()

sns.barplot(x='Embarked', y='Survived', data=train_df)
plt.title('Survival Rate by Port of Embarkation')
plt.show()

sns.histplot(train_df['Age'], kde=True)
plt.title('Age Distribution')
plt.show()

sns.histplot(train_df['Fare'], kde=True)
plt.title('Fare Distribution')
plt.show()

corr_matrix = train_df.corr()
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
plt.title('Correlation Matrix')
plt.show()
