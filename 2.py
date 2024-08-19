import pandas as pd

train_df = pd.read_csv("D:\\Study\\Intership\\Task 2\\titanic\\train.csv")
test_df = pd.read_csv("D:\\Study\\Intership\\Task 2\\titanic\\test.csv")
gender_submission_df = pd.read_csv("D:\\Study\\Intership\\Task 2\\titanic\\gender_submission.csv")

missing_train = train_df.isnull().sum()
missing_test = test_df.isnull().sum()

print("Missing values in train dataset:\n", missing_train)
print("\nMissing values in test dataset:\n", missing_test)

train_df['Age'] = train_df['Age'].fillna(train_df['Age'].median())
test_df['Age'] = test_df['Age'].fillna(test_df['Age'].median())

train_df['Embarked'] = train_df['Embarked'].fillna(train_df['Embarked'].mode()[0])

test_df['Fare'] = test_df['Fare'].fillna(test_df['Fare'].median())

train_df = train_df.drop('Cabin', axis=1)
test_df = test_df.drop('Cabin', axis=1)

print("\nMissing values in train dataset after cleaning:\n", train_df.isnull().sum())
print("\nMissing values in test dataset after cleaning:\n", test_df.isnull().sum())

import matplotlib
matplotlib.use('Agg')
import seaborn as sns
import matplotlib.pyplot as plt

sns.countplot(x='Survived', data=train_df)
plt.title('Survival Count')
plt.savefig('Survival_Count.png')
plt.close()

sns.barplot(x='Sex', y='Survived', data=train_df)
plt.title('Survival Rate by Gender')
plt.savefig('Survival_Rate_by_Gender.png')
plt.close()

sns.barplot(x='Pclass', y='Survived', data=train_df)
plt.title('Survival Rate by Passenger Class')
plt.savefig('Survival_Rate_by_Passenger_Class.png')
plt.close()

sns.barplot(x='Embarked', y='Survived', data=train_df)
plt.title('Survival Rate by Port of Embarkation')
plt.savefig('Survival_Rate_by_Port_of_Embarkation.png')
plt.close()

sns.histplot(train_df['Age'], kde=True)
plt.title('Age Distribution')
plt.savefig('Age_Distribution.png')
plt.close()

sns.histplot(train_df['Fare'], kde=True)
plt.title('Fare Distribution')
plt.savefig('Fare_Distribution.png')
plt.close()

numeric_cols = train_df.select_dtypes(include=['number']).columns
corr_matrix = train_df[numeric_cols].corr()
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
plt.title('Correlation Matrix')
plt.savefig('Correlation_Matrix.png')
plt.close()
