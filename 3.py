import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tkinter as tk
from tkinter import ttk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

train_df = pd.read_csv("D:\\Study\\Intership\\Task 2\\titanic\\train.csv")
test_df = pd.read_csv("D:\\Study\\Intership\\Task 2\\titanic\\test.csv")
gender_submission_df = pd.read_csv("D:\\Study\\Intership\\Task 2\\titanic\\gender_submission.csv")

missing_train = train_df.isnull().sum()
missing_test = test_df.isnull().sum()

train_df['Age'] = train_df['Age'].fillna(train_df['Age'].median())
test_df['Age'] = test_df['Age'].fillna(test_df['Age'].median())
train_df['Embarked'] = train_df['Embarked'].fillna(train_df['Embarked'].mode()[0])
test_df['Fare'] = test_df['Fare'].fillna(test_df['Fare'].median())
train_df = train_df.drop('Cabin', axis=1)
test_df = test_df.drop('Cabin', axis=1)

print("\nMissing values in train dataset after cleaning:\n", train_df.isnull().sum())
print("\nMissing values in test dataset after cleaning:\n", test_df.isnull().sum())

plot_options = [
    'Survival Count',
    'Survival Rate by Gender',
    'Survival Rate by Passenger Class',
    'Survival Rate by Port of Embarkation',
    'Age Distribution',
    'Fare Distribution',
    'Correlation Matrix'
]

def plot_visualization(index):
    fig, ax = plt.subplots()
    plot_type = plot_options[index]
    
    if plot_type == 'Survival Count':
        sns.countplot(x='Survived', data=train_df, ax=ax)
        ax.set_title('Survival Count')
    elif plot_type == 'Survival Rate by Gender':
        sns.barplot(x='Sex', y='Survived', data=train_df, ax=ax)
        ax.set_title('Survival Rate by Gender')
    elif plot_type == 'Survival Rate by Passenger Class':
        sns.barplot(x='Pclass', y='Survived', data=train_df, ax=ax)
        ax.set_title('Survival Rate by Passenger Class')
    elif plot_type == 'Survival Rate by Port of Embarkation':
        sns.barplot(x='Embarked', y='Survived', data=train_df, ax=ax)
        ax.set_title('Survival Rate by Port of Embarkation')
    elif plot_type == 'Age Distribution':
        sns.histplot(train_df['Age'], kde=True, ax=ax)
        ax.set_title('Age Distribution')
    elif plot_type == 'Fare Distribution':
        sns.histplot(train_df['Fare'], kde=True, ax=ax)
        ax.set_title('Fare Distribution')
    elif plot_type == 'Correlation Matrix':
        numeric_cols = train_df.select_dtypes(include=['number']).columns
        corr_matrix = train_df[numeric_cols].corr()
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', ax=ax)
        ax.set_title('Correlation Matrix')
    
    for widget in plot_frame.winfo_children():
        widget.destroy()
    
    canvas = FigureCanvasTkAgg(fig, master=plot_frame)
    canvas.draw()
    canvas.get_tk_widget().pack()

window = tk.Tk()
window.title('Titanic Data Visualization')

plot_frame = tk.Frame(window)
plot_frame.pack()

current_plot_index = 0

def show_next_plot():
    global current_plot_index
    current_plot_index = (current_plot_index + 1) % len(plot_options)
    plot_visualization(current_plot_index)

def show_previous_plot():
    global current_plot_index
    current_plot_index = (current_plot_index - 1) % len(plot_options)
    plot_visualization(current_plot_index)


plot_visualization(current_plot_index)

prev_button = ttk.Button(window, text='Previous', command=show_previous_plot)
prev_button.pack(side='left', padx=10, pady=10)

next_button = ttk.Button(window, text='Next', command=show_next_plot)
next_button.pack(side='right', padx=10, pady=10)

window.mainloop()
