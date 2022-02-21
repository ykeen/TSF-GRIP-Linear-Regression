import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

# Read data from CSV file
data = pd.read_csv('student_scores - student_scores.csv')
X = data.iloc[:,:-1].values
Y = data.iloc[:,1].values
# print(X)

# Split data into train and test
x_train, x_test, y_train, y_test = train_test_split(X,Y,test_size=0.2,random_state=0)

# Build Model
regModel = LinearRegression()
regModel.fit(x_train, y_train)

# Evaluate Model
predictedData = regModel.predict(x_test)
# print(predictedData)
Accuracy = mean_absolute_error(y_test,predictedData)
print("The Accuracy of the model = ",Accuracy)

# Predict Score if a student studies 9.25 hrs/day
hrs= [9.25]
predicted_hrs = regModel.predict([hrs])
print("the Predicted Score if a student studies 9.25 hrs/day = ",predicted_hrs)

