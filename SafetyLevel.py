import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler
import pickle

weather_data = pd.read_csv('M1_final.csv')


label_mapping = {
    ' Fair / Windy ': 3, ' Fair ': 1, ' Light Rain / Windy ': 7, ' Partly Cloudy ': 2,
    ' Mostly Cloudy ': 2, ' Cloudy ': 5, ' Light Rain ': 6, ' Mostly Cloudy / Windy ': 8,
    ' Partly Cloudy / Windy ': 5, ' Light Snow / Windy ': 4, ' Cloudy / Windy ': 5,
    ' Light Drizzle ': 5, ' Rain ': 6, ' Heavy Rain ': 9, ' Fog ': 8, ' Wintry Mix ': 4,
    ' Light Freezing Rain ': 8, ' Light Snow ': 3, ' Wintry Mix / Windy ': 4,
    ' Fog / Windy ': 8, ' Light Drizzle / Windy ': 6, ' Rain / Windy ': 7,
    ' Drizzle and Fog ': 9, ' Snow ': 3, ' Heavy Rain / Windy ': 10
}

# Map numerical labels to the 'Condition' column
weather_data['SafetyLevel'] = weather_data[' Condition '].map(label_mapping)

# Display the updated DataFrame with numerical labels
print(weather_data[[' Condition ', 'SafetyLevel']].head())

# Select features and target variable
features = ['Temperature', 'Humidity', 'Wind Speed', 'Pressure',]
X = weather_data[features]
y = weather_data['SafetyLevel']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# scaler = StandardScaler()
# X_train_scaled = scaler.fit_transform(X_train)
# X_test_scaled = scaler.transform(X_test)

model = MLPClassifier(random_state=42)
model.fit(X_train, y_train)
predictions = model.predict(X_test)

accuracy = accuracy_score(y_test, predictions)
classification_rep = classification_report(y_test, predictions)


# print(f'Accuracy: {accuracy:.2f}')
# print('Classification Report:')
# print(classification_rep)
# print('\n' + '-'*50 + '\n')
user_inputs = {
    'Temperature': 27,
    'Humidity': 41,
    'Wind Speed': 17,
    'Pressure' : 30.15
}

# Convert user inputs to a NumPy array
input_array = np.array([user_inputs[feature] for feature in features]).reshape(1, -1)

# Make a prediction using the trained model
prediction = model.predict(input_array)

# Print the predicted safety level
print(f'Predicted Safety Level: {prediction[0]}')
