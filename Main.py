import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler
from datetime import datetime, timedelta
from RouteOptimization import load_data,create_graph,find_optimized_path, plot
import ast
import warnings
warnings.filterwarnings("ignore")
fl_df = pd.read_csv('Flight_Database.csv') 
weather_data = pd.read_csv('M1_final.csv')
cities_weather = pd.read_csv('weather_data_cities.csv')
cities_time = pd.read_csv('Cities_FlightDuration_Mins.csv')

# List of possible weather condition labels and their assigned numerical values
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




def safetyCalculator(city, time):

    time_obj = datetime.strptime(time, "%H:%M") #rounds off to the nearest hour
    minute = time_obj.minute
    if minute >= 30:
        time_obj += timedelta(hours=1)
    time_obj = time_obj.replace(minute=0, second=0)
    time =  time_obj.strftime("%H:%M")

    string_stats = cities_weather[cities_weather['City'] == city][time].values[0] #converts string dict to dictionary
    string_stats = string_stats.replace("'", "").replace("{", "").replace("}", "")
    key_value_pairs = string_stats.split(", ")
    input_stats = {}
    for pair in key_value_pairs:
        key, value = pair.split(": ")
        input_stats[key] = float(value) 

    # Convert user inputs to a NumPy array
    input_array = np.array([input_stats[feature] for feature in features]).reshape(1, -1)
    prediction = model.predict(input_array)

    # Print the predicted safety level
    
    return prediction[0]

def unsafeCities(DEP_City, ARR_City, DEP_Time, G):
    unavailable_nodes = []
    safe_nodes=[]
    for node in G.nodes():
        if node == DEP_City:
            if (safetyCalculator(DEP_City,DEP_Time) >= 5 ):
                print('Delay takeoff')
        elif node == ARR_City:
            continue
        else:
            time_from_dep_2_node = cities_time[cities_time['City']==DEP_City][node].values[0]
            time_of_day = (datetime.strptime(DEP_Time, "%H:%M") + timedelta(minutes=int(time_from_dep_2_node))).strftime("%H:%M")
            if (safetyCalculator(node,time_of_day) >= 6 ):
                print(node,"at",time_of_day,":",safetyCalculator(node,time_of_day))
                unavailable_nodes.append(node) 
            else:   
                print(node,"at",time_of_day,":",safetyCalculator(node,time_of_day))
                safe_nodes.append(node) 

                
    print("Unsafe cities from",DEP_City,"to",ARR_City,":",unavailable_nodes)
    return unavailable_nodes
   
if __name__=="__main__": 
    filename = 'Cities_FlightDuration_Mins.csv' #Creating graph
    nodes, graph_data = load_data(filename)
    G = create_graph(nodes, graph_data)

    flightID = input("Enter flight: ")

    DEP_City = fl_df[fl_df['FlightID'] == flightID]['DEP_City'].values[0] #Grabbing values
    DEP_Time = fl_df[fl_df['FlightID'] == flightID]['Dep_Time'].values[0]
    ARR_City = fl_df[fl_df['FlightID'] == flightID]['ARR_City'].values[0]

    unavailable_nodes = unsafeCities(DEP_City, ARR_City, DEP_Time, G);  # Example of unavailable nodes

    primary_path, primary_time, alternate_path, alternate_time = find_optimized_path(G, DEP_City, ARR_City, unavailable_nodes)

    print("Primary path:", primary_path, "(Time:", primary_time, ")")
    # plot(G,primary_path)
    if alternate_path:
        print("Alternate path:", alternate_path, "(Time:", alternate_time, ")")
        plot(G,alternate_path)
        
    flightDuration = 100
    ARR_Time= (datetime.strptime(DEP_Time, "%H:%M") + timedelta(minutes=flightDuration)).strftime("%H:%M")  #Adds hh:mm format and minutes
    print("DEPARTURE: Safety level in",DEP_City,"at",DEP_Time,":",safetyCalculator(DEP_City, DEP_Time))
    print("ARRIVAL: Safety level in",ARR_City,"at",ARR_Time,":",safetyCalculator(ARR_City, ARR_Time))
    print("-------------------------------------------------------------")
    warnings.filterwarnings("ignore")