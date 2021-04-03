import pandas as pd
import sklearn
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from PIL import Image
import streamlit as st

maxUploadSize = 200

# Create a title and SubTitle
st.write("""
# Diabetes Detector
Detect if someone has diabetes using machine learning and python!\n
""")
# Open and display an image
image = Image.open(r'data/diabete.jpg').convert('RGB')
st.image(image, caption='Machine Learning', use_column_width=True)

# Get the data
data = pd.read_csv(r'data/diabetes.csv')
#
classifier_name = st.sidebar.selectbox('Select the classifier',
                                       ('Knn','Random Forest'))

# Set a subheader
st.subheader('Data information:')
# Show the data as a table
st.dataframe(data)
# Show statistics
st.write(data.describe())
# Show the data as a chart
chart = st.bar_chart(data)

# Split the data into independant (X) and dependant (Y)

X = data.iloc[:, 0:8].values
Y = data.iloc[:, -1].values

# Split into train and test 75% training and 25% testing ( test size = 25%)
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.25, random_state=0)


# Get the feature input from the user
def get_user_input():
    pregnancies = st.sidebar.slider("pregnancies", 0, 17, 3)
    glucose = st.sidebar.slider("glucose", 0, 199, 117)
    blood_pressure = st.sidebar.slider("blood_pressure", 0, 122, 72)
    skin_thickness = st.sidebar.slider("skin_thickness", 0, 99, 23)
    insulin = st.sidebar.slider("insulin", 0.0, 846.0, 30.5)
    BMI = st.sidebar.slider("BMI", 0.0, 67.1, 32.0)
    DPF = st.sidebar.slider("DPF", 0.078, 2.42, 0.3725)
    age = st.sidebar.slider("age", 21, 81, 29)

    # Store a dictionnary into a variable
    user_data = {'pregnancies': pregnancies,
                 'glucose': glucose,
                 'blood_pressure': blood_pressure,
                 'skin_thickness': skin_thickness,
                 'insulin': insulin,
                 'BMI': BMI,
                 'DPF': DPF,
                 'age': age
                 }
    #Transform the data into dataframe
    features = pd.DataFrame(user_data,index=[0])
    return features


# Store the users input into a variable
user_input = get_user_input()

# Set a subeader and diplay the use input
st.subheader('User input:')
st.write(user_input)

# Create and train the model
if classifier_name == "Random Forest":
    random_forest = RandomForestClassifier(n_estimators=10, criterion='entropy', random_state=0)
    random_forest.fit(x_train, y_train)

    # Show the models metrics
    st.subheader('Random Forest Model Test Accuracy score :')
    st.write(str(accuracy_score(y_test, random_forest.predict(x_test)) * 100), "%")

    # Store the model prediction in a variable
    prediction = random_forest.predict(user_input)

else:
    knn = KNeighborsClassifier(n_neighbors=7)
    knn.fit(x_train, y_train)

    st.subheader('KNN Model Test accuracy :')
    st.write(str(accuracy_score(y_test, knn.predict(x_test)) * 100), "%")
    prediction = knn.predict(user_input)

# Set a subheader and display the classification
st.subheader('Classification :')
st.write(prediction)

#  pour compiler streamlit run [nom du fichier python].py


