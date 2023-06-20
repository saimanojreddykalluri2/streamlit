import subprocess

def install(name):
    subprocess.call([sys.executable, '-m', 'pip', 'install', name])
install("streamlit")
install("joblib")
install("numpy")

import streamlit as st
import joblib
import numpy as np
import time
import xgboost
from xgboost import XGBRegressor


from PIL import Image
st.set_option('deprecation.showfileUploaderEncoding', False)
st.set_page_config(
    page_title="Fatigue Strength Prediction",
    page_icon="💡",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Set custom colors
PRIMARY_COLOR = '#336699'  # Blue
SECONDARY_COLOR = '#ff0000'  # Red
TEXT_COLOR = '#000000'  # Black
DARK_BLUE = "#00008B"

# Set researcher background image
# background_image = Image.open('path_to_background_image.png')

# Set CSS style for interface
st.markdown(
    f"""
    <style>
    }}
    .stButton button {{
        background-color: {PRIMARY_COLOR} !important;
        color: {TEXT_COLOR} !important;
    }}
    .stTextInput input, .stTextArea textarea {{
        background-color: {SECONDARY_COLOR} !important;
        color: {TEXT_COLOR} !important;
    }}
    </style>
    """,
    unsafe_allow_html=True
)

# Set title and subtitle
st.title("Fatigue Strength Prediction for Aluminium alloys")
st.markdown("---")

def hometab():
    css = """
    <style>
    .button-color {
        background-color: blue;
        color: white;
        padding: 0.5em 1em;
        border-radius: 0.5em;
        text-align: center;
    }
    </style>
    """

    # Render the CSS styles
    st.markdown(css, unsafe_allow_html=True)

    # st.markdown("SELECT from navbar TAB1 for .. TAb2 for .. TAb3 for .. ")
    st.write("<span style='font-size: 25px;'>Welcome to the Aluminum fatigue strength predictor. This tool is built using experimental data of more than 1000 Aluminum alloys, available on MakeItFrom website. This website consists of composition and mechanical properties information of Aluminum alloys, along with the experimentally measured rotating bending fatigue strength for nearly 10 million cycles. Various data analytics techniques such as feature selection and regression modeling were used to obtain highly accurate fatigue strength prediction models. Based on different inputs subsets three different machine learning models are developed, you can select based on your requirement.</span>", unsafe_allow_html=True)
    st.write("<span style='font-size: 25px;'> **Tab1 (required features) :** composition of different elements, Brinell hardness, ultimate tensile strength , tensile yield strength, temper used and type of the alloy.</span>", unsafe_allow_html=True)
    st.write("<span style='font-size: 25px;'> **Tab2 (required features) :** composition of different elements, tempers used and type of alloy.</span>", unsafe_allow_html=True)
    st.write("<span style='font-size: 25px;'> **Tab3 (required features) :** Brinell hardness, ultimate tensile strength , tensile yield strength.</span>", unsafe_allow_html=True)


    button1 = st.button("Tab1", key=1)
    button2 = st.button("Tab2", key=2)
    button3 = st.button("Tab3", key=3)
    st.write("<span style='font-size: 15px;'> **Note:** <strong><span style='color: red;'> All the fields should be filled for the selected tab</span></strong></span>",
             unsafe_allow_html=True)

    if button1:
        tab1()
    elif button2:
        tab2()
    elif button3:
        tab3()

def tab1():
    # st.write("I am in tab1")
    # Set page configuration

    # Load the trained model
    model = joblib.load(r'C:\Users\kallu\mtp_second_phase\Fatigue_strength_prediction_model.joblib')

    st.subheader("Predict fatigue strength based on features")

    # Feature input
    st.header("Feature Inputs")
    st.markdown("---")

    # Define feature names
    feature_names = [

        'Brinell_Hardness ', 'Ultimate_tensile_strength',
        'Tensile yield strength', 'Type of alloy(Cast or Wrought)', 'Temper used', 'Ag', 'Al', 'B',
        'Be', 'Bi', 'Co', 'Cr', 'Cu', 'Fe', 'Ga', 'Li', 'Mg', 'Mn', 'Ni', 'Pb',
        'Si', 'Sn', 'Ti', 'V', 'Zn', 'Zr', 'res',
        # Add more feature names as needed
    ]
    col1, col2, col3 = st.columns(3)

    # Distribute features among the columns
    num_features = len(feature_names)
    features_per_column = num_features // 3
    remainder = num_features % 3

    feature_values = []

    for i in range(features_per_column + min(1, remainder)):
        if i < features_per_column:
            with col1:
                value = st.number_input(feature_names[i], value=0.0)
                feature_values.append(value)
        if i < features_per_column:
            with col2:
                value = st.number_input(feature_names[i + features_per_column], value=0.0)
                feature_values.append(value)
        if i < features_per_column + min(1, remainder):
            with col3:
                value = st.number_input(feature_names[i + 2 * features_per_column], value=0.0)
                feature_values.append(value)

    # Prediction
    if st.button("Predict 1", key=4):
        # Perform prediction using the loaded model
        feature_values2=[]
        for i in range(3):
            for j in range(0,27,3):
                feature_values2.append(feature_values[i+j])
        print(feature_values2)
        time.sleep(10)
        # feature_values=[70,290,220,1,17,0,95.9260184953762,0,0,0,0,0.149962509372657,0.149962509372657,0.3499125218695330,0,0,2.0994751312172,0.849787553111722,0,0,0.199950012496876,0,0.049987503124219,0,0.149962509372657,0,0.0749812546863284]
        feature_values = np.array(feature_values2).reshape((1,27))
        prediction = model.predict(feature_values)
        st.success(f"Fatigue Strength Prediction: {prediction[0]:.2f} MPa")
        st.markdown("---")

    st.header("Description")

    # Display label with a paragraph
    st.markdown("The 'Temper' feature in the input data is typically a categorical variable. However, in order to utilize machine learning models effectively, it is necessary to convert this categorical variable into numerical labels using a label encoder. When interacting with the user, it is important to provide the following input values based on the temper used: 'F': 0, 'H11': 1, 'H111': 2, 'H112': 3, 'H116': 4, 'H12': 5, 'H13': 6, 'H14': 7, 'H15': 8, 'H16': 9, 'H17': 10, 'H18': 11, 'H19': 12, 'H21': 13, 'H22': 14, 'H24': 15, 'H25': 16, 'H26': 17, 'H27': 18, 'H28': 19, 'H29': 20, 'H32': 21, 'H321': 22, 'H322': 23, 'H34': 24, 'H36': 25, 'H38': 26, 'O': 27, 'T1': 28, 'T11': 29, 'T2': 30, 'T3': 31, 'T31': 32, 'T351': 33, 'T3510': 34, 'T3511': 35, 'T36': 36, 'T361': 37, 'T37': 38, 'T4': 39, 'T42': 40, 'T451': 41, 'T4510': 42, 'T4511': 43, 'T452': 44, 'T5': 45, 'T51': 46, 'T52': 47, 'T53': 48, 'T54': 49, 'T551': 50, 'T5510': 51, 'T5511': 52, 'T6': 53, 'T61': 54, 'T6151': 55, 'T62': 56, 'T64': 57, 'T65': 58, 'T651': 59, 'T6510': 60, 'T6511': 61, 'T652': 62, 'T66': 63, 'T7': 64, 'T71': 65, 'T72': 66, 'T73': 67, 'T7351': 68, 'T73510': 69, 'T73511': 70, 'T7352': 71, 'T74': 72, 'T7451': 73, 'T7452': 74, 'T7454': 75, 'T76': 76, 'T7651': 77, 'T76510': 78, 'T76511': 79, 'T8': 80, 'T81': 81, 'T83': 82, 'T831': 83, 'T832': 84, 'T851': 85, 'T8510': 86, 'T8511': 87, 'T852': 88, 'T861': 89, 'T87': 90, 'T9': 91")
    st.markdown("For type column if it is wrought alloy keep it as 1, if it is cast alloy keep it as 0")
    st.header("Another Description")

    # Display label with a paragraph
    st.markdown("This is a label with a paragraph. You can provide additional details and information here.")
    time.sleep(100)

def tab2():
    # st.write("I am in tab2")
    # Set page configuration


    # Load the trained model
    model = joblib.load(r'C:\Users\kallu\mtp_second_phase\only_composition_model.joblib')

    st.subheader("Predict fatigue strength based on features")

    # Feature input
    st.header("Feature Inputs")
    st.markdown("---")

    # Define feature names
    feature_names = [

        'Type of alloy(Cast or Wrought)', 'Temper used', 'Ag', 'Al', 'B',
        'Be', 'Bi', 'Co', 'Cr', 'Cu', 'Fe', 'Ga', 'Li', 'Mg', 'Mn', 'Ni', 'Pb',
        'Si', 'Sn', 'Ti', 'V', 'Zn', 'Zr', 'res',
        # Add more feature names as needed
    ]
    col1, col2, col3 = st.columns(3)

    # Distribute features among the columns
    num_features = len(feature_names)
    features_per_column = num_features // 3
    remainder = num_features % 3

    feature_values = []

    for i in range(features_per_column + min(1, remainder)):
        if i < features_per_column:
            with col1:
                value = st.number_input(feature_names[i], value=0.0)
                feature_values.append(value)
        if i < features_per_column:
            with col2:
                value = st.number_input(feature_names[i + features_per_column], value=0.0)
                feature_values.append(value)
        if i < features_per_column + min(1, remainder):
            with col3:
                value = st.number_input(feature_names[i + 2 * features_per_column], value=0.0)
                feature_values.append(value)

    # Prediction
    if st.button("Predict 2", key=6):
        # Perform prediction using the loaded model
        feature_values2 = []
        for i in range(3):
            for j in range(0, 24, 3):
                feature_values2.append(feature_values[i + j])
        print(feature_values2)
        time.sleep(10)
        # feature_values=[70,290,220,1,17,0,95.9260184953762,0,0,0,0,0.149962509372657,0.149962509372657,0.3499125218695330,0,0,2.0994751312172,0.849787553111722,0,0,0.199950012496876,0,0.049987503124219,0,0.149962509372657,0,0.0749812546863284]
        feature_values = np.array(feature_values2).reshape((1, 24))
        prediction = model.predict(feature_values)
        st.success(f"Fatigue Strength Prediction: {prediction[0]:.2f} MPa")
        st.markdown("---")

    # Data Details
    # st.header("Data Details")
    # data_description = st.text_area("Enter details about the data", height=150, disabled=True, value="", style="background-color: white; color: black",)
    # # st.write(data_description)
    # # Display Data Details
    # # st.subheader("Data Description")
    # st.header("Data Description")
    # data_description2 = st.text_area("Enter details about the data", height=150, disabled=True, value="mera data", style="background-color: white; color: black",)
    # # st.write(data_description2)

    st.header("Description")

    # Display label with a paragraph
    st.markdown(
        "The 'Temper' feature in the input data is typically a categorical variable. However, in order to utilize machine learning models effectively, it is necessary to convert this categorical variable into numerical labels using a label encoder. When interacting with the user, it is important to provide the following input values based on the temper used: 'F': 0, 'H11': 1, 'H111': 2, 'H112': 3, 'H116': 4, 'H12': 5, 'H13': 6, 'H14': 7, 'H15': 8, 'H16': 9, 'H17': 10, 'H18': 11, 'H19': 12, 'H21': 13, 'H22': 14, 'H24': 15, 'H25': 16, 'H26': 17, 'H27': 18, 'H28': 19, 'H29': 20, 'H32': 21, 'H321': 22, 'H322': 23, 'H34': 24, 'H36': 25, 'H38': 26, 'O': 27, 'T1': 28, 'T11': 29, 'T2': 30, 'T3': 31, 'T31': 32, 'T351': 33, 'T3510': 34, 'T3511': 35, 'T36': 36, 'T361': 37, 'T37': 38, 'T4': 39, 'T42': 40, 'T451': 41, 'T4510': 42, 'T4511': 43, 'T452': 44, 'T5': 45, 'T51': 46, 'T52': 47, 'T53': 48, 'T54': 49, 'T551': 50, 'T5510': 51, 'T5511': 52, 'T6': 53, 'T61': 54, 'T6151': 55, 'T62': 56, 'T64': 57, 'T65': 58, 'T651': 59, 'T6510': 60, 'T6511': 61, 'T652': 62, 'T66': 63, 'T7': 64, 'T71': 65, 'T72': 66, 'T73': 67, 'T7351': 68, 'T73510': 69, 'T73511': 70, 'T7352': 71, 'T74': 72, 'T7451': 73, 'T7452': 74, 'T7454': 75, 'T76': 76, 'T7651': 77, 'T76510': 78, 'T76511': 79, 'T8': 80, 'T81': 81, 'T83': 82, 'T831': 83, 'T832': 84, 'T851': 85, 'T8510': 86, 'T8511': 87, 'T852': 88, 'T861': 89, 'T87': 90, 'T9': 91")
    st.markdown("For type column if it is wrought alloy keep it as 1, if it is cast alloy keep it as 0")
    st.header("Another Description")

    # Display label with a paragraph
    st.markdown("This is a label with a paragraph. You can provide additional details and information here.")
    # time.sleep(100)
    time.sleep(100)
def tab3():
    # st.write("I am in tab3")
    # Set page fff

    # Load the trained model
    model = joblib.load(r'C:\Users\kallu\mtp_second_phase\prediction_Mech_based.joblib')

    st.subheader("Predict fatigue strength based on features")

    # Feature input
    st.header("Feature Inputs")
    st.markdown("---")

    # Define feature names
    feature_names = [

        'Brinell_Hardness ', 'Ultimate_tensile_strength',
        'Tensile yield strength',
        # Add more feature names as needed
    ]
    col1, col2, col3 = st.columns(3)

    # Distribute features among the columns
    num_features = len(feature_names)
    features_per_column = num_features // 3
    remainder = num_features % 3

    feature_values = []

    for i in range(features_per_column + min(1, remainder)):
        if i < features_per_column:
            with col1:
                value = st.number_input(feature_names[i], value=0.0)
                feature_values.append(value)
        if i < features_per_column:
            with col2:
                value = st.number_input(feature_names[i + features_per_column], value=0.0)
                feature_values.append(value)
        if i < features_per_column + min(1, remainder):
            with col3:
                value = st.number_input(feature_names[i + 2 * features_per_column], value=0.0)
                feature_values.append(value)

    # Prediction
    if st.button("Predict 3", key=8):
        # Perform prediction using the loaded model
        # feature_values2 = []
        # for i in range(3):
        #     for j in range(0, 27, 3):
        #         feature_values2.append(feature_values[i + j])
        # print(feature_values2)
        # time.sleep(10)
        # feature_values=[70,290,220,1,17,0,95.9260184953762,0,0,0,0,0.149962509372657,0.149962509372657,0.3499125218695330,0,0,2.0994751312172,0.849787553111722,0,0,0.199950012496876,0,0.049987503124219,0,0.149962509372657,0,0.0749812546863284]
        feature_values = np.array(feature_values).reshape((1, 3))
        prediction = model.predict(feature_values)
        st.success(f"Fatigue Strength Prediction: {prediction[0]:.2f} MPa")
        st.markdown("---")
    st.header("Description")

    # Display label with a paragraph
    st.markdown(
        "The 'Temper' feature in the input data is typically a categorical variable. However, in order to utilize machine learning models effectively, it is necessary to convert this categorical variable into numerical labels using a label encoder. When interacting with the user, it is important to provide the following input values based on the temper used: 'F': 0, 'H11': 1, 'H111': 2, 'H112': 3, 'H116': 4, 'H12': 5, 'H13': 6, 'H14': 7, 'H15': 8, 'H16': 9, 'H17': 10, 'H18': 11, 'H19': 12, 'H21': 13, 'H22': 14, 'H24': 15, 'H25': 16, 'H26': 17, 'H27': 18, 'H28': 19, 'H29': 20, 'H32': 21, 'H321': 22, 'H322': 23, 'H34': 24, 'H36': 25, 'H38': 26, 'O': 27, 'T1': 28, 'T11': 29, 'T2': 30, 'T3': 31, 'T31': 32, 'T351': 33, 'T3510': 34, 'T3511': 35, 'T36': 36, 'T361': 37, 'T37': 38, 'T4': 39, 'T42': 40, 'T451': 41, 'T4510': 42, 'T4511': 43, 'T452': 44, 'T5': 45, 'T51': 46, 'T52': 47, 'T53': 48, 'T54': 49, 'T551': 50, 'T5510': 51, 'T5511': 52, 'T6': 53, 'T61': 54, 'T6151': 55, 'T62': 56, 'T64': 57, 'T65': 58, 'T651': 59, 'T6510': 60, 'T6511': 61, 'T652': 62, 'T66': 63, 'T7': 64, 'T71': 65, 'T72': 66, 'T73': 67, 'T7351': 68, 'T73510': 69, 'T73511': 70, 'T7352': 71, 'T74': 72, 'T7451': 73, 'T7452': 74, 'T7454': 75, 'T76': 76, 'T7651': 77, 'T76510': 78, 'T76511': 79, 'T8': 80, 'T81': 81, 'T83': 82, 'T831': 83, 'T832': 84, 'T851': 85, 'T8510': 86, 'T8511': 87, 'T852': 88, 'T861': 89, 'T87': 90, 'T9': 91")
    st.markdown("For type column if it is wrought alloy keep it as 1, if it is cast alloy keep it as 0")
    st.header("Another Description")

    # Display label with a paragraph
    st.markdown("This is a label with a paragraph. You can provide additional details and information here.")
    # time.sleep(100)
    time.sleep(100)
hometab()
