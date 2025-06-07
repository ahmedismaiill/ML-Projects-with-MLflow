import joblib
import pandas as pd
import streamlit as st

# Load Model
model_XGBoost = joblib.load('XGBoost_model.pkl')
#-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# Build Streamlit
st.cache_data.clear()

st.set_page_config(page_title="Heart Failure Prediction",page_icon='🚨')

st.title(':red[Heart] Failure Prediction 🫀🏥')
st.markdown(
    """
    <div style="background-color:#000000; padding:10px; border-radius:5px">
        <h4 style="color:#faf7f7;">A heart failure prediction app uses machine learning and patient data to assess risk, enabling early diagnosis and informed medical decisions.🤖💡
        </h4>
    </div>
    """,
    unsafe_allow_html=True
)
st.image("heart_beating_0.gif",use_container_width=True)

Chest_Pain_list_selection = ['Yes','No']
Shortness_of_Breath_list_selection = ['Yes','No']
Fatigue_list_selection = ['Yes','No']
Palpitations_list_selection = ['Yes','No']
Dizziness_list_selection = ['Yes','No']
Swelling_list_selection = ['Yes','No']
Pain_Arms_Jaw_Back_list_selection = ['Yes','No']
Cold_Sweats_Nausea_list_selection = ['Yes','No']
High_BP_list_selection = ['Yes','No']
High_Cholesterol_list_selection = ['Yes','No']
Sedentary_Lifestyle_list_selection = ['Yes','No']  


with st.container(height=1400):
    col1 , col2 = st.columns(2)

    with col1:
        st.write("### Chest Pain")
        st.caption("Presence of chest pain (Yes/No)")
        Chest_Pain = st.selectbox("Select:",Chest_Pain_list_selection,key='Chest Pain')

        st.write("### Shortness of Breath")
        st.caption("Difficulty breathing (Yes/No)")
        Shortness_of_Breath = st.selectbox("Select:",Shortness_of_Breath_list_selection,key='Shortness of Breath')        
        
        st.write("### Fatigue")
        st.caption("Persistent tiredness without an obvious cause (Yes/No)")
        Fatigue = st.selectbox("Select:",Fatigue_list_selection,key='Fatigue')

        st.write("### Palpitations")
        st.caption("Irregular or rapid heartbeat (Yes/No)")
        Palpitations = st.selectbox("Select:",Palpitations_list_selection,key='Palpitations')

        st.write("### Dizziness")
        st.caption("Episodes of lightheadedness or fainting (Yes/No)")
        Dizziness = st.selectbox("Select:",Dizziness_list_selection,key='Dizziness')

        st.write("### Swelling")
        st.caption("Yes: if FastingBS > 120 mg/dl")
        Swelling = st.selectbox("Select:",Swelling_list_selection,key='Swelling')

        


    with col2:
        st.write("### Pain Arms Jaw Back ")
        st.caption("Radiating pain, a strong heart attack indicator.")
        Pain_Arms_Jaw_Back = st.selectbox("Select:",Pain_Arms_Jaw_Back_list_selection,key='Pain_Arms_Jaw_Back')
       
        st.write("### Cold Sweats Nausea")
        st.caption("Cold sweats and nausea (Yes/No)")
        Cold_Sweats_Nausea = st.selectbox("Select:",Cold_Sweats_Nausea_list_selection,key='Cold_Sweats_Nausea')

        st.write("### High Blood Pressure")
        st.caption("A leading cause of heart disease.")
        High_BP = st.selectbox("Select:",High_BP_list_selection,key='High_BP')        
        
        st.write("### High_Cholesterol")
        st.caption("Elevated cholesterol levels (Yes/No)")
        High_Cholesterol = st.selectbox("Select:",High_Cholesterol_list_selection,key='High_Cholesterol')

        st.write("### Sedentary_Lifestyle")
        st.caption("Lack of physical activity, leading to poor cardiovascular health")
        Sedentary_Lifestyle = st.selectbox("Select:",Sedentary_Lifestyle_list_selection,key='Sedentary_Lifestyle')

        st.write("### Age")
        Age = st.slider("You selected:",15,90,45)

        b = st.button("Start",icon='🚨',use_container_width=True)


Chest_Pain_encode = ['Yes','No']
encode_Chest_Pain = [1,0]
convert_Chest_Pain_encode = dict(zip(Chest_Pain_encode, encode_Chest_Pain))

Shortness_of_Breath_to_encode = ['Yes','No']
encode_Shortness_of_Breath = [1,0]
convert_Shortness_of_Breath = dict(zip(Shortness_of_Breath_to_encode, encode_Shortness_of_Breath))

Fatigue_to_encode = ['Yes','No']
encode_Fatigue = [1,0]
convert_Fatigue = dict(zip(Fatigue_to_encode, encode_Fatigue))

Palpitations_to_encode = ['Yes','No']
encode_Palpitations = [1,0]
convert_Palpitations = dict(zip(Palpitations_to_encode, encode_Palpitations))

Dizziness_to_encode = ['Yes','No']
encode_Dizziness = [1,0]
convert_Dizziness = dict(zip(Dizziness_to_encode, encode_Dizziness))

Swelling_to_encode = ['Yes','No']
encode_Swelling = [1,0]
convert_Swelling = dict(zip(Swelling_to_encode, encode_Swelling))

Pain_Arms_Jaw_Back_to_encode = ['Yes','No']
encode_Pain_Arms_Jaw_Back = [1,0]
convert_Pain_Arms_Jaw_Back = dict(zip(Pain_Arms_Jaw_Back_to_encode, encode_Pain_Arms_Jaw_Back))

Cold_Sweats_Nausea_to_encode = ['Yes','No']
encode_Cold_Sweats_Nausea = [1,0]
convert_Cold_Sweats_Nausea = dict(zip(Cold_Sweats_Nausea_to_encode, encode_Cold_Sweats_Nausea))

High_BP_to_encode = ['Yes','No']
encode_High_BP = [1,0]
convert_High_BP = dict(zip(High_BP_to_encode, encode_High_BP))

High_Cholesterol_to_encode = ['Yes','No']
encode_High_Cholesterol = [1,0]
convert_High_Cholesterol = dict(zip(High_Cholesterol_to_encode, encode_High_Cholesterol))

Sedentary_Lifestyle_to_encode = ['Yes','No']
encode_Sedentary_Lifestyle = [1,0]
convert_Sedentary_Lifestyle = dict(zip(Sedentary_Lifestyle_to_encode, encode_Sedentary_Lifestyle))

								
try:
 df = pd.DataFrame({
          'Chest_Pain':convert_Chest_Pain_encode[Chest_Pain],
          'Shortness_of_Breath':convert_Shortness_of_Breath[Shortness_of_Breath],
          'Fatigue':convert_Fatigue[Fatigue],
          'Palpitations':convert_Palpitations[Palpitations],
          'Dizziness':convert_Dizziness[Dizziness],
          'Swelling':convert_Swelling[Swelling],
          'Pain_Arms_Jaw_Back':convert_Pain_Arms_Jaw_Back[Pain_Arms_Jaw_Back],
          'Cold_Sweats_Nausea':convert_Cold_Sweats_Nausea[Cold_Sweats_Nausea],    
          'High_BP':convert_High_BP[High_BP],
          'High_Cholesterol':convert_High_Cholesterol[High_Cholesterol],
          'Sedentary_Lifestyle':convert_Sedentary_Lifestyle[Sedentary_Lifestyle],
          'Age':[Age],},index=[0])
except ValueError:
    st.error("Please ensure all numerical fields are correctly filled! ❌")

print(df)


def predict_heart_failure(df):
    prediction = model_XGBoost.predict(df)
    prediction_prob = model_XGBoost.predict_proba(df)  
    return prediction, prediction_prob
prediction,prediction_prob = predict_heart_failure(df)

no_patient = str((prediction_prob[0,0])*100)[:5] + '%'
patient = str((prediction_prob[0,1])*100)[:5] + '%'

if b:
 st.balloons()
 with st.sidebar:
    st.markdown(
    """
    <div style="text-align: center;">
        <h1>🚀 Prediction Is</h1>
    </div>
    """,
    unsafe_allow_html=True
    )
    with st.sidebar.container(height=400):
       co1 , co2 = st.columns(2)
       with co1:
          st.image('giphy.gif'
                 ,use_container_width=True)
          st.write(f"# Prediction Probability: {patient}")
          st.subheader(":red[*Heart Patient*]")


     
     
       with co2:
            st.image('Black And White Heart GIF.gif'
                     ,use_container_width=True)
            st.write(f"# Prediction Probability: {no_patient}")
            st.subheader(":green[*No Heart Patient*]")

          
             
    if prediction == 1:  
        st.markdown(
                      """
                 <div style="text-align: center;">
                 <h1>Expected He Is</h1>
                 </div>
                      """,
        unsafe_allow_html=True)
        st.markdown(
                          """
                <div style="text-align: center;">
                <h1 style="color: red;">Heart Patient</h1>
                </div>
                """,
                 unsafe_allow_html=True)
    else:
         st.markdown(
                      """
                 <div style="text-align: center;">
                 <h1>Expected He Is</h1>
                 </div>
                      """,
        unsafe_allow_html=True)
         st.markdown(
                          """
                <div style="text-align: center;">
                <h1 style="color: green;">Not Heart Patient</h1>
                </div>
                """,
                 unsafe_allow_html=True)
