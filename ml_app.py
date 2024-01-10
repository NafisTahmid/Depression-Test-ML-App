# Install Core Packages
import streamlit as st

# Load ML Packages
import joblib
import os
import sklearn


# Load EDA Packages
import numpy as np

attrib_info = """
#### Attribute Information:
    - Sex 1. Male, 2.Female
    - Age 18-30
    - University Year: 1. First year, 2. Second year, 3. Third year, 4. Final year, 5. Final Year(Medical), 6. Post-Graduation
    - Family Members: 1. One, 2. Two, 3. Three, 4. Four, 5. Five, 6. Six, 7. More than six
    - Educational Background: 1. Bangla Medium, 2. English Medium, 3. Madrasah
    - Relationship Status: 1. Single, 2. In a relationship, 3. Married,
    - Number of children: 1. No children, 2. One, 3. Two, 4. Three, 5. Four, 6. More than four
    - Brought up in the capital city 1.Yes, 2.No.
    - Has any part time job or income 1.Yes, 2.No.
    - Monthly income: 1. Not earning now, 2. Nearly 5K, 3. Nearly 10K, 4. Nearly 20K, 5. Nearly 40K, 6. Nearly 60K, 7. Nearly 80K, 8. Nearly 100K, 9. More than 100K
    - Monthly living expense: 1. Nearly 5K, 2. Nearly 10K, 3. Nearly 20K, 4. Nearly 40K, 5. Nearly 60K, 6. Nearly 80K, 9. Nearly 100K, 10. More than 100K
    - Academic performance satisfaction 1.Yes, 2.No.
    - Has any physical disabilities 1.Yes, 2.No.
    - Ever been in a road accident 1.Yes, 2.No.
    - Has any childhood trauma 1.Yes, 2.No.
    - Is taking any doctor prescribed medication  1.Yes, 2.No.
    - Is a religious person 1.Yes, 2.No.
    - Social gathering time: 1. Rarely go to social gathering, 2. Go out on weekends and also weekdays, 3. Weekends only
    - Regular participant in indoor activity 1.Yes, 2.No.
    - Does sports or gym regularly 1.Yes, 2.No.
    - Social media time: 1. Less than thirty minutes, 2. One hour, 3. Two hours, 4. Three hours, 5. Four hours, 6. Five hours, 7. More than five hours
    - Social life satisfaction: 1. Disappointed, 2. Not Satisfied, 3. Satisfied, 4. Very Satisfied
    - Is a tea/coffee person  1.Yes, 2.No.
    - Has any substance addiction 1.Yes, 2.No.
    - Sleep duration: 1. Three hours, 2. Four hours, 3. Five hours, 4. Six hours, 5. Seven hours, 6. Eight hours, 7. Nine hours, 8. More than nine hours
    - Class 1.Positive, 2.Negative.

"""
label_dict = {"No":0,"Yes":1}
# gender_map = {"Female":0,"Male":1}
gender_map = {'Male':0, 'Female':1}
age_map = {'18':0, '19':1, '20':2, '21':3, '22':4, '23':5, '24':6, '25':7, '26':8, '27':9, '30':10, '30+':11}
university_year_map = {'1st':0, '2nd':1, '3rd':2, '4th':3, 'Post-graduation':4, 'Final(Medical)':5}
family_members_map = {'1':0, '2':1, '3':2, '4':3, '5':4, '6':5, '6+':6}
educational_background_map = {'Bangla Medium':0, 'English Medium':1, 'Madrasah': 2}
relationship_status_map = {'Single':0, 'In a Relationship':1, 'Married': 2}
children_map = {'0':0, '1':1, '2':2, '3':3, '4':4, '4+':5}
monthly_income_map = {'Not earning as of now':0, 'Nearly 5K':1, 'Nearly 10K':2, 'Nearly 20K':3, 'Nearly 40K':4, 'Nearly 60K':5, 'Nearly 80K':6, 'Nearly 100K':7, 'More than 100K':8}
monthly_living_expense_map = {'Nearly 5K':0, 'Nearly 10K':1, 'Nearly 20K':2, 'Nearly 40K':3, 'Nearly 60K':4, 'Nearly 80K':5, 'Nearly 100K':6, 'More than 100K':7}
social_gathering_time_map = {'I rarely go to social gatherings': 0, 'I go out on weekends and also weekdays': 1, 'Weekends only':2}
social_media_time_map = {'Less than 30 minutes': 0, '1 hour':1, '2 hours':2, '3 hours':3, '4 hours':4, '5 hours':5, 'More than 5 hours':6}
sleep_duration_map = {"3 hours":0, "4 hours":1, "5 hours":2, "6 hours":3, "7 hours":4, "8 hours":5, "8 hours":6, "More than 9 hours":7}
social_life_satisfaction_map = {"Disappointed":0, "Not satisfied":1, "Satisfied":2, "Very satisfied":3}

# target_label_map = {"Negative":0,"Positive":1}

def get_fvalue(val):
    feature_dict = {"No":0, "Yes":1}
    for key,value in feature_dict.items():
        if val == key:
            return value

def get_value(val, my_dict):
    for key,value in my_dict.items():
        if val == key:
            return value

# Load ML Models
# @st._cache_data
def load_model(model_file):
    loaded_model = joblib.load(open(os.path.join(model_file), "rb"))
    return loaded_model

def run_ml_app():
    st.subheader("ML Prediction")

    with st.expander("Attribute Info"):
        st.write(attrib_info)

    # Layout
    col1, col2 = st.columns(2)

    with col1:
        age = st.number_input("Age",18,35)
        gender = st.radio("Gender", ["Female", "Male"])
        university_year = st.selectbox("University Year", ['1st', '2nd', '3rd', '4th', 'Post-graduation', 'Final(Medical)'])
        family_members = st.selectbox("Family Members", ['1', '2', '3', '4', '5', '6', '6+'])
        educational_background = st.radio("Educational Background", ['Bangla Medium', 'English Medium', 'Madrasah'])
        relationship_status = st.radio("Relationship Status", ['Single', 'In a Relationship', 'Married'])
        number_of_children = st.select_slider("Number of children", ['0', '1', '2', '3', '4', '4+'])
        brought_in_capital = st.radio("Brought up in capital", ["No", "Yes"])
        part_time_income = st.radio("Part time job or income", ["No", "Yes"])
        monthly_income = st.select_slider("Monthly income", ['Not earning as of now', 'Nearly 5K', 'Nearly 10K', 'Nearly 20K', 'Nearly 40K', 'Nearly 60K', 'Nearly 80K', 'Nearly 100K', 'More than 100K'])
        monthly_living_expense = st.selectbox("Monthly living expense", ['Nearly 5K', 'Nearly 10K', 'Nearly 20K', 'Nearly 40K', 'Nearly 60K', 'Nearly 80K', 'Nearly 100K', 'More than 100K'])
        academic_performance_satisfaction = st.radio("Academic performance satisfaction", ["No", "Yes"])
        physical_disabilities = st.radio("Any physical disabilities", ["No", "Yes"])



    with col2:
        ever_in_a_road_accident = st.radio("Ever in a road accident", ["No", "Yes"])
        childhood_trauma = st.radio("Childhood trauma", ["No", "Yes"])
        medication = st.radio("Doctor prescribed medication regularly", ["No", "Yes"])
        religious_person = st.radio("A religious person", ["No", "Yes"])
        social_gathering_time = st.selectbox("Social Gathering time", ['I rarely go to social gatherings', 'I go out on weekends and also weekdays', 'Weekends only'])
        participant_in_indoor_activity = st.radio("Rregular participant in indoor activity", ["No", "Yes"])
        sports_or_gym = st.radio("Do sports of gym regularly", ["No", "Yes"])
        social_media_time = st.select_slider("Social media time", ['Less than 30 minutes', '1 hour', '2 hours', '3 hours', '4 hours', '5 hours', 'More than 5 hours'])
        social_life_satisfaction = st.selectbox("Social life satisfaction", ["Disappointed", "Not satisfied", "Satisfied", "Very satisfied"])
        tea_person = st.radio("Tea or coffee person", ["No", "Yes"])
        substance_addiction = st.radio("Any Substance addiction", ["No", "Yes"])
        sleep_duration = st.select_slider("Sleep duration", ["3 hours", "4 hours", "5 hours", "6 hours", "7 hours", "8 hours", "8 hours", "More than 9 hours"])

    with st.expander("Your Selected Options:"):
        result = {'age':age,
                  'gender':gender,
                  'university_year':university_year,
                  'family_members':family_members,
                  'educational_background':educational_background,
                  'relationship_status':relationship_status,
                  'number_of_children':number_of_children,
                  'brought_up_in_capital':brought_in_capital,
                  'part_time_income':part_time_income,
                  'monthly_income':monthly_income,
                  'monthly_living_expense':monthly_living_expense,
                  'academic_performance_satisfaction':academic_performance_satisfaction,
                  'physical_disabilities':physical_disabilities,
                  'ever_in_a_road_accident':ever_in_a_road_accident,
                  'childhood_trauma':childhood_trauma,
                  'medication':medication,
                  'religious_person': religious_person,
                  'social_gathering_time':social_gathering_time,
                  'participant_in_indoor_activity': participant_in_indoor_activity,
                  'sports_or_gym': sports_or_gym,
                  'social_media_time': social_media_time,
                  'social_life_satisfaction': social_life_satisfaction,
                  'tea_or_coffee_person': tea_person,
                  'any_substance_addiction': substance_addiction,
                  'sleep_duration': sleep_duration
                 }
        st.write(result)

        encoded_result = []
        for i in result.values():
            if type(i) == int:
                encoded_result.append(i)
            elif i in ["Female", "Male"]:
                res = get_value(i, gender_map)
                encoded_result.append(res)
            elif i in ['1st', '2nd', '3rd', '4th', 'Post-graduation', 'Final(Medical)']:
                res = get_value(i, university_year_map)
                encoded_result.append(res)
            elif i in ['1', '2', '3', '4', '5', '6', '6+']:
                res = get_value(i, family_members_map)
                encoded_result.append(res)
            elif i in ['Bangla Medium', 'English Medium', 'Madrasah']:
                res = get_value(i, educational_background_map)
                encoded_result.append(res)
            elif i in ['Single', 'In a Relationship', 'Married']:
                res = get_value(i, relationship_status_map)
                encoded_result.append(res)
            elif i in ['0', '1', '2', '3', '4', '4+']:
                res = get_value(i, children_map)
                encoded_result.append(res)
            elif i in ['Not earning as of now', 'Nearly 5K', 'Nearly 10K', 'Nearly 20K', 'Nearly 40K', 'Nearly 60K', 'Nearly 80K', 'Nearly 100K', 'More than 100K']:
                res = get_value(i, monthly_income_map)
                encoded_result.append(res)
            elif i in ['Nearly 5K', 'Nearly 10K', 'Nearly 20K', 'Nearly 40K', 'Nearly 60K', 'Nearly 80K', 'Nearly 100K', 'More than 100K']:
                res = get_value(i, monthly_living_expense_map)
                encoded_result.append(res)
            elif i in ['I rarely go to social gatherings', 'I go out on weekends and also weekdays', 'Weekends only']:
                res = get_value(i, social_gathering_time_map)
                encoded_result.append(res)
            elif i in ['Less than 30 minutes', '1 hour', '2 hours', '3 hours', '4 hours', '5 hours', 'More than 5 hours']:
                res = get_value(i, social_media_time_map)
                encoded_result.append(res)
            elif i in ["3 hours", "4 hours", "5 hours", "6 hours", "7 hours", "8 hours", "8 hours", "More than 9 hours"]:
                res = get_value(i, sleep_duration_map)
                encoded_result.append(res)
            elif i in ["Disappointed", "Not satisfied", "Satisfied", "Very satisfied"]:
                res = get_value(i, social_life_satisfaction_map)
                encoded_result.append(res)
            else:
                encoded_result.append(get_fvalue(i))

        st.write(encoded_result)

    with st.expander("Prediction Result:"):
        single_sample = np.array(encoded_result).reshape(1, -1)
        # st.write(single_sample)

        model = load_model("models/depression_dataset_trained_model_updated.sav")
        prediction = model.predict(single_sample)
        predict_prob = model.predict_proba(single_sample)
        # st.write(prediction)
        # st.write(predict_prob)

        if prediction == 1:
            st.warning("Positive Risk {}".format(prediction[0]))
            pred_probability_score = {"Negative DM Risk": predict_prob[0][0]*100,
                                      "Positive DM Risk": predict_prob[0][1]*100}
            st.write(pred_probability_score)

        else:
            st.success("Negative Risk {}".format(prediction[0]))
            pred_probability_score = {"Negative DM Risk": predict_prob[0][0]*100,
                                      "Positive DM Risk": predict_prob[0][1]*100}
            st.write(pred_probability_score)


