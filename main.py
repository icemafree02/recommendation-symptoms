import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import json
from sklearn.preprocessing import LabelEncoder

df = pd.read_csv('D:\symptoms_recommendation\data.csv')
df

#small EDA dataset.

df.duplicated().sum()

df = df.drop_duplicates()

df.duplicated().sum()

# Apply json format to python.

summary = df['summary'].apply(json.loads)
summary

no_symptoms = summary.apply(lambda x: x['no_symptoms'])
for i in no_symptoms:
  if i != []:
    print(i)

count = 0
for i in no_symptoms:
  if i != []:
    count += 1
print(count)

no_symptoms_text = no_symptoms.apply(lambda text: [x['text'] for x in text])
no_symptoms_text = no_symptoms_text[no_symptoms_text.str.len() > 0]
no_symptoms_text = no_symptoms_text.to_frame(name="no_symptoms_text")
no_symptoms_text

total = len(no_symptoms_text)
total

no_symptoms_answers = []
count = 0
for i in no_symptoms:
  if i != []:
    for j in i:
      no_symptoms_answers.append(j['answers'])
      count += 1
no_symptoms_answers

idk_symtopms = summary.apply(lambda x: x['idk_symptoms'])
idk_symtopms = idk_symtopms[idk_symtopms.str.len() > 0]
idk_symtopms

idk_symtopms_text = idk_symtopms.apply(lambda text: [x['text'] for x in text])
idk_symtopms_text = idk_symtopms_text.apply(lambda x:' '.join(x))
idk_symtopms_text

idk_symtopms_answers = idk_symtopms.apply(lambda answers: [x['answers'] for x in answers])
idk_symtopms_answers

yes_symptoms = summary.apply(lambda x: x['yes_symptoms'])
yes_symptoms

# Extract the symptoms of patient they are experiencing.

yes_symptoms_text = yes_symptoms.apply(lambda text: [x['text'] for x in text])
yes_symptoms_text

count = 0
for i in yes_symptoms_text:
  if i != []:
    count += 1
print(count)

yes_symptoms_text = yes_symptoms_text.apply(lambda x:x[:-1])
yes_symptoms_text

# Convert search_term to list for each patient.

search_term = df['search_term']
search_term = search_term.str.split(",")
search_term.head(20)

search_term_value = []
for i in df['search_term']:
  for j in i:
    search_term_value.append(j)

# Extract the blank space form list and values inside of it.

search_term_value = [x.strip() for x in search_term_value if x.strip()]
search_term_value

search_term_columns = pd.DataFrame(search_term_value)
search_term_columns = search_term_columns.rename(columns={0: 'serch_term_values'})
search_term_columns['serch_term_values'].unique()

yes_symptoms_text_columns_list = ['เสมหะ', 'ไอ', 'น้ำมูกไหล', 'ปวดท้อง', 'ตาแห้ง', 'ปวดกระดูก',
       'เจ็บคอ', 'อาเจียน', 'ปวดเมื่อยกล้ามเนื้อ', 'เสมหะไหลลงคอ',
       'Fever', 'คันจมูก', 'บวมตามร่างกาย', 'ปวดข้อเท้า', 'ปวดข้อ',
       'ประวัติอุบัติเหตุ', 'ท้องเสีย', 'ปวดข้อมือ', 'เป็นไข้',
       'หายใจมีเสียงวี๊ด', 'หายใจหอบเหนื่อย', 'ถ่ายเป็นเลือด', 'ปวดหลัง',
       'เสียงแหบ', 'ปวดหัว', 'ชา', 'diarrhea', 'ปากบวม หน้าบวม',
       'แผลในปาก', 'ตาบวม', 'ก้อนที่ผิวหนัง', 'ท้องผูก', 'ปวดเอว',
       'abdominal pain', 'หูอื้อ', 'straining to urinate', 'ปวดเข่า',
       'ผื่น', 'ปวดไหล่', 'คัดจมูก', 'กลืนติด กลืนลำบาก', 'back pain',
       'ปวดจมูก', 'cough', 'wheezing sound', 'Dizzy', 'ตกขาวผิดปกติ',
       'ปวดต้นคอ', 'ปวดขา', 'เวียนศีรษะ บ้านหมุน', 'หน้ามืด',
       'ประวัติความดันสูง', 'axillary mass', 'acne', 'ตาพร่ามัว',
       'จุดดำลอยในตา', 'ผิวแห้ง', 'ปวดตา', 'เรอเปรี้ยว', 'ปวดท้องน้อย',
       'shortness of breath', 'คัน', 'Jaundice', 'ปวดหู',
       'eye mucus, eye discharge', 'ปวดแขน', 'itch', 'skin rash',
       'จามบ่อย', 'ear discharge/drainage', 'hearing loss', 'คอแห้ง',
       'Headache', 'sore throat', 'ตาแดง', 'ฝ้าขาวที่ลิ้น', 'ท้องอืด',
       'ประวัติโรคกระเพาะ', 'เคืองตา', 'ขี้ตา', 'ก้อนบริเวณหู',
       'คลื่นไส้', 'ตากระตุก', 'ปวดมือ', 'ปวดเท้า', 'vomiting',
       'dry skin', 'ear pain', 'foot pain', 'joint pain', 'ปัสสาวะขุ่น',
       'unsteady, loss of balance', 'การได้ยินลดลง',
       'Animal related injury', 'เดินเซ ทรงตัวไม่ได้',
       'เสียงดังรบกวนในหู', 'ปัสสาวะบ่อย ถี่', 'ผมร่วง', 'runny nose',
       'fatigue associated with exertion', 'dry throat',
       'ปัสสาวะเป็นเลือด', 'ปัสสาวะแสบขัด', 'ประจำเดือนมากกว่าปกติ',
       'ปวดข้อศอก', 'กลืนเจ็บ', 'ปวดซี่โครง', 'slurred speech',
       'ก้อนที่รักแร้', 'แขนขาอ่อนแรง', 'ร้อนวูบวาบ', 'น้ำตาไหล',
       'ก้อนบริเวณใบหน้า', 'เจ็บหน้าอก', 'Blood in urine', 'stuffy nose',
       'shortness of breath when lying down', 'Weight loss',
       'Loss of appetite', 'ปัสสาวะกะปริบกะปรอย', 'phlegm',
       'มองเห็นภาพซ้อน', 'ง่วงเยอะ', 'นอนไม่หลับ', 'ความเครียด',
       'รู้สึกไร้ค่า', 'วิตกกังวล', 'คิดอยากฆ่าตัวตาย ทำร้ายตนเอง',
       'History of trauma', 'มือสั่น', 'อุจจาระลำเล็กลง',
       'ก้อนบริเวณขาหนีบ', 'eye irritation',
       'History of hypertension (high blood pressure)', 'nausea',
       'neck pain', 'ankle pain', 'ไอเป็นเลือด', 'เลือดกำเดาไหล',
       'ปวดกราม', 'genitals itching', 'narrow stool', 'ประวัติไขมันสูง',
       'ปวดบริเวณใบหน้า', 'itchy nose', 'จ้ำเลือด',
       'ประจำเดือนมาน้อย, ประจำเดือนขาด', 'จุดเลือดออก', 'blurry vision',
       'fainted', 'lightheadness', 'วูบ']

search_term_columns_list = ['มีเสมหะ', 'ไอ', 'น้ำมูกไหล', 'ปวดท้อง', 'ตาแห้ง', 'ปวดกระดูก',
       'คันจมูกจามบ่อย', 'คันคอ', 'อาเจียน', 'ปวดเมื่อยกล้ามเนื้อทั่วๆ',
       'เสมหะไหลลงคอ', 'Fever', 'คันจมูก', 'บวม', 'ปวดข้อเท้า', 'เจ็บคอ',
       'ท้องเสีย', 'ปวดข้อมือ', 'ไข้', 'หายใจมีเสียงวี๊ด',
       'หายใจหอบเหนื่อย', 'ถ่ายเป็นเลือดสด', 'ปวดหลัง', 'เสียงแหบ',
       'ปวดหัว', 'ชา', 'diarrhea', 'ปากบวม', 'แผลในช่องปาก', 'ตาบวม',
       'ก้อนที่ผิวหนัง', 'มีเสมหะน้ำมูกไหล', 'ท้องผูก', 'ปวดบั้นเอว',
       'น้ำมูกไหลมีเสมหะ', 'น้ำมูกไหลคัดจมูก', 'ปวดข้อ', 'stomachache',
       'คัดจมูก', 'กลืนเจ็บ', 'หูอื้อ', 'น้ำมูกไหลไอ', 'ไอกลางคืนมีเสมหะ',
       'straining to urinate', 'ปวดเข่า', 'คลื่นไส้', 'อาเจียนแสบท้อง',
       'ไอผื่น', 'ปวดหัวไหล่', 'จมูกตัน', 'หายใจไม่สะดวก', 'จุกแน่นท้อง',
       'กลืนลำบาก', 'back pain', 'แน่นจมูก', 'ปวดจมูก',
       'ก้อนบริเวณท้องน้อยปวดท้อง', 'แสบท้อง', 'ท้องอืด', 'คัดจมูกไอ',
       'cough', 'night coughFever', 'phlegm', 'Sore throat', 'wheezing',
       'Stuffy nose', 'nasal congestion', 'labored breathing', 'Dizzy',
       'coughLightheaded', 'ตกขาวผิดปกติ', 'มึนศีรษะอาเจียน',
       'ไอไอกลางคืน', 'ถ่ายเหลว', 'ปวดต้นคอ', 'ตัวร้อน', 'ปวดน่อง',
       'มึนศีรษะ', 'เวียนศีรษะ', 'หน้ามืด', 'กระแทก', 'armpit lump',
       'skin lumpAcne', 'ตามองเห็นไม่ชัด', 'มองเห็นจุดดำ เงาดำ',
       'ผิวแห้ง', 'ผื่น', 'ปวดตา', 'ปวดลูกตา', 'เรอเปรี้ยวปวดท้อง',
       'ปวดท้องน้อย', 'คัน', 'Yellowish skin', 'เจ็บคอไข้', 'ปวดหู',
       'Eye discharge', 'ปวดแขน', 'itch', 'skin rash', 'จามบ่อย',
       'ไอคัดจมูก', 'Hearing loss', 'Ear dischargeEar pain', 'คอแห้ง',
       'มีแผล', 'Headache', 'ตาแดง', 'ฝ้าขาวที่ลิ้น', 'ตาเหลือง',
       'ปวดตาเคืองตา', 'บ้านหมุน', 'มีเสมหะคัดจมูก', 'คลื่นไส้อาเจียน',
       'คัดจมูกมีเสมหะ', 'คันตา', 'ขี้ตาเยอะ', 'เคืองตา', 'ก้อนที่หลังหู',
       'ก้อนที่ขา', 'อาเจียนคลื่นไส้', 'ตากระตุก', 'มีแผลผื่น', 'ฟกช้ำ',
       'จาม', 'คันจมูกน้ำมูกไหล', 'breathless on lyingStuffy nose',
       'จุกแสบลำคอ', 'ปวดนิ้วมือ', 'ตาพร่ามัว', 'ปวดเท้า', 'ปวดคอ',
       'vomit', 'ปวดบ่า', 'dry skin', 'Ear discharge', 'Ear pain',
       'แผลริมฝีปาก', 'foot pain', 'ปัสสาวะเป็นตะกอน',
       'ท้องเสียปวดท้องน้อย', 'มีเสมหะไอ', 'การได้ยินลดลง', 'Animal bite',
       'Headachecough', 'เสียงดังในหู', 'ปัสสาวะบ่อย', 'ผมร่วง',
       'มีเสมหะคันจมูก', 'ไอกลางคืน', 'Runny nose', 'ปวดบ่าปวดหัวไหล่',
       'หายใจหอบเหนื่อยเหนื่อย', 'ไอมีเสมหะ', 'Sore throatStuffy nose',
       'exertion fatique', 'sneezing', 'Eye pain', 'Dry throat',
       'ปัสสาวะขุ่น', 'ปัสสาวะเป็นเลือดปัสสาวะแสบขัด', 'ปัสสาวะไม่สุด',
       'เจ็บเวลาปัสสาวะ', 'ปัสสาวะกะปริบกะปรอย', 'ปัสสาวะเล็ดราด',
       'ปวดข้อศอก', 'เคืองตาตาแดง', 'แขนบวม', 'ไข้คอแดง', 'คอแห้งคอแดง',
       'ร้อนวูบวาบ', 'คอแดง', 'ปวดซี่โครง', 'เรอเปรี้ยว',
       'pain on swallowing', 'difficulty swallowingSore throat',
       'difficulty speaking', 'Hoarseness', 'Stuffy ears', 'ก้อนที่ศีรษะ',
       'ก้อนที่รักแร้', 'มืออ่อนแรง', 'แขนอ่อนแรง', 'ปวดต้นคอปวดบ่า',
       'ปวดสีข้าง', 'น้ำตาไหล', 'เวียนศีรษะน้ำมูกไหล', 'ก้อนบริเวณใบหน้า',
       'ผิวแห้งผื่น', 'เจ็บหน้าอก', 'เหนื่อย', 'ปวดข้อนิ้วเท้า',
       'ปัสสาวะเป็นเลือด', 'มีเสมหะเจ็บคอ', 'มีเสมหะไอกลางคืน', 'ปวดขา',
       'bloody urine', 'ตาลายหน้ามืด', 'breathless on lying',
       'ปัสสาวะแสบขัด', 'จุกแน่นท้องปวดท้อง', 'Loss of appetite',
       'Weight lossstomachache', 'เจ็บคอน้ำมูกไหล', 'มองเห็นภาพซ้อน',
       'มีเสมหะจมูกตัน', 'บาดเจ็บ', 'เจ็บคอผื่น', 'แสบตา', 'ไอเจ็บคอ',
       'ปวดคอปวดหัวไหล่', 'ง่วงตลอดเวลา', 'ปวดขมับ', 'นอนไม่หลับ',
       'เครียด', 'รู้สึกไร้ค่า', 'วิตกกังวล', 'ทำร้ายตัวเอง',
       'เก็บตัวอยากตาย', 'joint pain', 'มือสั่น', 'อุจจาระลำเล็กลง',
       'ปวดมือ', 'ก้อนบริเวณขาหนีบ', 'คันคอเสมหะไหลลงคอ',
       'Eye irritation', 'ท้องอืดปวดท้อง', 'น้ำมูกไหลไข้', 'ไอน้ำมูกไหล',
       'nausea', 'รถล้ม', 'คันคอมีเสมหะ',
       'เจ็บเวลาปัสสาวะปัสสาวะเป็นเลือด', 'neck pain', 'ankle pain',
       'ปวดข้อนิ้วมือ', 'ไอเป็นเลือด', 'หายใจไม่ออก',
       'แน่นจมูกเลือดกำเดาไหล', 'ปวดกระบอกตา', 'ปวดหัวปวดท้ายทอย',
       'หายใจไม่สะดวกจมูกตัน', 'ปวดกราม', 'Hearing lossEar discharge',
       'genitals itching', 'Decreased stool caliber', 'ขาบวม', 'หูดับ',
       'น้ำมูกไหลจามบ่อย', 'ปวดกระบอกตาปวดหัว', 'กลืนติด',
       'เลือดกำเดาไหล', 'ถ่ายปนเลือด', 'itchy nosesneezing', 'จ้ำเลือด',
       'จุดเลือดออก', 'Blurry vision', 'เจ็บคอไอ', 'ร้อนวูบวาบไข้',
       'Lightheaded', 'DizzyBlack out', 'swaying',
       'วูบ หมดสติชั่วขณะเดินเซ', 'มึนศีรษะวูบ หมดสติชั่วขณะ',
       'วูบ หมดสติชั่วขณะ', 'มึนศีรษะหน้ามืด', 'เดินเซ', 'คิดวนซ้ำๆ',
       'คิดฆ่าตัวตาย']

yes_symptoms_text

yes_symptoms_answers = yes_symptoms.apply(lambda answers: [x['answers'] for x in answers])
yes_symptoms_answers = yes_symptoms_answers.apply(lambda x: [item for sublist in x for item in sublist])
yes_symptoms_answers = yes_symptoms_answers.to_frame(name="yes_symptoms_answers")
yes_symptoms_answers

yes_symptoms_text.to_frame(name="yes_symptoms_text")

# Convert search_term for each row to list.

df['search_term'] = df['search_term'].str.split(',')

df = df.drop('summary', axis=1)
df = pd.concat([df, yes_symptoms_text], axis=1)

df

le = LabelEncoder()

df['gender'] = le.fit_transform(df['gender'])

df

# Create search_term_mapping for TF_IDF.

search_term_list = []

for index, row in df.iterrows():
  text = ' '.join(row['search_term'])
  search_term_list.append(text)

search_term_list


# ----------------------------------------------------------------------------------
# Machine Learning part
# ----------------------------------------------------------------------------------

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from collections import Counter
from scipy.sparse import hstack
from pydantic import BaseModel

tfidf_vectorizer = None
knn_model = None
symptom_dict = {}

# Compile each search_term to every real symptoms.

def symptom_mapping(df):
    global symptom_mapping

    for index, row in df.iterrows():
        search_terms = row['search_term']
        symptoms = row['summary']

        for search_values in search_terms:
            if search_values not in symptom_dict:
                symptom_dict[search_values] = []
            symptom_dict[search_values].extend(symptoms)


# Use TF-IDF to turn words into numerical vectors.

def train_model(df):
    global tfidf_vectorizer, knn_model

    symptom_mapping(df)

    tfidf_vectorizer = TfidfVectorizer()
    tfidf_matrix = tfidf_vectorizer.fit_transform(search_term_list)

    demo_feature = []
    for index, row in df.iterrows():
        if row['gender'] == 1:
            gender = 1
        else:
            gender = 0
        age = row['age'] / 100.0
        demo_feature.append([gender, age])

    demo_features = np.array(demo_feature)

    # Combine patient profile with search_term_list.

    combined_features = hstack([tfidf_matrix, demo_features])

    # Train model.

    knn_model = NearestNeighbors(n_neighbors=5, metric='cosine')
    knn_model.fit(combined_features)

    print("Model training completed")

# Get the high similiar symptoms from symptoms_dict.

def get_recommendations_from_mapping(input_symptoms):
    recommendations = Counter()
    for symptom in input_symptoms:
        if symptom in symptom_dict:
            actual_symptoms = symptom_dict[symptom]
            for actual_symptom in actual_symptoms:
                if actual_symptom not in input_symptoms:
                    recommendations[actual_symptom] += 1
    return dict(recommendations)

# Get the high similiar symptoms from similiar patient.

def get_recommendations_from_similar_patients(input_symptoms, gender, age):
    global tfidf_vectorizer, knn_model
    if tfidf_vectorizer is None or knn_model is None:
        return {}

    input_text = ' '.join(input_symptoms)
    input_tfidf = tfidf_vectorizer.transform([input_text])

    if gender == 1:
       gender_feat = 1
    else: gender_feat = 0

    if age:
       age_feat = age / 100.0 
    else: age_feat = 0

    demo_feat = np.array([[gender_feat, age_feat]])

    input_combined = hstack([input_tfidf, demo_feat])

    distances, indices = knn_model.kneighbors(input_combined)

    recommendations = Counter()
    for idx in indices[0]:
        similar_patient = df.iloc[idx]
        for symptom in similar_patient['summary']:
            if symptom not in input_symptoms:
                recommendations[symptom] += 1

    return dict(recommendations)

# Run the recommendation systems and collect the high similair symptoms score.

def recommend_symptoms(input_symptoms, gender, age, top_n):
    mapping_scores = get_recommendations_from_mapping(input_symptoms)

    similarity_scores = get_recommendations_from_similar_patients(input_symptoms, gender, age)

    all_recommendations = Counter()
    for symptom, score in mapping_scores.items():
        all_recommendations[symptom] += score * 3
    for symptom, score in similarity_scores.items():
        all_recommendations[symptom] += score * 2

    top_recommendations = all_recommendations.most_common(top_n)
    return [symptom for symptom, _ in top_recommendations]

# The main function to describe input before run other functions.

def predict_symptoms(gender, age, symptoms, n_results=6):
    if gender.lower() == "male":
      gender = 1
    else: gender = 0
    symptoms = [symptoms]
    print(gender)
    print(age)
    print(symptoms)
    train_model(df)
    return recommend_symptoms(symptoms, gender, age, n_results)

class PredictRequest(BaseModel):
  gender:str
  age:int
  symptoms:str

