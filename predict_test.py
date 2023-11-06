
import requests

url = "http://localhost:9696/predict"
customer_data = {'age': 67,
 'cholesterol': 208,
 'heart_rate': 72,
 'diabetes': 0,
 'family_history': 0,
 'smoking': 1,
 'obesity': 0,
 'alcohol_consumption': 0,
 'exercise_hours_per_week': 4.168188835442079,
 'medication_use': 0,
 'stress_level': 9,
 'sedentary_hours_per_day': 6.61500145291406,
 'income': 261404,
 'triglycerides': 286,
 'physical_activity_days_per_week': 0,
 'sleep_hours_per_day': 6,
 'country': 'argentina',
 'systolic': 158,
 'diastolic': 88}

# really answer = 0

response = requests.post(url, json=customer_data)

if response.status_code == 200:
    result = response.json()
    if result["heart attack prediction"] == True:
        print("There is heart attack risk.")
    else:
        print("There is no heart attack risk.")
else:
    print("Error:", response.status_code)


