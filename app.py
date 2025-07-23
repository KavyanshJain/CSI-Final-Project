import streamlit as st
import pandas as pd
import joblib

# Load the trained model
model = joblib.load('model/credit_model.pkl')

# Define mappings for categorical variables
checking_status_map = {
    '< 0 DM': 'A11',
    '0 <= ... < 200 DM': 'A12',
    '>= 200 DM': 'A13',
    'no checking account': 'A14'
}
credit_history_map = {
    'no credits taken/all credits paid back duly': 'A30',
    'all credits at this bank paid back duly': 'A31',
    'existing credits paid back duly till now': 'A32',
    'delay in paying off in the past': 'A33',
    'critical account/other credits existing': 'A34'
}
purpose_map = {
    'car (new)': 'A40',
    'car (used)': 'A41',
    'furniture/equipment': 'A42',
    'radio/television': 'A43',
    'domestic appliances': 'A44',
    'repairs': 'A45',
    'education': 'A46',
    'retraining': 'A48',
    'business': 'A49',
    'others': 'A410'
}
savings_status_map = {
    '< 100 DM': 'A61',
    '100 <= ... < 500 DM': 'A62',
    '500 <= ... < 1000 DM': 'A63',
    '>= 1000 DM': 'A64',
    'unknown/no savings account': 'A65'
}
employment_map = {
    'unemployed': 'A71',
    '< 1 year': 'A72',
    '1 <= ... < 4 years': 'A73',
    '4 <= ... < 7 years': 'A74',
    '>= 7 years': 'A75'
}
personal_status_map = {
    'male: divorced/separated': 'A91',
    'female: divorced/separated/married': 'A92',
    'male: single': 'A93',
    'male: married/widowed': 'A94',
    'female: single': 'A95'
}
other_parties_map = {
    'none': 'A101',
    'co-applicant': 'A102',
    'guarantor': 'A103'
}
property_magnitude_map = {
    'real estate': 'A121',
    'building society savings/life insurance': 'A122',
    'car or other': 'A123',
    'unknown/no property': 'A124'
}
other_payment_plans_map = {
    'bank': 'A141',
    'stores': 'A142',
    'none': 'A143'
}
housing_map = {
    'rent': 'A151',
    'own': 'A152',
    'for free': 'A153'
}
job_map = {
    'unemployed/unskilled non-resident': 'A171',
    'unskilled resident': 'A172',
    'skilled employee/official': 'A173',
    'management/self-employed/highly qualified': 'A174'
}
own_telephone_map = {
    'none': 'A191',
    'yes': 'A192'
}
foreign_worker_map = {
    'yes': 'A201',
    'no': 'A202'
}

# Streamlit app
st.title('Creditworthiness Prediction App')
st.write('Enter the financial attributes to predict creditworthiness.')

# Input fields
with st.form('input_form'):
    checking_status = st.selectbox('Checking Status', list(checking_status_map.keys()))
    duration = st.number_input('Duration (months)', min_value=1, max_value=100, value=12)
    credit_history = st.selectbox('Credit History', list(credit_history_map.keys()))
    purpose = st.selectbox('Purpose', list(purpose_map.keys()))
    credit_amount = st.number_input('Credit Amount (DM)', min_value=100, max_value=20000, value=1000)
    savings_status = st.selectbox('Savings Status', list(savings_status_map.keys()))
    employment = st.selectbox('Employment', list(employment_map.keys()))
    installment_rate = st.number_input('Installment Rate (% of disposable income)', min_value=1, max_value=4, value=2)
    personal_status = st.selectbox('Personal Status and Sex', list(personal_status_map.keys()))
    other_parties = st.selectbox('Other Parties', list(other_parties_map.keys()))
    residence_since = st.number_input('Residence Since (years)', min_value=1, max_value=10, value=2)
    property_magnitude = st.selectbox('Property Magnitude', list(property_magnitude_map.keys()))
    age = st.number_input('Age (years)', min_value=18, max_value=100, value=30)
    other_payment_plans = st.selectbox('Other Payment Plans', list(other_payment_plans_map.keys()))
    housing = st.selectbox('Housing', list(housing_map.keys()))
    existing_credits = st.number_input('Existing Credits at Bank', min_value=1, max_value=10, value=1)
    job = st.selectbox('Job', list(job_map.keys()))
    num_dependents = st.number_input('Number of Dependents', min_value=1, max_value=10, value=1)
    own_telephone = st.selectbox('Own Telephone', list(own_telephone_map.keys()))
    foreign_worker = st.selectbox('Foreign Worker', list(foreign_worker_map.keys()))

    submitted = st.form_submit_button('Predict')

if submitted:
    # Create input DataFrame
    input_data = pd.DataFrame({
        'checking_status': [checking_status_map[checking_status]],
        'duration': [np.log1p(duration)],  # Apply log transformation
        'credit_history': [credit_history_map[credit_history]],
        'purpose': [purpose_map[purpose]],
        'credit_amount': [np.log1p(credit_amount)],  # Apply log transformation
        'savings_status': [savings_status_map[savings_status]],
        'employment': [employment_map[employment]],
        'installment_rate': [installment_rate],
        'personal_status': [personal_status_map[personal_status]],
        'other_parties': [other_parties_map[other_parties]],
        'residence_since': [residence_since],
        'property_magnitude': [property_magnitude_map[property_magnitude]],
        'age': [age],
        'other_payment_plans': [other_payment_plans_map[other_payment_plans]],
        'housing': [housing_map[housing]],
        'existing_credits': [existing_credits],
        'job': [job_map[job]],
        'num_dependents': [num_dependents],
        'own_telephone': [own_telephone_map[own_telephone]],
        'foreign_worker': [foreign_worker_map[foreign_worker]]
    })

    # Rename columns to match training data
    input_data = input_data.rename(columns={'credit_amount': 'log_credit_amount', 'duration': 'log_duration'})

    # Make prediction
    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0][1]

    # Display result
    st.subheader('Prediction Result')
    if prediction == 0:
        st.success('Creditworthy')
    else:
        st.error('Not Creditworthy')
    st.write(f'Probability of Bad Credit: {probability:.2f}')
