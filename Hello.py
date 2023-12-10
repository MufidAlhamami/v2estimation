import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np
from xgboost import XGBRegressor
import streamlit as st

st.set_page_config(page_title="Tradinos", page_icon=":bar_chart:", layout="wide")




google_sheets_link = 'https://docs.google.com/spreadsheets/d/1HuM1aFhv68nK49AaRxQ9R0I5cwZ4dKEjd_-H9h9nWeA/export?format=csv'
df = pd.read_csv(google_sheets_link)

# Assuming df is your DataFrame and 'ColumnName' is the column you're interested in
business_domain_values = df['Business Domain'].unique()
# Convert the unique values array to a list if needed
business_domain_values_list = list(business_domain_values)

# Assuming df is your DataFrame and 'ColumnName' is the column you're interested in
service_values = df['Service'].unique()
# Convert the unique values array to a list if needed
service_values_list = list(service_values)
print(service_values_list)

# Assuming df is your DataFrame and 'ColumnName' is the column you're interested in
Deal_value = df['Deal value'].unique()
# Convert the unique values array to a list if needed
Deal_value_list = list(Deal_value)

# Assuming df is your DataFrame and 'ColumnName' is the column you're interested in
Branch_value = df['Branch'].unique()
# Convert the unique values array to a list if needed
Branch_list = list(Branch_value)

# Assuming df is your DataFrame and 'ColumnName' is the column you're interested in
Country_values = df['Country'].unique()
# Convert the unique values array to a list if needed
Country_list = list(Country_values)

# Assuming df is your DataFrame and 'ColumnName' is the column you're interested in
City_values = df['City'].unique()
# Convert the unique values array to a list if needed
City_list = list(City_values)

# Assuming df is your DataFrame and 'ColumnName' is the column you're interested in
Tags_values = df['Tags'].unique()
# Convert the unique values array to a list if needed
Tags_list = list(Tags_values)

# Assuming df is your DataFrame and 'ColumnName' is the column you're interested in
Type_values = df['Type'].unique()
# Convert the unique values array to a list if needed
Type_list = list(Type_values)

# Assuming df is your DataFrame and 'ColumnName' is the column you're interested in
Source_values = df['Source'].unique()
# Convert the unique values array to a list if needed
Source_list = list(Source_values)






# Convert 'Deal value' column to numeric after removing commas and handling decimals
df['Deal value'] = df['Deal value'].replace('[\$,]', '', regex=True).astype(float)

# Encode categorical variables using LabelEncoder
label_encoder1 = preprocessing.LabelEncoder()
df["Business Domain"] = label_encoder1.fit_transform(df['Business Domain'])

label_encoder2 = preprocessing.LabelEncoder()
df["Service"] = label_encoder2.fit_transform(df['Service'])

label_encoder3 = preprocessing.LabelEncoder()
df["Branch"] = label_encoder3.fit_transform(df['Branch'])

label_encoder4 = preprocessing.LabelEncoder()
df["Country"] = label_encoder4.fit_transform(df['Country'])

label_encoder5 = preprocessing.LabelEncoder()
df["City"] = label_encoder5.fit_transform(df['City'])

label_encoder6 = preprocessing.LabelEncoder()
df["Type"] = label_encoder6.fit_transform(df['Type'])

label_encoder7 = preprocessing.LabelEncoder()
df["Source"] = label_encoder7.fit_transform(df['Source'])

label_encoder8 = preprocessing.LabelEncoder()
df["Tags"] = label_encoder8.fit_transform(df['Tags'])


###############################################--Training chart--#####################################################

def cu_p(df):
    # Remove commas and convert the 'AVG Estimation from DU by hours' column to numeric
    df['AVG Estimation from DU by hours'] = pd.to_numeric(df['AVG Estimation from DU by hours'].str.replace(',', ''))

    # Split the data into training and testing sets (80% train, 20% test)
    X = df.drop('AVG Estimation from CU by hours', axis=1)  # Features
    y = df['AVG Estimation from CU by hours']  # Target variable

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Initialize the XGBoost model
    model = XGBRegressor()
    # Train the model on the training set
    model.fit(X_train, y_train)
    # Predict on the test set
    y_pred = model.predict(X_test)

    # Create a DataFrame for visualization
    visualization_df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})

    # Plot the bar chart
    bar_width = 0.35
    index = np.arange(len(visualization_df))

    fig, ax = plt.subplots(figsize=(6, 2))
    ax.bar(index, visualization_df['Actual'], bar_width, label='Actual', color='b')
    ax.bar(index + bar_width, visualization_df['Predicted'], bar_width, label='Predicted', color='r')

    # Calculate the percentage difference in errored hours
    hours_remain = ((visualization_df['Predicted'].sum() - visualization_df['Actual'].sum()) / visualization_df[
        'Predicted'].sum()) * 100
    st.write(f"## The predicted and actual times for the difference in errored hours: {hours_remain:.2f}%")

    ax.set_xlabel('Sample Index')
    ax.set_ylabel('AVG Estimation from DU by hours')
    ax.set_title('Actual vs. Predicted DU' if who == 'DU' else 'Actual vs. Predicted CU')
    ax.set_xticks(index + bar_width / 2)
    ax.set_xticklabels(visualization_df.index)
    ax.legend()
    st.pyplot(fig)

def du_p(df):
    # Remove commas from the 'AVG Estimation from DU by hours' column and convert to numeric
    df['AVG Estimation from DU by hours'] = pd.to_numeric(df['AVG Estimation from DU by hours'].str.replace(',', ''),
                                                          errors='coerce')

    # Drop unnecessary columns
    df = df.drop(['AVG Estimation from CU by hours', 'Deal value'], axis=1)

    # Drop rows with missing values (NaN) resulting from the conversion
    df = df.dropna()

    # Split the data into training and testing sets (80% train, 20% test)
    X = df.drop('AVG Estimation from DU by hours', axis=1)  # Features
    y = df['AVG Estimation from DU by hours']  # Target variable
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Initialize the XGBoost model
    model = XGBRegressor()

    # Train the model on the training set
    model.fit(X_train, y_train)

    # Predict on the test set
    y_pred = model.predict(X_test)

    # Create a DataFrame for visualization
    visualization_df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})

    # Plot the bar chart
    bar_width = 0.35
    index = np.arange(len(visualization_df))

    fig, ax = plt.subplots(figsize=(6, 2))
    ax.bar(index, visualization_df['Actual'], bar_width, label='Actual', color='b')
    ax.bar(index + bar_width, visualization_df['Predicted'], bar_width, label='Predicted', color='r')

    # Calculate the percentage difference in errored hours
    hours_remain = ((visualization_df['Predicted'].sum() - visualization_df['Actual'].sum()) / visualization_df[
        'Predicted'].sum()) * 100
    st.write(f"## The predicted and actual times for the difference in errored hours: {hours_remain:.2f}%")

    ax.set_xlabel('Sample Index')
    ax.set_ylabel('AVG Estimation from DU by hours')
    ax.set_title('Actual vs. Predicted DU' if who == 'DU' else 'Actual vs. Predicted CU')
    ax.set_xticks(index + bar_width / 2)
    ax.set_xticklabels(visualization_df.index)
    ax.legend()

    st.pyplot(fig)

##################################################################################################################

def get_user_input():
    col1, col2 = st.columns(2)
    with col1:
        st.write(" ## Choose a service:")
        estimate_service = st.radio("", tuple(service_values_list))
    with col2:
        st.write("## Choose a business domain:")
        estimate_business_domain = st.radio("", tuple(business_domain_values_list))

    col3, col4 = st.columns(2)
    with col3:
        st.write("## Choose a Branch:")
        estimate_Branch = st.radio("", tuple(Branch_list))
    with col4:
        st.write(" ## Choose a Country:")
        estimate_Country = st.radio("", tuple(Country_list))

    col5, col6 = st.columns(2)
    with col5:
        st.write("## Choose a City:")
        estimate_City = st.radio("", tuple(City_list))
    with col6:
        st.write("## Choose a Tags:")
        estimate_Tags = st.radio("", tuple(Tags_list))

    col7, col8 = st.columns(2)
    with col7:
        st.write("## Choose a Type:")
        estimate_Type = st.radio("", tuple(Type_list))
    with col8:
        st.write("## Choose a Source:")
        estimate_Source = st.radio("", tuple(Source_list))

    return estimate_service, estimate_business_domain, estimate_Branch, estimate_Country, estimate_City, estimate_Tags, estimate_Type, estimate_Source

def predict_time(model, X_test):
    y_pred = model.predict(X_test)
    y_pred = int(y_pred[0])
    y_pred = round(y_pred, 2)
    st.write(" # Estimated time: ", y_pred)


def cu_e(df):
    # Remove commas and convert the 'AVG Estimation from DU by hours' column to numeric
    df['AVG Estimation from DU by hours'] = pd.to_numeric(df['AVG Estimation from DU by hours'].str.replace(',', ''))

    # Drop unnecessary columns
    df = df.drop(['AVG Estimation from DU by hours', 'Deal value'], axis=1)

    # Split the data into training and testing sets (80% train, 20% test)
    X = df.drop('AVG Estimation from CU by hours', axis=1)  # Features
    y = df['AVG Estimation from CU by hours']  # Target variable
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Initialize the XGBoost model
    model = XGBRegressor()

    # Train the model on the training set
    model.fit(X_train, y_train)
    # Predict on the test set
    y_pred = model.predict(X_test)


    new_val1 = label_encoder1.transform([estimate_business_domain])
    new_val2 = label_encoder2.transform([estimate_service])
    new_val3 = label_encoder3.transform([estimate_Branch])
    new_val4 = label_encoder4.transform([estimate_Country])
    new_val5 = label_encoder5.transform([estimate_City])
    new_val6 = label_encoder6.transform([estimate_Type])
    new_val7 = label_encoder7.transform([estimate_Source])
    new_val8 = label_encoder8.transform([estimate_Tags])


    test = np.array([new_val1[0], new_val2[0], new_val3[0], new_val4[0], new_val5[0], new_val6[0], new_val7[0], new_val8[0]])
    predict_time(model, test.reshape((1, 8)))

    return estimate_service, estimate_business_domain, estimate_Branch, estimate_Country, estimate_City, estimate_Tags, estimate_Type, estimate_Source

def du_e(df):
    # Remove commas from the 'AVG Estimation from DU by hours' column and convert to numeric
    df['AVG Estimation from DU by hours'] = pd.to_numeric(df['AVG Estimation from DU by hours'].str.replace(',', ''),
                                                          errors='coerce')

    # Drop unnecessary columns
    df = df.drop(['AVG Estimation from CU by hours', 'Deal value'], axis=1)

    # Drop rows with missing values (NaN) resulting from the conversion
    df = df.dropna()

    # Split the data into training and testing sets (80% train, 20% test)
    X = df.drop('AVG Estimation from DU by hours', axis=1)  # Features
    y = df['AVG Estimation from DU by hours']  # Target variable
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Initialize the XGBoost model
    model = XGBRegressor()

    # Train the model on the training set
    model.fit(X_train, y_train)
    # Predict on the test set
    y_pred = model.predict(X_test)


    new_val1 = label_encoder1.transform([estimate_business_domain])
    new_val2 = label_encoder2.transform([estimate_service])
    new_val3 = label_encoder3.transform([estimate_Branch])
    new_val4 = label_encoder4.transform([estimate_Country])
    new_val5 = label_encoder5.transform([estimate_City])
    new_val6 = label_encoder6.transform([estimate_Type])
    new_val7 = label_encoder7.transform([estimate_Source])
    new_val8 = label_encoder8.transform([estimate_Tags])


    test = np.array([new_val1[0], new_val2[0], new_val3[0], new_val4[0], new_val5[0], new_val6[0], new_val7[0], new_val8[0]])
    predict_time(model, test.reshape((1, 8)))


    return estimate_service, estimate_business_domain, estimate_Branch, estimate_Country, estimate_City, estimate_Tags, estimate_Type, estimate_Source

def price(df):
    df = df.drop(['AVG Estimation from CU by hours', 'AVG Estimation from DU by hours'], axis=1)

    # df['Deal value'] = type(df['Deal value'], int)
    # Split the data into training and testing sets (80% train, 20% test)
    X = df.drop('Deal value', axis=1)  # Features
    y = df['Deal value']  # Target variable
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Initialize the XGBoost model
    model = XGBRegressor()

    # Train the model on the training set
    model.fit(X_train, y_train)
    # Predict on the test set
    y_pred = model.predict(X_test)


    new_val1 = label_encoder1.transform([estimate_business_domain])
    new_val2 = label_encoder2.transform([estimate_service])
    new_val3 = label_encoder3.transform([estimate_Branch])
    new_val4 = label_encoder4.transform([estimate_Country])
    new_val5 = label_encoder5.transform([estimate_City])
    new_val6 = label_encoder6.transform([estimate_Type])
    new_val7 = label_encoder7.transform([estimate_Source])
    new_val8 = label_encoder8.transform([estimate_Tags])


    test = np.array([new_val1[0], new_val2[0], new_val3[0], new_val4[0], new_val5[0], new_val6[0], new_val7[0], new_val8[0]])
    y_pred = model.predict(test.reshape((1, 8)))
    y_pred = int(y_pred[0])
    y_pred = round(y_pred, 2)
    st.write(" # Estimated price: ", y_pred)

    return estimate_service, estimate_business_domain, estimate_Branch, estimate_Country, estimate_City, estimate_Tags, estimate_Type, estimate_Source



##################################################################################################################

st.write("# Estimation time")


# Define the link
google_sheets_link = "https://docs.google.com/spreadsheets/d/1HuM1aFhv68nK49AaRxQ9R0I5cwZ4dKEjd_-H9h9nWeA/edit?usp=sharing"
note_text = (
    "### If you encounter a "
    '<span style="color: #FF4B4B;">ValueError</span>, please check this '
    'link:'
)
# Make the link clickable
link_text = f'<a href="{google_sheets_link}" target="_blank">Check the Google Sheets link</a>'
full_note = f"{note_text} {link_text}"
st.markdown(full_note, unsafe_allow_html=True)
st.write("### And ensure that the V2 page is set as the first page")



# Use a radio button to choose between viewing charts and estimating hours
what_to_do = st.radio("Choose an option:", ('View training, actual, and predicted charts', 'Estimate hours and price'))

if what_to_do == 'View training, actual, and predicted charts':
    who = st.radio("Training for CU or DU?", ('CU', 'DU'))
    if who == 'CU':
        cu_p(df)
    elif who == 'DU':
        du_p(df)



elif what_to_do == 'Estimate hours and price':
    who = st.radio("Estimation for CU or DU?", ('CU', 'DU'))
    estimate_service, estimate_business_domain, estimate_Branch, estimate_Country, estimate_City, estimate_Tags, estimate_Type, estimate_Source = get_user_input()
    if who == 'CU':
        estimate_service, estimate_business_domain, estimate_Branch, estimate_Country, estimate_City, estimate_Tags, estimate_Type, estimate_Source = cu_e(df)
        price(df)
    elif who == 'DU':
        estimate_service, estimate_business_domain, estimate_Branch, estimate_Country, estimate_City, estimate_Tags, estimate_Type, estimate_Source = du_e(df)
        price(df)
