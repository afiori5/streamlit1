import streamlit as st 
import pandas as pd
import numpy as np
import altair as alt
import sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

st.markdown("LinkedIn User Prediction App")

"Please select the options that apply to you."
Income = st.selectbox(label="Household Income",
options=("$10,000 to $20,000", 
"$20,000 to $30,000", 
"$40,000 to $50,000", 
"$50,000 to $75,000", 
"$100,000 to $150,000", 
"$150,000+", 
"Don't Know"))

Age = st.slider(label="Enter Your Age", 
        min_value=1,
        max_value=100,
        value=50)

education = st.selectbox(label="Highest Level of Education",
options=("Some High School",  
"High School Graduate", 
"Some College, no Degree", 
"Two -year College or University", 
"Bachelor's degree", 
"Postgraduate Degree"))

Marital = st.selectbox(label="Are you married?",
options=("Yes", 
"No"))


Parent = st.selectbox(label="Are you the parent of a child under the age of 18?",
options=("Yes", 
"No"))

female = st.selectbox(label="Are you a male or female?",
options=("Female", 
"Male"))



# #### 1. Read in the data, call the dataframe "s"  and check the dimensions of the dataframe

s = pd.read_csv("social_media_usage.csv")
s.head()


print(s.shape)


# #### 2. Define a function called clean_sm that takes one input, x, and uses `np.where` to check whether x is equal to 1. If it is, make the value of x = 1, otherwise make it 0. Return x. Create a toy dataframe with three rows and two columns and test your function to make sure it works as expected

def clean_sm(x):
    return np.where(x ==1,
             1,
             0)


df1 = pd.DataFrame({'col':[1,2,3],'col2':[4,5,6]})
print(df1)

clean_sm(df1) #function returns an array with the expected values


# #### 3. Create a new dataframe called "ss". The new dataframe should contain a target column called sm_li which should be a binary variable ( that takes the value of 1 if it is 1 and 0 otherwise (use clean_sm to create this) which indicates whether or not the individual uses LinkedIn, and the following features: income (ordered numeric from 1 to 9, above 9 considered missing), education (ordered numeric from 1 to 8, above 8 considered missing), parent (binary), married (binary), female (binary), and age (numeric, above 98 considered missing). Drop any missing values. Perform exploratory analysis to examine how the features are related to the target.

ss = pd.DataFrame({
    "sm_li":np.where(s["web1h"] == 1, 1, 0),
    "Income":np.where(s["income"]==9, np.nan, s["income"]),
    "Age":np.where(s["age"]>98, np.nan, s["age"]),
    "education":np.where(s["educ2"] >8,np.nan, s["educ2"]),
    "Marital":np.where(s["marital"] == 1, 1, 0),
    "Parent":np.where(s["par"]==1,1,0),
    "female":np.where(s["gender"] == 2, 1, 0)})


ss.head()


# #### 4. Create a target vector (y) and feature set (X)

ss  = ss.dropna()


y = ss["sm_li"]
x = ss[["Income", "Age", "education", "Marital", "Parent", "female"]]


# #### 5. Split the data into training and test sets. Hold out 20% of the data for testing. Explain what each new object contains and how it is used in machine learning


x_train, x_test, y_train, y_test = train_test_split(x,
                                          y,
                                          stratify=y,
                                          test_size=0.2,
                                          random_state=987)


# #### 6. Instantiate a logistic regression model and set class_weight to balanced. Fit the model with the training data.


lr = LogisticRegression(class_weight="balanced")



lr.fit(x_train,y_train)


# #### 7. Evaluate the model using the testing data. What is the model accuracy for the model? Use the model to make predictions and then generate a confusion matrix from the model. Interpret the confusion matrix and explain what each number means.


y_pred = lr.predict(x_test)


# #### 8. Create the confusion matrix as a dataframe and add informative column names and index names that indicate what each quadrant represents


confusion_matrix(y_test,y_pred)



pd.DataFrame(confusion_matrix(y_test,y_pred),
            columns=["Predicted Negative","Predicted Positive"],
            index=["Actual Negative","Actual Positive"]).style.background_gradient(cmap="PiYG")



print(classification_report(y_test,y_pred))


# #### 10. Use the model to make predictions. For instance, what is the probability that a high income (e.g. income=8), with a high level of education (e.g. 7), non-parent who is married female and 42 years old uses LinkedIn? How does the probability change if another person is 82 years old, but otherwise the same?


newdata = pd.DataFrame({
    "Income": [1,8,8],
    "Age": [12,42,82],
    "education":[1,7,1],
    "Marital":[0,1,1],
    "Parent": [0,0,0],
    "female": [0,1,1],
})


newdata["sm_li"] = lr.predict(newdata)


newdata

st.write(f"This person is {mar_label}, a {deg_label}, and in a {inc_label} bracket")



