# mlZoomcampMidtermProject

You can get dataset here https://www.kaggle.com/datasets/iamsouravbanerjee/heart-attack-prediction-dataset/data 

The Heart Attack Risk Prediction Dataset serves as a valuable resource for delving into the intricate dynamics of heart health and its predictors. Heart attacks, or myocardial infarctions, continue to be a significant global health issue, necessitating a deeper comprehension of their precursors and potential mitigating factors. This dataset encapsulates a diverse range of attributes including age, cholesterol levels, blood pressure, smoking habits, exercise patterns, dietary preferences, and more, aiming to elucidate the complex interplay of these variables in determining the likelihood of a heart attack. 

To see the construction stages of the project, download the "heart_attack_prediction_dataset.csv" and "mlz_midterm.ipynb" files and put them in the same folder. Then run mlz_midterm.ipynb in jupyter notebook.

First we read the data. We delete the "Patient ID" column, which is specific to each row and therefore will not be of any use to us in data analysis.

The "Blood Pressure" column is listed as "Systolic"/"Diastolic" in the dataset. We split this into 2 parts. We delete the now unnecessary "Blood Pressure" column.

To make the data set easier to use, we replaced spaces with _ and converted uppercase letters to lowercase letters.

We look at the unique values in each column and how many unique values there were in total.

Since there are no NaN values in the data set, we did not bother with these values. But if it were NaN value, we would have to find out which of our model would be more successful by doing df2 = df.fillna(value=0) and df3 = df.fillna(df.mean()).

We get an idea if df.describe() is an outlier. We need to pay attention to values that deviate significantly from the average. We also need to better understand the dataset by visualizing each column.

We convert our target column to numerical values.

We identify columns containing numerical and categorical values in the data set. We will look at the relationship between these.

With mutual_info_score, we look at categorical columns and remove low correlation columns from our data set. Then we look at the numerical columns with the correlation matrix and remove the low correlation columns from our data set.

We divide our data set into 0.6 train, 0.2 validation and 0.2 test.

We make encoding with DictVectorizer. Then we try our binary classification algorithms one by one. In this process, we try to determine which model is the most successful by optimizing the parameters and, we are trying to make our parameter selection process better for ourselves with visualizations.

Finally, we train our model with the most appropriate parameters with full_train, which is the combination of train and validation data sets.







