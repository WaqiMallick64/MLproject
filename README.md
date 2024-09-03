## ML PROJECT

This project is a machine learning pipeline designed to predict math scores based on various features like gender, race/ethnicity, parental level of education, lunch type, and test preparation course. The project includes data ingestion, data transformation, model training, and prediction components, all wrapped in a simple web application using Flask.

## INSTALLATION

Clone the repository: git clone https://github.com/WaqiMallick64/MLproject.git
Navigate to the project directory: "cd MLproject"
Install the required packages: "pip install -r requirements.txt"

## USAGE

Start the Flask web application to use the prediction interface: "python app.py" 

## COMPONENTS

## Data Ingestion :
The data ingestion component reads the dataset (StudentsPerformance.csv) and splits it into training and testing datasets. The processed data is stored in the artifacts folder as train.csv and test.csv.

## Data Transformation :
This component transforms the data by handling missing values, encoding categorical variables, and scaling numerical features. It uses OneHotEncoder and StandardScaler from sklearn to preprocess the data. The transformed data is saved as preprocessor.pkl in the artifacts folder.

## Model Training
The model training component trains multiple regression models (Random Forest, Decision Tree, Gradient Boosting, Linear Regression, XGBRegressor, CatBoosting Regressor, AdaBoost Regressor) and selects the best model based on R2 score. The trained model is saved as model.pkl in the artifacts folder.

## Prediction Pipeline
The prediction pipeline loads the trained model and preprocessor to make predictions based on user input data. It transforms the input data using the preprocessor and uses the trained model to predict the math score.

## Web Application
The project includes a simple Flask web application that provides a user interface for inputting data and predicting math scores. The app is hosted on localhost and can be accessed by navigating to http://127.0.0.1:5000/ in a web browser.

## LOGGING AND ERROR HANDLING
## Logging:
 The project uses Python's logging module to log important events and errors. Logs are saved in the logs directory.

## Custom Exception Handling:
 Custom exceptions are defined in exception.py to handle errors gracefully and provide meaningful error messages.
