# Sales_prediction_model:
deployed here: https://bigsalesprediction.streamlit.app/

The dataset consists of two CSV files named Train.csv and Test.csv. The first ten rows of the training set are displayed with train_df.head(10). The information about each column is obtained via train_df.info(). A loop iterates over all columns to find their unique values. Descriptive statistics are computed using train_df.describe().
Correlation Analysis
A correlation matrix between relevant features such as Item_Weight, Item_Visibility, Item_MRP, and Item_Outlet_Sales is created using Pearson's method. This helps identify relationships among these variables.
Visualization
Heatmaps showing correlations and pair plots comparing outlet types are used to visualize the data. These help reveal patterns that may aid feature engineering and model selection.
Feature Engineering
Some preliminary steps include handling missing values, encoding categorical variables like Outlet_Size, Outlet_Location_Type, and Item_Fat_Content, and creating new derived features such as Item_Type_Category.
Model Selection
Various regression models including linear regression, ridge regression, random forest regression, and k-nearest neighbors regression are imported and tested. TensorFlow and its Keras API are also utilized to build neural network architectures consisting of dense layers, LSTMs, and dropouts.
Preprocessing
Data normalization is performed using MinMaxScaler. Additionally, label encoding is applied to convert categorical labels into numerical ones.
Metrics Computation
Model performance is evaluated using Mean Squared Error (MSE), which provides insight into how well the predictions match the actual values.
Overall, this notebook demonstrates a systematic approach to building predictive models involving exploratory analysis, feature engineering, and model evaluation.
