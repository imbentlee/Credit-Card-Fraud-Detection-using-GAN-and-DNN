# Credit Card Fraud Detection using GAN and DNN

## TAKE NOTE OF THE FOLLOWING:
The resource folder containing the creditcard.csv file cannot be uploaded as it is too large.

It can be downloaded from https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud
Credit to Machine Learning Group ULB for the dataset which I retreived off Kaggle.

To ensure the code runs as expected, ensure the 2 files 'Credit_Card_Fraud_Detection_using_GAN_and_DNN.ipynb' and 'fraud_app.py' are placed at the same level along with a 'resources' folder containing the 'creditcard.csv' file.
Alternatively, you can change the code in code cell 3 of the .ipynb file from 'df = pd.read_csv("resources/creditcard.csv")' to 'df = pd.read_csv("creditcard.csv")' and save the creditcard.csv at the same level as the other 2 files, without a resources folder.

### How to run the code:
1. Simply ensure the files are saved to the correct levels:
- 'Credit_Card_Fraud_Detection_using_GAN_and_DNN.ipynb' and 'fraud_app.py' must be the same level
- Create a folder called 'resources' and place it at the same level as 'Credit_Card_Fraud_Detection_using_GAN_and_DNN.ipynb' and 'fraud_app.py'
- creditcard.csv should be saved in the 'resources' folder (the prior NOTE gives an alternative to creating a 'resources' folder)

2. Run the full Jupyter Notebook file
3. Once the file completes its run a Streamlit web application should popup in your browser where you can test the application using the csv files created through the Jupyter Notebook
4. If the web application does not run, simply run the 'fraud_app.py' file via the terminal.

*Download and open the 'Credit-Card-Fraud-Detection-using-GAN-and-DNN.html' file on your browser to see the expected result after running the Jupyter Notebook.*
