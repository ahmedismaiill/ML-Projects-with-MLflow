import mlflow 
import joblib
import seaborn as sns
import matplotlib.pyplot as plt
from plotly.subplots import make_subplots
from sklearn.metrics import  ConfusionMatrixDisplay,confusion_matrix, classification_report , accuracy_score ,f1_score,recall_score,precision_score
from mlflow.models.signature import infer_signature
import warnings
warnings.filterwarnings('ignore')
#-----------Load Saved Model ------------
Xgboost_model = joblib.load('XGBoost_model.pkl')
x_test = joblib.load('x_test.pkl')
y_test = joblib.load('y_test.pkl')

#----------- MLflow ---------------
mlflow.set_experiment("Heart Failure") 
with mlflow.start_run() as run :
   
   #-------------Model Predict-------------
   y_predict = Xgboost_model.predict(x_test)
   
   #-------------Model Matrics------------------
   Xgboost_score=Xgboost_model.score(x_test,y_test)
   acc_test = accuracy_score(y_test , y_predict)
   precision=precision_score(y_test , y_predict)
   recall=recall_score(y_test , y_predict)
   f1 = f1_score(y_test , y_predict)
   
   #---------Log Model Parameters------------
   params = Xgboost_model.get_params()
   filtered_params = {k: v for k, v in params.items() if v is not None}
   mlflow.log_params(filtered_params)
   
   #---------- Log Matrics --------------
   mlflow.log_metrics({
   "Model_Score": round(Xgboost_score, 3),
   'Precision': round(precision, 3),
   'Recall': round(recall, 3),
   'F1-Score': round(f1, 3)
                              })
   
   #----------Add Classification Report----------------
   report = classification_report(y_test, y_predict)
   with open("classification_report.txt", "w") as f:
      f.write(report)
   mlflow.log_artifact("classification_report.txt", artifact_path="reports")
   
   #--------Add Features And Output ---------------
   input_example = x_test.iloc[:1]
   predicted_example = Xgboost_model.predict(input_example)
   signature = infer_signature(input_example, predicted_example)
   
   #----------Add Confusion Matrix -------------
   plt.figure(figsize=(10,6))
   sns.heatmap(confusion_matrix(y_test , y_predict) ,annot=True ,cbar=False , fmt='g' , cmap='Blues' )
   plt.title('Xgboost Confusion Matrix')
   conf_matrix_fig = plt.gcf()
   mlflow.log_figure(conf_matrix_fig, "plots/Xgboost Confusion Matrix.png")
   plt.close()
   
   #----------Log Model -----------
   mlflow.xgboost.log_model(
      Xgboost_model,
      artifact_path="model",
      signature=signature,
      input_example = input_example
      )
   
   #----------Add Model PKL File ----------- 
   mlflow.log_artifact("XGBoost_model.pkl", artifact_path="pkl_model")
   
   report = classification_report(y_test, y_predict)
   with open("classification_report.txt", "w") as f:
      
      f.write(report)
   mlflow.log_artifact("classification_report.txt", artifact_path="reports")