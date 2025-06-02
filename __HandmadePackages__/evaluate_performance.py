# Metriche di performance
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score,confusion_matrix

import numpy as np
import pandas as pd

def performance_binary(test_y : np.ndarray,
                       predizioni: np.ndarray,
                       verbose : bool = False,
                       labels_float : dict = {}
                      ) -> dict : 
   """Funzione per ottenere metriche di performance dati i valori veri per la variabile target y 
      ed i valori predetti dal modello di classificazione.

      Prende in input:
         - test_y (np.ndarray): valori veri variabile target
         - predizioni (np.ndarray): valori predetti variabile target
         - verbose (bool) opzionale: avere dei print o no
         - labels_float (dict) opzionale: inserire un dizionario per avere degli output più comprensibili.
                                          Sostituisce i numeri delle classi con i loro nomi   

      Output:
         - df_report (pd.DataFrame): metriche di performance per ogni classe e globali
         - df_report_cm (pd.DataFrame): metriche di performance per la classe positiva (codifica 1)
   """

   if not isinstance(verbose,bool):
      raise TypeError("verbose deve essere un booleano")


   # Ottenere report con varie metriche di performance
   report_scikit_learn = classification_report(test_y, predizioni,output_dict=True)

   # Trasformare il report di perfomance in un formato più leggibile (pandas dataframe e rinominare classi da numeriche ad categoriche ovvero i nomi originali)
   df_report = pd.DataFrame.from_dict(report_scikit_learn)

   
   

   #-----#
   # Matrice di confusione
   cm = confusion_matrix(test_y, predizioni)
   
   if len(labels_float) > 0:
      df_report.rename(columns=labels_float, inplace=True)
      
      # Crea un DataFrame con etichette
      df_cm = pd.DataFrame(cm,
         index=[f"Actual {labels_float['0']}", f"Actual {labels_float['1']}"],
         columns=[f"Predicted {labels_float['0']}", f"Predicted {labels_float['1']}"]
      )
   else:
      df_cm = pd.DataFrame(cm,
         index=['Actual 0', 'Actual 1'],
         columns=['Predicted 0', 'Predicted 1']
      )



   # Ottenere metriche di performance dalla confusion matrix
   TN, FP, FN, TP = cm.ravel() # Estrazione dei valori dalla confusion matrix

   # Calcolare la specificità
   specificity_1 = TN / (TN + FP)  # Specificità per classe 1 = TN della classe 1
   specificity_0 = TP / (TP + FN)  # Specificità per classe 0 = TN rispetto alla classe 0 (simmetrico)

   # Specificando average = None si riesce a 
   precision = precision_score(test_y, predizioni, average=None)
   recall = recall_score(test_y, predizioni, average=None)
   f1 = f1_score(test_y, predizioni, average=None)


   # Unire in un dizionario tutte le metriche di performance riguardanti la classe 0
   metrics_scikit_learn_class0 = {
      'Accuracy': accuracy_score(test_y, predizioni),
      'Precision': precision[0],
      'Recall': recall[0],
      'F1-Score': f1[0],
      'Specificity': specificity_0,
   }
   
   # Unire in un dizionario tutte le metriche di performance riguardanti la classe 1
   metrics_scikit_learn_class1 = {
      'Accuracy': accuracy_score(test_y, predizioni),
      'Precision': precision[1],
      'Recall': recall[1],
      'F1-Score': f1[1],
      'Specificity': specificity_1,
   }
   
   # Classe 0, inserire le metriche dentro un pandas dataframe per un print migliore
   df_report_classi0 = pd.DataFrame.from_dict(metrics_scikit_learn_class0,orient="index").T
   df_report_classi0.rename(index={0: f"{labels_float['0']}"}, inplace=True) # rinominare indice di riga con nome patologia es. GBM
   
   # Classe 1, inserire le metriche dentro un pandas dataframe per un print migliore
   df_report_classi1 = pd.DataFrame.from_dict(metrics_scikit_learn_class1,orient="index").T
   df_report_classi1.rename(index={0: f"{labels_float['1']}"}, inplace=True) # rinominare indice di riga con nome patologia es. HC
   
   # Unire le due classi in un unico dataframe per un output migliore
   df_report_classi = pd.concat([df_report_classi0,df_report_classi1])



   # region Verbose
   if verbose:
      display(df_report)
      display(df_report_classi)

   # endregion
   output = { "df_report":df_report,
              "df_report_cm":df_report_classi,
              "confusion_matrix":df_cm 
            }
   return output

def performance_binary_fold(fold_report_cm:pd.DataFrame(),
                            fold_report:pd.DataFrame()
                           ) -> dict:
   """
      Funzione per riassumere (fare media e std) delle metriche di performance di ogni fold
   """
   metrics_mean = fold_report_cm.mean()
   metrics_std = fold_report_cm.std()
   risultati_fold_report_cm = pd.DataFrame({
      "mean (class 1)": metrics_mean,
      "std (class 1)": metrics_std
   }).T  


   # Rimuovere support, da fastidio
   fold_report.drop(index='support',inplace=True)

   fold_report.reset_index(inplace=True)
   metrics_mean_fold_report = fold_report.groupby("index").mean()
   metrics_mean_fold_report.reset_index(inplace=True)
   metrics_mean_fold_report = metrics_mean_fold_report.rename(columns={"index":"metrics"})

   metrics_std_fold_report = fold_report.groupby("index").std()
   metrics_std_fold_report.reset_index(inplace=True)
   metrics_std_fold_report = metrics_std_fold_report.rename(columns={"index":"metrics"})


   output = {"risultati_fold_report_cm":risultati_fold_report_cm,
             "metrics_mean_fold_report":metrics_mean_fold_report,
             "metrics_std_fold_report":metrics_std_fold_report
            }


   return output

def performance_multiclass(test_y : np.ndarray,
                       predizioni: np.ndarray,
                       verbose : bool = False,
                       labels_float : dict = {}
                      ) -> dict : 
   """Funzione per ottenere metriche di performance dati i valori veri per la variabile target y 
      ed i valori predetti dal modello di classificazione.

      Prende in input:
         - test_y (np.ndarray): valori veri variabile target
         - predizioni (np.ndarray): valori predetti variabile target
         - verbose (bool) opzionale: avere dei print o no
         - labels_float (dict) opzionale: inserire un dizionario per avere degli output più comprensibili.
                                          Sostituisce i numeri delle classi con i loro nomi   

      Output:
         - df_report (pd.DataFrame): metriche di performance per ogni classe e globali
         - df_report_cm (pd.DataFrame): metriche di performance per la classe positiva (codifica 1)

      Ottenere le metriche aggiuntive dalla confusion matrix preferito a  classification_report in 
      questo modo si può calcolare anche la sensitivity.
      Calcola: 
         - sensitivity (recall)
         - specificity
         - precision
         - F1Score
         - Accuracy

      Utilizza le funzioni:
        - Non utilizza funzioni esterne, utilizza quelle dei pacchetti ad esempio sklearn

      Prende in input:
        - ConfusioMatrix

      Output:
         - metriche: Pandas dataframe con le metriche di performance per ogni classe
   """ 

   if not isinstance(verbose,bool):
      raise TypeError("verbose deve essere un booleano")


   # Ottenere report con varie metriche di performance
   report_scikit_learn = classification_report(test_y, predizioni,output_dict=True)

   # Trasformare il report di perfomance in un formato più leggibile (pandas dataframe e rinominare classi da numeriche ad categoriche ovvero i nomi originali)
   df_report = pd.DataFrame.from_dict(report_scikit_learn)

 
   
   # Inizializzare il dataframe ove verranno salvate 
   metriche = pd.DataFrame()

   # Total number of samples
   ConfusioMatrix = confusion_matrix(test_y, predizioni)
   total_samples = np.sum(ConfusioMatrix)  

   # region Calcolare le metriche
   # Iterare per  ogni riga della confusion matrix, quindi per ogni classe 
   for i in range(len(ConfusioMatrix)):
      TP = ConfusioMatrix[i, i]  # True Positives for class i
      FP = ConfusioMatrix[:, i].sum() - TP  # False Positives for class i (other classes predicted as class i)
      FN = ConfusioMatrix[i, :].sum() - TP  # False Negatives for class i (class i predicted as other classes)
      TN = total_samples - (TP + FP + FN)  # True Negatives for class i

      # Calcolare le metriche, i nomi si spiegano da soli, inoltre mettere dei check aggiuntivi sulla divisione per zero
      specificity = TN / (TN + FP) if (TN + FP) > 0 else 0
      precision = TP / (TP + FP) if (TP + FP) > 0 else 0
      recall = TP  / (TP+FN) if (TP + FN) > 0 else 0
      f1_score = 2 * ((precision *recall)/(precision  + recall))  if (precision  + recall) > 0 else 0


      # Un check per vedere se i risultati corrispondono
      if TP+TN+FP+FN != total_samples:
         raise ValueError("Error of some kind occurred")

      # Calcolare l'accuracy
      accuracy = (TP+TN)/(total_samples)

      # Salvare le metriche ottenute per la classe i-esima
      row = pd.DataFrame({ "recall":[recall],
                           "specificity":[specificity],
                           "precision":[precision],
                           "F1Score":[f1_score],
                           "accuracy":[accuracy]})


      # Salvare  le metriche nel dataframe principale
      metriche=pd.concat([metriche,row])
   # endregion

   # Assegnare gli index al dataframe, senza questa riga erano tutti zeri.
   metriche.index = [float(i) for i in range(len(ConfusioMatrix))]


   if len(labels_float) > 0:
      df_report.rename(columns=labels_float, inplace=True)
      metriche.index = [str(i) for i in metriche.index]
      metriche.rename(index=labels_float,inplace=True)

   output = {"df_report":df_report,
             "df_report_cm":metriche}
   return output

def performance_multiclass_fold(fold_report_cm:pd.DataFrame(),
                            fold_report:pd.DataFrame()
                           ) -> dict:
   """
      Funzione per riassumere (fare media e std) delle metriche di performance di ogni fold
   """
   fold_report_cm.reset_index(inplace=True)

   metrics_mean = fold_report_cm.groupby("index").mean()
   metrics_mean.reset_index(inplace=True)
   metrics_mean.rename(columns={"index":"class"},inplace=True)

   metrics_std = fold_report_cm.groupby("index").std()
   metrics_std.reset_index(inplace=True)
   metrics_std.rename(columns={"index":"class"},inplace=True)


   # Rimuovere support, da fastidio
   fold_report.drop(index='support',inplace=True)

   fold_report.reset_index(inplace=True)
   metrics_mean_fold_report = fold_report.groupby("index").mean()
   metrics_mean_fold_report.reset_index(inplace=True)
   metrics_mean_fold_report = metrics_mean_fold_report.rename(columns={"index":"metrics"})

   metrics_std_fold_report = fold_report.groupby("index").std()
   metrics_std_fold_report.reset_index(inplace=True)
   metrics_std_fold_report = metrics_std_fold_report.rename(columns={"index":"metrics"})


   output = {
            "metrics_mean_fold_report_cm":metrics_mean,
            "metrics_std_fold_report_cm":metrics_std,
             "metrics_mean_fold_report":metrics_mean_fold_report,
             "metrics_std_fold_report":metrics_std_fold_report
            }


   return output