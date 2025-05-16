from typing import Union, List # Per specificare campi multipli nelll'input funzione
import pandas as pd 

def feature_to_numeric(dataset:pd.DataFrame(),
                       feature_name: Union[str, List[str]],
                       verbose : bool = False
                     ) -> dict : 

   """ Funzione per ricodificare le classi da stringhe a numeri. Molto utile per il preprocessing prima delle classificazioni
   Invece di utilizzare "from sklearn.preprocessing import LabelEncoder", utilizzo direttamente una funzione pandas.
   # basata su:  astype('category') / cat.codes / cat.categories

   Prende in input:
      - dataset (pd.DataFrame): dataset nel quale c'è la variabile da ricodificare
      - feature_name (str o lista di str): variabile/i da ricodificare
      - verbose (bool) :  se mostrare i risultati

   Output:
      - dataset (pd.DataFrame): con variabile ricodificata
      - labels (dict) : mappa con corrispondenza numeri / classi
   """
   # region Controllo input
   if not isinstance(dataset,pd.DataFrame):
      raise TypeError("dataset deve essere un pandas dataframe")
   
   if not isinstance(verbose,bool):
      raise TypeError("verbose deve essere un booleano")

   # Per il controllo delle variabili in input ci sono dei passaggi aggiuntivi data la doppia natura
   ## Se è una stringa inserirla dentro una lista
   if isinstance(feature_name, str):
        feature_name = [feature_name]  # Converti in lista se è una stringa

   for feature in feature_name:
      if not isinstance(feature,str):
         raise TypeError("Specificare le variabili sotto forma di stringa")

      if feature not in dataset.columns:
         raise ValueError(f"Variabile {feature_name} non è presente nel dataset")
   # endregion

   # region Main
   labels_list = {} # inizializzare due dizionari per salvare le labels
   labels_float_list = {}
   for feature in feature_name:
      # Convertire variabile originale in formato "category", operazione necessaria per poi ricodificare
      dataset[f'{feature}'] = dataset[f'{feature}'].astype('category')

      # Creazione del dizionario delle labels
      labels = dict(enumerate(dataset[f'{feature}'].cat.categories))
      labels_list[feature]=labels # Salvare queste labels

      # Le classi nella valutazione della performance vengono codificate in float, trasformarle subito il float per avere una corrispondenza
      labels_float = {str(float(label)):labels[label]  for  label in labels.keys()} 
      labels_float_list[feature] = labels_float # Salvare queste labels

      # Creare nuoba variabile ricodificata i numeri, la precedente viene mantenuta per ottenere mappatura labels
      dataset[f'{feature}'] = dataset[f'{feature}'].cat.codes
      # endregion

   # region Verbose
   if verbose:
      # Printare labels
      print(f"labels: {labels_list}")

      # Overview dataset
      display(dataset.head())
   # endregion

   output = {"dataset":dataset,
             "labels":labels_list #"labels_float": labels_float_list      - labels_float (dict) : mappa con corrispondenza numeri / classi modificata per compatibilità con i risultati delle performance metrics 
            }
   return output