
import pandas as pd 
import random

from sklearn.svm import SVC 
from sklearn.linear_model import LogisticRegression 

from sklearn.model_selection import KFold, LeaveOneOut, GridSearchCV, StratifiedKFold, train_test_split
from sklearn.metrics import  confusion_matrix, make_scorer, accuracy_score, f1_score, recall_score, precision_score
import sys

sys.path.append('/Users/mattia/Desktop/Università/Dottorato/EvaluateKeyGenes/__HandmadePackages__')
import evaluate_performance # Funzioni selfmade per valutare performance modelli


from typing import Literal,List,Annotated  # LITERAL: Per usare type hints in Python, un modo per indicare esplicitamente il tipo di variabili, parametri e ritorni delle funzioni.



class Evaluate:
   """Classe per testare l'efficacia diagnostica di un subset di keygens
      Prende in input dati di TRAIN e TEST e il nome del modello di classificazione binaria da utilizzare.
      Restituisce le performance ottenute. 
   """


   # Inizializzazione dell'oggetto
   # DA CAPIRE !!! -> Passare un pandas dataframe, il suo percorso file, o entrambe le possibilità? Sicuramente con entrambe non sbagli. Per ora solo pandas dataframe
   def __init__(self,
                train_data : pd.DataFrame(),
                test_data  : pd.DataFrame(),
                labels: dict,
                target_feature_name : str, 
                seed: int = None, 
                verbose: bool = False  
               ):
      """ 
      Funzione di inizializzazione.
      Richiede:
         - dati di TRAIN (pd.DataFrame)
         - dati di TEST (pd.DataFrame)
         - le labels (dizionario) nomi da assegnare alle due classi 0 ed 1, migliora output.
         - la variabile target (stringa) nome delle variabile y nei dataset.
         - il seed (opzionale, intero);
         - verbose (opzionale, booleano), se True più output mostrato a schermo.
      """
      # In questa funzione di inizializzazione vengono assegnati molti attributi per
      # facilitare il loro utilizzo nelle funzioni successive.   
      if not isinstance(train_data,pd.DataFrame):
         raise TypeError("'train_data' deve essere un pandas dataframe")

      if not isinstance(test_data,pd.DataFrame):
         raise TypeError("'test_data' deve essere un pandas dataframe")

      if not isinstance(labels,dict):
         raise TypeError("'labels' deve essere un dizionario")

      if not isinstance(target_feature_name,str):
         raise TypeError("'target_feature_name' deve essere una stringa")

      # Se un seed non è specificato viene scelto randomicamente tra 0 e 100 000 
      if not isinstance(seed,int): 
         self.seed = random.randint(0,100000) # Assegnare il seed randomico all'attributo .seed
         print(f"Seed non specificato o inserito un parametro sbagliato, assegnato un valore randomico pari a: {self.seed}")
      else:
         self.seed = seed                     # Assegnare il seed scelto all'attributo .seed
         
      # Assegnare all'attributo labels le labels passate in input
      self.labels = {str(key):str(value) for key,value in labels.items()}

      # Geni comuni al train dataset ed al test dataset
      train_genes = set(train_data.columns) # Prendere i nomi dei  geni nel TRAIN SET
      test_genes  = set(test_data.columns)  # Prendere i nomi dei geni nel TEST SET 
      common_genes = list(train_genes.intersection(test_genes)) # Prendere i geni comuni trs TRAIN e TEST set
      common_genes.remove(target_feature_name) # Rimuovere la variabile target dalla lista dei geni comumi

      # Definire i dataframe train e test per l'intero oggetto, saranno presi in considerazione solo geni comuni
      self.train_X =  train_data[common_genes].copy()        # Count matrix di TRAIN
      self.train_Y =  train_data[target_feature_name].copy() # variabili target di TRAIN

      self.test_X =  test_data[common_genes].copy()        # Count matrix di TEST
      self.test_Y =  test_data[target_feature_name].copy() # variabili target di TEST



      # Se verbose è TRUE mostrare a schermo dettagli su: Train, Test, geni comuni, seed scelto, labels.
      if verbose:
         print(f"train_data shape: {train_data.shape}")
         print(f"test_data shape: {test_data.shape}")
         print(f"Numero di geni comuni: {len(common_genes)}")
         print(f"Seed utilizzato: {self.seed}")
         print(self.labels)
         
   def SupportVectorClassifier(self):
      """ 
      Funzione per richiamare il Support vector classifier. 
      Classificazione binaria che utilizza i TRAIN e TEST specificati nell'inizializzazione della classe.
      """
      # Definizion del modello (ignorata definizione hyper parameters)
      model = SVC(random_state=self.seed)

      # Addestramento del modello su dati di train
      model.fit(self.train_X, self.train_Y)

      # Testare il modello e ottenere predizioni
      predizioni = model.predict(self.test_X)

      # Calcolo metriche di performance 
      self.SupportVectorClassifierResults  = evaluate_performance.performance_binary(test_y = self.test_Y,
                                                                                     predizioni = predizioni,
                                                                                     labels_float = self.labels)

   def SupportVectorClassifierKFold(self,   
                                       metric_to_optimize: Literal['accuracy','f1','precision', 'recall'] = 'accuracy', # Metrica da ottimizzare
                                       stratify : bool = True, # Bilanciare o meno TRAIN e VALIDATION set
                                       fold : int = 5, # Numero di Fold 
                                       param_grid: dict = { 'C': [0.1, 1, 10, 100],   # Esempio di valori per il parametro C
                                                            'kernel': ['linear', 'rbf'],# Sperimenta sia kernel lineari che RBF
                                                            'gamma': ['scale', 'auto']  # Opzioni per il parametro gamma
                                                            }
                                       ):
      """ 
      Funzione per richiamare il Support vector classifier con ottimizzazione degli hyperparamaters tramite Kfold.
      Classificazione binaria che utilizza i TRAIN e TEST specificati nell'inizializzazione della classe.
      Prende in input i seguenti parametri:
         - metric_to_optimize: ["accuracy","f1score","recall","precision"];
         - stratify: ["True","False"] True per dividere TRAIN e VALIDATION set in maniera bilanciata rispetto alle y;
         - fold: Deve essere un intero, numero di fold per il KFold;
         - parama_grid: Dizionario degli hyper parametri da ottimizzare.
      """
      if not isinstance(stratify,bool):
         raise TypeError(f"'stratify' deve essere un booleano")      

      if not isinstance(fold,int):
         raise TypeError(f"'fold' deve essere un intero")      

      if not isinstance(param_grid,dict):
         raise TypeError(f"'parama_grid' deve essere un dizionario")      

      if metric_to_optimize not in ['accuracy','f1','precision', 'recall']:
         raise ValueError(f"'metric_to_optimize' devere essere in {['accuracy','f1','precision', 'recall']}")
      

      score = {"accuracy":accuracy_score,"f1":f1_score,"recall":recall_score,"precision":precision_score}
         
      # Se "stratify" True la divisione del train set tra TRAIN e VAL viene fatta separando
      # i due dataset bilanciando le classi della variabile target y, stessa cosa valida per i KFold
      if stratify:
         # Divisione in TRAIN e VAL tramite stratify
         train_X, X_val, train_y, y_val = train_test_split(self.train_X,
                                                         self.train_Y,
                                                         test_size=0.30,
                                                         random_state=self.seed,
                                                         stratify=self.train_Y) 
         
         # KFold con Fold stratified
         kf = StratifiedKFold(n_splits=fold, shuffle=True, random_state = self.seed)
      else:
         # Division TRAIN e VAL senza stratify
         train_X, X_val, train_y, y_val = train_test_split(self.train_X,
                                                           self.train_Y,
                                                           test_size=0.30,
                                                           random_state=self.seed) 
         
         # KFold con Fold senza stratified
         kf = KFold(n_splits=fold, shuffle=True, random_state = self.seed)
         

      # Definire il modello di classificazione per il GridSearchCV      
      base_model = SVC(random_state = self.seed)
   
      # Parametro da ottimizzare durante il GridSearchCV
      custom_scorer = make_scorer(score[metric_to_optimize], average=None, labels=[0])
   
      # Definire il GridSearchCV
      grid_search = GridSearchCV(estimator=base_model,  # Modello di classificazione 
                                 param_grid=param_grid, # Parametri del modello di classificazione da ottimizzare
                                 cv=kf,                 # Schema di KFold da utilizzare
                                 n_jobs=-1,
                                 scoring = custom_scorer # Metrica da ottimizzare
                                 )
   
      # Fornire al GridSearchCV i dati, in questo modo si addestrano gli hyperparametri.
      grid_search.fit(X_val, y_val)
      
      # Definire il modello finale utilizzando i hyperparameters ottimizzati tramite GridSearchCV
      final_model = SVC(**grid_search.best_params_, random_state = self.seed)#,probability=True)
   
      # Addestrare il modello finale con hyperparameters ottimizzati
      final_model.fit(train_X, train_y)
   
      # Ottenere predizioni utilizzando il modello finale per poterlo valutare
      predizioni = final_model.predict(self.test_X)
   
      # Valutare le performance del modello tramite funzione esterna
      self.SupportVectorClassifierKFoldResults  = evaluate_performance.performance_binary(test_y = self.test_Y,
                                                                                                 predizioni = predizioni,
                                                                                                 labels_float = self.labels)

   def LogisticRegression(self):
      """ 
      Funzione per richiamare la Logistic Regression.
      Classificazione binaria che utilizza i TRAIN e TEST specificati nell'inizializzazione della classe.
      """
      # Definizion del modello (ignorata definizione hyper parameters)
      model = LogisticRegression(max_iter=10000,
                                 random_state=self.seed)

      # Addestramento del modello su dati di train
      model.fit(self.train_X, self.train_Y)

      # Testare il modello e ottenere predizioni
      predizioni = model.predict(self.test_X)

      # Calcolo metriche di performance 
      self.LogisticRegressionResults  = evaluate_performance.performance_binary(test_y = self.test_Y,
                                                                                predizioni = predizioni,
                                                                                labels_float=self.labels)

   def RandomForest():
      """
      Classificazione binaria che utilizza i TRAIN e TEST specificati nell'inizializzazione della classe.
      """
      pass

   def KNearestNeighbors():
      """Classificazione binaria che utilizza i TRAIN e TEST specificati nell'inizializzazione della classe."""
      pass

   def NaiveBayesBinomial():
      """Classificazione binaria che utilizza i TRAIN e TEST specificati nell'inizializzazione della classe."""
      pass

   def NaiveBayesGaussian():
      """Classificazione binaria che utilizza i TRAIN e TEST specificati nell'inizializzazione della classe."""
      pass


   # Calcolo delle performance
   def get_performance(self,
                        modelli: List[Annotated[str, Literal['SVC', 'SVC_KF', 'LR']]], # Specificare modelli che si possono utilizzare,
                        verbose : bool = True
                     ):

      print("\n\n")
      if   "LR" in modelli:
         print("Logistic Regression")
         display(self.LogisticRegressionResults["df_report"])    ## Risultati per classe
         display(self.LogisticRegressionResults["df_report_cm"]) ## Risultati relativi alla classe positiva (codifica 1)
         
      if  "SVC" in modelli:
         print("Support Vector Classifier")
         display(self.SupportVectorClassifierResults["df_report"])    ## Risultati per classe
         display(self.SupportVectorClassifierResults["df_report_cm"]) ## Risultati relativi alla classe positiva (codifica 1)
         display(self.SupportVectorClassifierResults["df_report_classi"])
         
      if  "SVC_KF" in modelli:
         print("Support Vector Classifier KFold")
         display(self.SupportVectorClassifierKFoldResults["df_report"])    ## Risultati per classe
         display(self.SupportVectorClassifierKFoldResults["df_report_cm"]) ## Risultati relativi alla classe positiva (codifica 1)
         
      return 
      

   def get_report(self,
                  parametri):
      pass



# C'è da lavorare sull'output, aggiungere modelli, aggiungere le versioni con tuning dei modelli, commentare, inserire le doc string, capire come richiamare i vari modelli, inserire  RAISE ERROR, la creazione di report la escludo sarà fatta direttamente in fase di utilizzo per essere più flessibile.


