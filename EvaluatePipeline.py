# Richiede python 3.9 o successivi. (Dovuto a from typing import Annotated)
import pandas as pd 
import random

from sklearn.svm import SVC 
from sklearn.linear_model import LogisticRegression 
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import BernoulliNB
from sklearn.naive_bayes import GaussianNB

from sklearn.model_selection import KFold, LeaveOneOut, GridSearchCV, StratifiedKFold, train_test_split
from sklearn.metrics import  confusion_matrix, make_scorer, accuracy_score, f1_score, recall_score, precision_score
import sys

sys.path.append('/Users/mattia/Desktop/Università/Dottorato/EvaluateKeyGenes/__HandmadePackages__')
import evaluate_performance # Funzioni selfmade per valutare performance modelli


from typing import Literal,List,Annotated,Optional  # LITERAL: Per usare type hints in Python, un modo per indicare esplicitamente il tipo di variabili, parametri e ritorni delle funzioni. Optional per parametri opzionali



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
      self.common_genes = list(train_genes.intersection(test_genes)) # Prendere i geni comuni trs TRAIN e TEST set
      self.common_genes.remove(target_feature_name) # Rimuovere la variabile target dalla lista dei geni comumi


      # Definire i dataframe train e test per l'intero oggetto, saranno presi in considerazione solo geni comuni
      self.train_X =  train_data[self.common_genes].copy()        # Count matrix di TRAIN
      self.train_Y =  train_data[target_feature_name].copy() # variabili target di TRAIN

      self.test_X =  test_data[self.common_genes].copy()        # Count matrix di TEST
      self.test_Y =  test_data[target_feature_name].copy() # variabili target di TEST



      # Se verbose è TRUE mostrare a schermo dettagli su: Train, Test, geni comuni, seed scelto, labels.
      self.train_data_shape = train_data.shape
      self.test_data_shape = test_data.shape
      
      # Inizializzare indicatori di uso
      # serviranno a capire se un modello è stato utilizzato, in questo modo si potranno gestire meglio gli output
      self.LogisticRegression_called = False
      self.SupportVectorClassifier_called = False
      self.SupportVectorClassifierKFold_called = False
      self.RandomForest_called = False
      self.KNearestNeighbors_called = False
      self.NaiveBayesBinomial_called = False
      self.NaiveBayesGaussian_called = False


      if verbose:
         print(f"train_data shape: {self.train_data_shape}")
         print(f"test_data shape: {self.test_data_shape}")
         print(f"Numero di geni comuni: {len(self.common_genes)}")
         print(f"Seed utilizzato: {self.seed}")
         print(f"Labels utilizzate: {self.labels}")
         
   def __LeaveOneOut__(self,modello):
      """Metodo privato per eseguire una leave one out classification.
         Molto utile per verificare performance del modello nei train set.
         Prende in input:
         -  modello di classificazione da utilizzare
         
         In output restituisce:
            - y_true: valori veri delle variabili target
            - y_pred: valori predetti delle variabili target
      """
      # Define X and y
      X = self.train_X.copy()
      y = self.train_Y.copy()
      
      # Siccome loo.get_n_splits(X) utilizza come indici numeri, droppare i nomi dei pazienti e sostituirli con numeri per evitare errori
      X.reset_index(drop=True,inplace=True)      
      y.reset_index(drop=True,inplace=True)      

      # Define LOOCV
      loo = LeaveOneOut()
      loo.get_n_splits(X)

      # Define true and predict list
      y_true,y_pred = [],[]
      
      # LeaveOneOut classification
      for train_index, test_index in loo.split(X):
         
         # Dividere in Train e Test le variabili esplicative
         X_train, X_test = X.loc[train_index], X.loc[test_index]
         
         # Dividere in train e test la variabile target
         y_train, y_test = y.loc[train_index], y.loc[test_index]
         
         # Inizializzare modello per l'iesima previsione sulla iesima riga
         model = modello
         
         # Fittare il modello per l'iesima previsione sulla iesima riga
         model.fit(X_train,y_train)
         
         # Ottenere la prediction
         yhat = model.predict(X_test)
         
         # Salvare il vero valore della variabile target
         y_true.append(y_test)
         
         # Salvare la predizione del modello per la variabile target
         y_pred.append(yhat)
         

      return y_true,y_pred

   def SupportVectorClassifier(self,
                               train : bool = False,
                               verbose : bool = False):
      """ 
      Funzione per richiamare il Support vector classifier. 
      Classificazione binaria che utilizza i TRAIN e TEST specificati nell'inizializzazione della classe.
      Parametri:
         - verbose (opzionale, booleano), se True più output mostrato a schermo.
         - train (opzionale, booleano), se True viene testato il modello anche sul train set.
      """
      if not isinstance(verbose,bool):
         raise TypeError("'verbose' deve essere un booleano")      
      
      if not isinstance(train,bool):
         raise TypeError("'train' deve essere un booleano")   

      # Check performance sul dati di TRAIN
      # Il check e la scelta del modello devono essere fatti sul TRAIN, il test solo per la convalida finale      
      model = SVC(random_state=self.seed)
      
      # Leave One out 
      if train:
         y_true_train,y_pred_train = self.__LeaveOneOut__(modello=model)
         self.SupportVectorClassifierResults_train  = evaluate_performance.performance_binary(test_y = y_true_train,
                                                                                             predizioni = y_pred_train,
                                                                                             labels_float = self.labels,
                                                                                             verbose = verbose
                                                                                             )
      else:
         self.SupportVectorClassifierResults_train = {"df_report":pd.DataFrame(),
                                                      "df_report_cm":pd.DataFrame(),
                                                      "confusion_matrix":pd.DataFrame()
                                                      }

      # TEST del modello su DATI NUOVI (test dataset)
      # Definizion del modello (ignorata definizione hyper parameters)
      # Addestramento del modello su dati di train
      model.fit(self.train_X, self.train_Y)

      # Testare il modello e ottenere predizioni
      predizioni = model.predict(self.test_X)

      self.SupportVectorClassifierParameters = model.get_params()

      # Calcolo metriche di performance 
      self.SupportVectorClassifierResults  = evaluate_performance.performance_binary(test_y = self.test_Y,
                                                                                     predizioni = predizioni,
                                                                                     labels_float = self.labels,
                                                                                     verbose = verbose
                                                                                     )

      self.SupportVectorClassifier_called = True
  
   def SupportVectorClassifierKFold(self,   
                                    metric_to_optimize: Literal['accuracy','f1','precision', 'recall'] = 'accuracy', # Metrica da ottimizzare
                                    class_to_optimize : int = 1,
                                    stratify : bool = True, # Bilanciare o meno TRAIN e VALIDATION set
                                    fold : int = 5, # Numero di Fold 
                                    param_grid: dict = { 'C': [0.1, 1, 10, 100],   # Esempio di valori per il parametro C
                                                         'kernel': ['linear', 'rbf'],# Sperimenta sia kernel lineari che RBF
                                                         'gamma': ['scale', 'auto']  # Opzioni per il parametro gamma
                                                         },
                                    verbose = False
                                    ):
      """ 
      Funzione per richiamare il Support vector classifier con ottimizzazione degli hyperparamaters tramite Kfold.
      Classificazione binaria che utilizza i TRAIN e TEST specificati nell'inizializzazione della classe.
      Prende in input i seguenti parametri:
         - metric_to_optimize: ["accuracy","f1score","recall","precision"];
         - class_to_optimize: classe da ottimizzare [0,1];
         - stratify: ["True","False"] True per dividere TRAIN e VALIDATION set in maniera bilanciata rispetto alle y;
         - fold: Deve essere un intero, numero di fold per il KFold;
         - parama_grid: Dizionario degli hyper parametri da ottimizzare.
         - verbose (opzionale, booleano), se True più output mostrato a schermo.
      """
      if not isinstance(stratify,bool):
         raise TypeError(f"'stratify' deve essere un booleano")      

      if not isinstance(fold,int):
         raise TypeError(f"'fold' deve essere un intero")      

      if not isinstance(param_grid,dict):
         raise TypeError(f"'parama_grid' deve essere un dizionario")      

      if not isinstance(class_to_optimize,int):
         raise TypeError(f"'class_to_optimize' deve essere un intero")      
      
      if class_to_optimize not in {0,1}:
         raise ValueError(f"'class_to_optimize' deve essere 0 oppure 1")   

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
      custom_scorer = make_scorer(score[metric_to_optimize], average=None, labels=[class_to_optimize])
   
      # Definire il GridSearchCV
      grid_search = GridSearchCV(estimator=base_model,  # Modello di classificazione 
                                 param_grid=param_grid, # Parametri del modello di classificazione da ottimizzare
                                 cv=kf,                 # Schema di KFold da utilizzare
                                 n_jobs=-1,
                                 scoring = custom_scorer # Metrica da ottimizzare
                                 )
   
      # Fornire al GridSearchCV i dati, in questo modo si addestrano gli hyperparametri.
      grid_search.fit(X_val, y_val)
      
      self.SupportVectorClassifierKFoldParametri = grid_search.best_params_

      # Definire il modello finale utilizzando i hyperparameters ottimizzati tramite GridSearchCV
      final_model = SVC(**grid_search.best_params_, random_state = self.seed)#,probability=True)
   
      # Addestrare il modello finale con hyperparameters ottimizzati
      final_model.fit(train_X, train_y)
   
      # Ottenere predizioni utilizzando il modello finale per poterlo valutare
      predizioni = final_model.predict(self.test_X)
   
      if verbose:
         print("Parametri selezionati:")
         print(grid_search.best_params_)

      # Valutare le performance del modello tramite funzione esterna
      self.SupportVectorClassifierKFoldResults  = evaluate_performance.performance_binary(test_y = self.test_Y,
                                                                                          predizioni = predizioni,
                                                                                          labels_float = self.labels,
                                                                                          verbose = verbose
                                                                                          )
      
      self.SupportVectorClassifierKFold_called = True

   def LogisticRegression(self,
                          train: bool = False,
                          verbose : bool = False):                          
      """ 
      Funzione per richiamare la Logistic Regression.
      Classificazione binaria che utilizza i TRAIN e TEST specificati nell'inizializzazione della classe.
      Parametri:
         - verbose (opzionale, booleano), se True più output mostrato a schermo.
         - train (opzionale, booleano), se True viene testato il modello anche sul train set.
      """
      if not isinstance(verbose,bool):
         raise TypeError("'verbose' deve essere un booleano")      
      
      if not isinstance(train,bool):
         raise TypeError("'train' deve essere un booleano")   

      # Definizion del modello (ignorata definizione hyper parameters)
      model = LogisticRegression(max_iter=10000,
                                 random_state=self.seed)
      
      # Leave One out 
      if train:
         y_true_train,y_pred_train = self.__LeaveOneOut__(modello=model)
         self.LogisticRegressionResults_train  = evaluate_performance.performance_binary(test_y = y_true_train,
                                                                                          predizioni = y_pred_train,
                                                                                          labels_float = self.labels,
                                                                                          verbose = verbose
                                                                                       )
      else:
         self.LogisticRegressionResults_train = {"df_report":pd.DataFrame(),
                                                 "df_report_cm":pd.DataFrame(),
                                                 "confusion_matrix":pd.DataFrame()
                                                }      

      # Addestramento del modello su dati di train
      model.fit(self.train_X, self.train_Y)

      # Testare il modello e ottenere predizioni
      predizioni = model.predict(self.test_X)

      self.LogisticRegressionParameters = model.get_params()

      # Calcolo metriche di performance 
      self.LogisticRegressionResults  = evaluate_performance.performance_binary(test_y = self.test_Y,
                                                                                predizioni = predizioni,
                                                                                labels_float=self.labels,
                                                                                verbose=verbose
                                                                                )

      self.LogisticRegression_called = True
     
   def RandomForest(self,
                    train : bool = False,
                    verbose : bool = False):            
      """
      Funzione per richiamare la Logistic Regression.
      Classificazione binaria che utilizza i TRAIN e TEST specificati nell'inizializzazione della classe.
      Parametri:
         - verbose (opzionale, booleano), se True più output mostrato a schermo.
         - train (opzionale, booleano), se True viene testato il modello anche sul train set.
      """
      if not isinstance(verbose,bool):
         raise TypeError("'verbose' deve essere un booleano")  
      
      if not isinstance(train,bool):
         raise TypeError("'train' deve essere un booleano")   
         
      # Per ora ignorati
      possibili_parametri_RF = {'n_estimators': [50, 100, 150, 200],  # Numero di alberi nella foresta
                           'max_depth': [None, 10, 20, 30],  # Profondità massima degli alberi
                           'min_samples_split': [2, 5, 10],  # Numero minimo di campioni per dividere un nodo
                           'min_samples_leaf': [1, 2, 4],  # Numero minimo di campioni per essere una foglia
                           'max_features': ['auto', 'sqrt', 'log2'],  # Numero di caratteristiche da considerare per ogni divisione
                         }

      # Definizion del modello (ignorata definizione hyper parameters)
      model = model = RandomForestClassifier(random_state=self.seed)
      
      # Leave One out 
      if train:
         y_true_train,y_pred_train = self.__LeaveOneOut__(modello=model)
         self.RandomForestResults_train  = evaluate_performance.performance_binary(test_y = y_true_train,
                                                                                          predizioni = y_pred_train,
                                                                                          labels_float = self.labels,
                                                                                          verbose = verbose
                                                                                       )
      else:
         self.RandomForestResults_train = {"df_report":pd.DataFrame(),
                                             "df_report_cm":pd.DataFrame(),
                                             "confusion_matrix":pd.DataFrame()
                                          }      

      


      # Addestramento del modello su dati di train
      model.fit(self.train_X, self.train_Y)

      # Testare il modello e ottenere predizioni
      predizioni = model.predict(self.test_X)

      self.RandomForestParameters = model.get_params()

      # Calcolo metriche di performance 
      self.RandomForestResults  = evaluate_performance.performance_binary(test_y = self.test_Y,
                                                                          predizioni = predizioni,
                                                                          labels_float=self.labels,
                                                                           verbose=verbose
                                                                          )
      self.RandomForest_called = True

   def KNearestNeighbors(self,
                         train : bool = False,
                         verbose : bool = False):            
      """
      Funzione per richiamare la KNearestNeighbors.
      Classificazione binaria che utilizza i TRAIN e TEST specificati nell'inizializzazione della classe.
      Parametri:
         - verbose (opzionale, booleano), se True più output mostrato a schermo.
         - train (opzionale, booleano), se True viene testato il modello anche sul train set.
      """
      if not isinstance(verbose,bool):
         raise TypeError("'verbose' deve essere un booleano")  
      
      if not isinstance(train,bool):
         raise TypeError("'train' deve essere un booleano")   
         
      # Per ora ignorati
      possibili_parametri_KNN = {'n_neighbors': [3, 5, 7, 10, 15],  # Numero di vicini da provare
                                 'weights': ['uniform', 'distance'],  # Tipi di pesatura per i vicini
                                 'metric': ['euclidean', 'manhattan']  # Differenti metriche di distanza
                              }

      # Definizion del modello (ignorata definizione hyper parameters)
      model = KNeighborsClassifier() # non ha come parametro random_state
      
      # Leave One out 
      if train:
         y_true_train,y_pred_train = self.__LeaveOneOut__(modello=model)
         self.KNearestNeighborsResults_train  = evaluate_performance.performance_binary(test_y = y_true_train,
                                                                                          predizioni = y_pred_train,
                                                                                          labels_float = self.labels,
                                                                                          verbose = verbose
                                                                                       )
      else:
         self.KNearestNeighborsResults_train = {"df_report":pd.DataFrame(),
                                             "df_report_cm":pd.DataFrame(),
                                             "confusion_matrix":pd.DataFrame()
                                          }            



      # Addestramento del modello su dati di train
      model.fit(self.train_X, self.train_Y)

      # Testare il modello e ottenere predizioni
      predizioni = model.predict(self.test_X)

      self.KNearestNeighborsParameters = model.get_params()

      # Calcolo metriche di performance 
      self.KNearestNeighborsResults  = evaluate_performance.performance_binary(test_y = self.test_Y,
                                                                               predizioni = predizioni,
                                                                               labels_float=self.labels,
                                                                               verbose=verbose
                                                                              )
      self.KNearestNeighbors_called = True

   def NaiveBayesBinomial(self,
                          train : bool = False,
                          verbose : bool = False):            
      """
      Funzione per richiamare la Naive Bayes Binomial.
      Classificazione binaria che utilizza i TRAIN e TEST specificati nell'inizializzazione della classe.
      Parametri:
         - verbose (opzionale, booleano), se True più output mostrato a schermo.
         - train (opzionale, booleano), se True viene testato il modello anche sul train set.
      """
      if not isinstance(verbose,bool):
         raise TypeError("'verbose' deve essere un booleano")  
      
      if not isinstance(train,bool):
         raise TypeError("'train' deve essere un booleano")   
         
 
      # Per ora non utilizzati
      possibili_parametri_NB_Binomial =  {'alpha': [0.1, 0.5, 1.0, 2.0],  # Parametro di regolarizzazione (default è 1.0)
                                          'binarize': [0.0, 0.1, 0.5, 1.0],  # Soglia per binarizzare i dati (trasformare le caratteristiche in 0 o 1)
                                       }

      # Definizion del modello (ignorata definizione hyper parameters)
      model =  BernoulliNB() # non ha come parametro random_state
      
      if train:
         y_true_train,y_pred_train = self.__LeaveOneOut__(modello=model)
         self.NaiveBayesBinomialResults_train  = evaluate_performance.performance_binary(test_y = y_true_train,
                                                                                          predizioni = y_pred_train,
                                                                                          labels_float = self.labels,
                                                                                          verbose = verbose
                                                                                       )
      else:
         self.NaiveBayesBinomialResults_train = {"df_report":pd.DataFrame(),
                                                "df_report_cm":pd.DataFrame(),
                                                "confusion_matrix":pd.DataFrame()
                                               }          

      # Addestramento del modello su dati di train
      model.fit(self.train_X, self.train_Y)

      # Testare il modello e ottenere predizioni
      predizioni = model.predict(self.test_X)

      self.NaiveBayesBinomialParameters = model.get_params()

      # Calcolo metriche di performance 
      self.NaiveBayesBinomialResults  = evaluate_performance.performance_binary(test_y = self.test_Y,
                                                                               predizioni = predizioni,
                                                                               labels_float=self.labels,
                                                                               verbose=verbose
                                                                              )
      self.NaiveBayesBinomial_called = True

   def NaiveBayesGaussian(self,
                          train : bool = False,
                          verbose : bool = False):            
      """
      Funzione per richiamare la Naive Bayes Gaussian.
      Classificazione binaria che utilizza i TRAIN e TEST specificati nell'inizializzazione della classe.
      Parametri:
         - verbose (opzionale, booleano), se True più output mostrato a schermo.
         - train (opzionale, booleano), se True viene testato il modello anche sul train set.
      """
      if not isinstance(verbose,bool):
         raise TypeError("'verbose' deve essere un booleano")  
         
      if not isinstance(train,bool):
         raise TypeError("'train' deve essere un booleano")   
 
      # Possibili parametri, per ora non utilizzati
      possibili_parametri_NB_Gaussian = { 'var_smoothing': [1e-9, 1e-8, 1e-7, 1e-6]  # Parametro per regolarizzare la varianza dei dati
                                  }

      # Definizion del modello (ignorata definizione hyper parameters)
      model =  GaussianNB() # non ha come parametro random_state
      
      if train:
         y_true_train,y_pred_train = self.__LeaveOneOut__(modello=model)
         self.NaiveBayesGaussianResults_train  = evaluate_performance.performance_binary(test_y = y_true_train,
                                                                                          predizioni = y_pred_train,
                                                                                          labels_float = self.labels,
                                                                                          verbose = verbose
                                                                                       )
      else:
         self.NaiveBayesGaussianResults_train = {"df_report":pd.DataFrame(),
                                                "df_report_cm":pd.DataFrame(),
                                                "confusion_matrix":pd.DataFrame()
                                               }          

      # Addestramento del modello su dati di train
      model.fit(self.train_X, self.train_Y)

      # Testare il modello e ottenere predizioni
      predizioni = model.predict(self.test_X)

      self.NaiveBayesGaussianParameters = model.get_params()

      # Calcolo metriche di performance 
      self.NaiveBayesGaussianResults  = evaluate_performance.performance_binary(test_y = self.test_Y,
                                                                               predizioni = predizioni,
                                                                               labels_float=self.labels,
                                                                               verbose=verbose
                                                                              )
      
      self.NaiveBayesGaussian_called = True

   def get_performance(self,
                       verbose : bool = True
                     ):
      """
      Questa funzione prende in input:
         - verbose (opzionale): se mostrare (verbose = True) a schermo i risultati delle performance dei modelli specificati;
      Gli output  dei modelli che si otteranno sono:
         - risultati della classificazione in formato dataframe (due versioni distinte):
         - hyperparametri dei modelli.
      """
      
      self.risultati = None
      risultati = {}
      
      # Logistic Regression
      if self.LogisticRegression_called:
         
         if verbose: # Se verbose = True mostra a schermo i risultati della classificazione
            print("\n\n")
            print("Logistic Regression")
            display(self.LogisticRegressionResults["df_report"])    ## Risultati per classe
            display(self.LogisticRegressionResults["df_report_cm"]) ## Risultati relativi alla classe positiva (codifica 1)
        
         risultati["LR"] = { "df_report":   self.LogisticRegressionResults["df_report"],
                             "df_report_cm":self.LogisticRegressionResults["df_report_cm"],
                             "parametri":   self.LogisticRegressionParameters, # Parametri di default
                             "confusion_matrix": self.LogisticRegressionResults["confusion_matrix"],
                             "df_report_train":   self.LogisticRegressionResults_train["df_report"],
                             "df_report_cm_train":self.LogisticRegressionResults_train["df_report_cm"],
                             "confusion_matrix_train": self.LogisticRegressionResults_train["confusion_matrix"],
                           }
   
      # Support vector classifier
      if self.SupportVectorClassifier_called:
         if verbose: # Se verbose = True mostra a schermo i risultati della classificazione
            print("\n\n")
            print("Support Vector Classifier")
            display(self.SupportVectorClassifierResults["df_report"])    ## Risultati per classe
            display(self.SupportVectorClassifierResults["df_report_cm"]) ## Risultati relativi alla classe positiva (codifica 1)
            
         risultati["SVC"] = { "df_report":   self.SupportVectorClassifierResults["df_report"],
                              "df_report_cm":self.SupportVectorClassifierResults["df_report_cm"],
                              "parametri" :  self.SupportVectorClassifierParameters, # Parametri di default,
                              "confusion_matrix": self.SupportVectorClassifierResults["confusion_matrix"],
                              
                              "df_report_train":   self.SupportVectorClassifierResults_train["df_report"],
                              "df_report_cm_train":self.SupportVectorClassifierResults_train["df_report_cm"],
                              "confusion_matrix_train": self.SupportVectorClassifierResults_train["confusion_matrix"]
                            }

      # Support vector classifier con ottimizzazione hyperparameters tramite KFold   
      if self.SupportVectorClassifierKFold_called:
         if verbose: # Se verbose = True mostra a schermo i risultati della classificazione
            print("\n\n")
            print("Support Vector Classifier KFold")
            display(self.SupportVectorClassifierKFoldResults["df_report"])    ## Risultati per classe
            display(self.SupportVectorClassifierKFoldResults["df_report_cm"]) ## Risultati relativi alla classe positiva (codifica 1)
         
         risultati["SVC_KF"] = { "df_report":   self.SupportVectorClassifierKFoldResults["df_report"],
                                 "df_report_cm":self.SupportVectorClassifierKFoldResults["df_report_cm"],
                                 "parametri":   self.SupportVectorClassifierKFoldParametri, # In aggiunta ha i parametri migliori scelti per la classificazione,
                                 "confusion_matrix": self.SupportVectorClassifierKFoldResults["confusion_matrix"],
                                 
                                 "df_report_train":   pd.DataFrame(),
                                 "df_report_cm_train":pd.DataFrame(),
                                 "confusion_matrix_train": pd.DataFrame(),
                               }       
         
      # RandomForest classifier
      if self.RandomForest_called:
         if verbose: # Se verbose = True mostra a schermo i risultati della classificazione
            print("\n\n")
            print("Random Forest Classifier")
            display(self.RandomForestResults["df_report"])    ## Risultati per classe
            display(self.RandomForestResults["df_report_cm"]) ## Risultati relativi alla classe positiva (codifica 1)
         
         risultati["RF"] = { "df_report":   self.RandomForestResults["df_report"],
                             "df_report_cm":self.RandomForestResults["df_report_cm"],
                             "parametri":   self.RandomForestParameters, # In aggiunta ha i parametri migliori scelti per la classificazione,
                             "confusion_matrix": self.RandomForestResults["confusion_matrix"],
                             "df_report_train":   self.RandomForestResults_train["df_report"],
                             "df_report_cm_train":self.RandomForestResults_train["df_report_cm"],
                             "confusion_matrix_train": self.RandomForestResults_train["confusion_matrix"],
                           }       
            # RandomForest con ottimizzazione hyperparameters tramite KFold   
      
      # K Nearest Neighbors classifier
      if self.KNearestNeighbors_called:
         if verbose: # Se verbose = True mostra a schermo i risultati della classificazione
            print("\n\n")
            print("K Nearest Neighbors classifier")
            display(self.KNearestNeighborsResults["df_report"])    ## Risultati per classe
            display(self.KNearestNeighborsResults["df_report_cm"]) ## Risultati relativi alla classe positiva (codifica 1)
         
         risultati["KNN"] = { "df_report":   self.KNearestNeighborsResults["df_report"],
                              "df_report_cm":self.KNearestNeighborsResults["df_report_cm"],
                              "parametri":   self.KNearestNeighborsParameters, # In aggiunta ha i parametri migliori scelti per la classificazione,
                              "confusion_matrix": self.KNearestNeighborsResults["confusion_matrix"],
                              "df_report_train":   self.KNearestNeighborsResults_train["df_report"],
                              "df_report_cm_train":self.KNearestNeighborsResults_train["df_report_cm"],
                              "confusion_matrix_train": self.KNearestNeighborsResults_train["confusion_matrix"]}    
      
      # Naive Bayes Binomial
      if self.NaiveBayesBinomial_called:
         if verbose: # Se verbose = True mostra a schermo i risultati della classificazione
            print("\n\n")
            print("K Nearest Neighbors classifier")
            display(self.NaiveBayesBinomialResults["df_report"])    ## Risultati per classe
            display(self.NaiveBayesBinomialResults["df_report_cm"]) ## Risultati relativi alla classe positiva (codifica 1)
         
         risultati["NB_Binomial"] = { "df_report":   self.NaiveBayesBinomialResults["df_report"],
                                      "df_report_cm":self.NaiveBayesBinomialResults["df_report_cm"],
                                       "parametri":   self.NaiveBayesBinomialParameters, # In aggiunta ha i parametri migliori scelti per la classificazione,
                                       "confusion_matrix": self.NaiveBayesBinomialResults["confusion_matrix"],
                                       "df_report_train":   self.NaiveBayesBinomialResults_train["df_report"],
                                       "df_report_cm_train":self.NaiveBayesBinomialResults_train["df_report_cm"],
                                       "confusion_matrix_train": self.NaiveBayesBinomialResults_train["confusion_matrix"]
                                    }      
         
      # Naive Bayes Gaussian
      if self.NaiveBayesGaussian_called:
         if verbose: # Se verbose = True mostra a schermo i risultati della classificazione
            print("\n\n")
            print("K Nearest Neighbors classifier")
            display(self.NaiveBayesGaussianResults["df_report"])    ## Risultati per classe
            display(self.NaiveBayesGaussianResults["df_report_cm"]) ## Risultati relativi alla classe positiva (codifica 1)
         
         risultati["NB_Gaussian"] = { "df_report":   self.NaiveBayesGaussianResults["df_report"],
                                      "df_report_cm":self.NaiveBayesGaussianResults["df_report_cm"],
                                      "parametri":   self.NaiveBayesGaussianParameters, # In aggiunta ha i parametri migliori scelti per la classificazione,
                                      "confusion_matrix": self.NaiveBayesGaussianResults["confusion_matrix"],
                                      "df_report_train":   self.NaiveBayesGaussianResults_train["df_report"],
                                      "df_report_cm_train":self.NaiveBayesGaussianResults_train["df_report_cm"],
                                      "confusion_matrix_train": self.NaiveBayesGaussianResults_train["confusion_matrix"]
                                    }       
      
      self.risultati = risultati
      return risultati
   
   def performance_summary(self, condition):
      """Estrarre dai risultati una tabella con le sole performance per una determinata condizione per ogni classificatore
         Funzione da utilizzare solo dopo get_performance.
         Input:
         - condition: condizione di interesse per la quale visualizzare i risultati.   
         
         Output:
         - summary: pandas dataframe contenente il sommario delle performance per ogni classificatore
      """
      summary = pd.DataFrame() 
      
      # Iterare su  ogni modello addestrato 
      for modello in self.risultati.keys():
         
         # Per ogni modello estrarre il sommario relativo alla condizione scelta
         model_summary = pd.DataFrame(self.risultati[modello]["df_report_cm"].loc[condition])
         model_summary.rename(columns={condition:modello},inplace=True)         

         # Inserire tutte le condizioni in un unico dataframe
         summary = pd.concat([summary,model_summary],axis=1)

      summary = summary.T

      return summary
   
   def performance_summary_train(self, condition):
      """Estrarre dai risultati una tabella con le sole performance per una determinata condizione per ogni classificatore
         Funzione da utilizzare solo dopo get_performance.
         In particolare questa funzione è relativa solo alle performance ottenute sui train set.
         è stata necessaria un'altra funzione perchè i risultati sui dati di test vengono gestiti diversamente
         Input:
            - condition: condizione di interesse per la quale visualizzare i risultati.   
            
         Output:
            - summary: pandas dataframe contenente il sommario delle performance per ogni classificatore
      """
      summary = pd.DataFrame() 
      
      # Iterare su  ogni modello addestrato 
      for modello in self.risultati.keys():
         # Se per il modello in considerazione sono state calcolate le performance su dati di train
         # il pandas dataframe risulterà con shape diversa da (0,0) perciò si può procedere con la creazione
         # del sommario
         if self.risultati[modello]["df_report_cm_train"].shape != (0,0):
            
            # Per ogni modello estrarre il sommario relativo alla condizione scelta
            model_summary = pd.DataFrame(self.risultati[modello]["df_report_cm_train"].loc[condition])
            model_summary.rename(columns={condition:modello},inplace=True)         

            # Inserire tutte le condizioni in un unico dataframe
            summary = pd.concat([summary,model_summary],axis=1)

      summary = summary.T

      return summary

   def get_good_results(self,thresholds = None):
      """Estrarre solo i modelli nei quali le performance sono buone.
         Threshold per identificare i modelli con buone performance.
      """
      pass


# C'è da lavorare sull'output, aggiungere modelli, aggiungere le versioni con tuning dei modelli, commentare, inserire le doc string, capire come richiamare i vari modelli, inserire  RAISE ERROR, la creazione di report la escludo sarà fatta direttamente in fase di utilizzo per essere più flessibile.


