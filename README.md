
Una repository contenente una classe per  costruire una pipeline atta a  valutare i subset di keygenes ed i loro metodi di selezione.		
La valutazione avviene tramite l'analisi delle metriche di performance.


I file:
- **EvaluatePipeline.py** contiene la classe Python;
- Nel folder **__HandmadePackages__**  contenente le classi e le funzioni di ausilio da me create per la classe contenenuta in **EvaluatePipeline.py**;
  	+  **current_time**: Contiene una funzione che se richiamata fornisce la data attuale, anno - mese - giorno - ora - minutii,  usata per semplicare i timestamp.
  	+ **evaluate_performance**: Contiene diverse funzioni per misurare le performance di diversi tipi di modelli di classificazione.
  	+ **LaTeX**: classe per la creazione di report pdf in python.
  	+  **preprocessing**: Funzioni per il preprocessing, ad esempio semplificare la ricodifica di variabili qualitative.

- **tryme.ipynb** Python notebook, esempio base per utilizzare la classe Python creata;
- **Esempio applicazione pratica.ipynb** python notebook che contiene un esempio di applicazione reale della classe affiancata dalla classe LaTeX per la creazione del report finale;
- **Report esempio applicazione pratica.pdf** report proveniente dallo script descritto nel punto precedente;







	

