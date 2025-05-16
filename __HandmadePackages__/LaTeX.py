import pandas as pd 
import numpy as np 
import subprocess
import os
import time
import re
class LaTeX:
   """Classe per creare report LaTeX.
      Permette di inizializzare un oggetto LaTeX e di aggiungerci:
      - titolo
      - indice
      - testo
      - dataframe in formato tabelle
      - sezioni
      - sottosezioni
      - sottosottosezioni
      - nuova pagina

      Opzionalmente permette di:
         - scegliere il formato del documento, se verticale o orizzontale.
         - specificare le distanze dai margini dal foglio
   """

   def __grassetto__(testo : str):
      """Per ottenere dei print python in grassetto.
         Input la stringa che deve essere in grassetto.  
         Output la stringa sarà in grassetto.
      """
      return  f"\033[1m{testo}\033[0m"

   def __colore__(testo : str ,colore : str ):
      """Per ottenere dei print python con dei colori.
         Input la stringa che deve essere in grassetto.  
         Output la stringa sarà in grassetto.
      """
      colori = {'nero': '\033[30m',
               'rosso': '\033[31m',
               'verde': '\033[32m',
               'giallo': '\033[33m',
               'blu': '\033[34m',
               'magenta': '\033[35m',
               'ciano': '\033[36m',
               'bianco': '\033[37m',
               'grigio_chiaro': '\033[90m',
               'rosso_chiaro': '\033[91m',
               'verde_chiaro': '\033[92m',
               'giallo_chiaro': '\033[93m',
               'blu_chiaro': '\033[94m',
               'magenta_chiaro': '\033[95m',
               'ciano_chiaro': '\033[96m',
               'bianco_luminoso': '\033[97m'
            }

      # region Controllo parametri
      if not isinstance(testo,str):
         raise TypeError(f"Il {self.__grassetto__('testo')} deve essere una stringa")

      if not isinstance(colore,str):
         raise TypeError(f"Il {self.__grassetto__('colore')} deve essere una stringa")


      if colore not in list(colori.keys()): 
         raise ValueError(f"Selezionare un {self.__grassetto__('colore')} tra {list(colori.keys())}")

      # endregion 

      return  f"{colori[colore]}{testo}\033[0m" 

   def __init__(self,
               formato : str = "V",
               titolo_report: str = "",
               indice=False,
               geometry : str = 'left= 3.81cm,right=2.54cm,top=2.54cm,bottom=3.81cm',
               autore : str = "",
               data : str = ""
               ):
      """ Inizializzare il report.
          Input:
            - formato ("V","H"): V per formato verticale classico,  H per formato orizzontale
            - titolo_report (opzionale): titolo da inserire nel report
            - indice (True,False): se inserire un indice 
            - geometry: speficare la distanza dei margini
            - autore: specificare l'autore del report
            - data: data da inserire nel report

      Ad esempio: doc = LaTeX(formato="H",titolo_report="",indice=False,autore="Mattia",data="\\today")
      """


      # region Controlli parametri

      ## Controllare FORMATO
      if not isinstance(formato,str):
         raise ValueError(f"La variabile  {self.__grassetto__('formato')} deve essere una stringa")

      formati_possibili = ["V","H"]
      if formato not in formati_possibili:
         raise ValueError(f"Valore per il parametro {self.__grassetto__('formato')} sbagliato. Valori possibili per la variabile formato: {formati_possibili}")

      ## Controllare  TITOLO REPORT
      if not isinstance(titolo_report,str):
         raise TypeError(f"La variabile  {self.__grassetto__('titolo_report')} deve essere una stringa")

      ## Controllare INDICE
      if not isinstance(indice,bool):
         raise TypeError(f"La variabile  {self.__grassetto__('indice')} deve essere una booleano")

      ## Controllare GEOMETRY
      if not isinstance(geometry,str):
         raise TypeError (f"La variabile  {self.__grassetto__('geometry')} deve essere una stringa")

      ## Controllare AUTORE
      if not isinstance(autore,str):
         raise TypeError (f"La variabile  {self.__grassetto__('autore')} deve essere una stringa")

      ## Controllare DATA
      if not isinstance(data,str):
         raise TypeError(f"La variabile  {self.__grassetto__('data')} deve essere una stringa")
      # endregion

      # region Costruzione File LaTeX
      # Iniziare intestazione 
      self.file_latex = """ \\documentclass{article}
                             \\usepackage{graphicx} 
                             \\usepackage{lscape}  
                             \\usepackage{geometry} 
                             \\usepackage{hyperref} 
                              \\hypersetup{
                                 colorlinks=false,
                                 pdfborder={0 0 0},
                              }\n\n       

                             \\setcounter{tocdepth}{2} 
                          """  

      # Aggiungere le distanze dai margini
      geometry_to_add = " \n \\geometry{" + geometry + "}\n "
      self.file_latex += geometry_to_add


      # Aggiungere il titolo se specificato
      if len(titolo_report) > 0:
         self.file_latex += "\n \\title{" + titolo_report + "}\n "

      # Aggiungere l'autore se specificato
      if len(autore) > 0:
         self.file_latex += "\n \\author{" + autore + "}\n "

      # Aggiungere la data se specificata
      if len(data) > 0:
         self.file_latex += "\n \\date{" + data + "}\n "

      # Se il formato specificato è orizzontale 
      if formato == "H": 
         self.file_latex += "\n \\geometry{landscape}\n"


      # Aggiungere inizio documento
      self.file_latex += "\n \\begin{document}\n"
      

      # Se è stato specificato un titolo o un indice specificare le pagine successive in numeri romani
      if (len(titolo_report) > 0) or (indice==True):
         self.file_latex += "\n \\pagenumbering{Roman}\n"

      # Aggiungere titolo
      if len(titolo_report) > 0:
         self.file_latex += "\n \\maketitle \n"
         self.file_latex += "\n \\newpage \n"


      # Aggiungere indice
      if indice:
         self.file_latex += "\n  \\tableofcontents \n"
         self.file_latex += "\n  \\newpage \n"

      # Se è stato specificato un titolo o un indice specificare le pagine successive in numeri arabi
      if (len(titolo_report) > 0) or (indice==True):
         self.file_latex += """\n  \\setcounter{page}{1} \n 
                              \n    \\pagenumbering{arabic} \n 
                              """
      # endregion

   def add_newpage(self):
      """Aggiungere una nuova pagina"""
      self.file_latex += "\n \\newpage \n"

   def add_text(self,testo : str ):
      """Aggiungere del testo"""
      if not isinstance(testo,str):
               raise TypeError(f"Specificare una stringa per il parametro {self.__grassetto__('testo')}")
      self.file_latex += testo

      self.file_latex += "\n"

   def add_section(self,nome_sezione : str):
      """ Aggiungere una sezione con il suo titolo"""

      if not isinstance(nome_sezione,str):
         raise TypeError(f"Specificare una stringa per il parametro {self.__grassetto__('nome_sezione')}")

      self.file_latex += "\\section{" + nome_sezione + "} \n"

   def add_subsection(self,nome_subsection:str):
      """ Aggiungere una subsection con il suo titolo"""

      if not isinstance(nome_subsection,str):
         raise TypeError(f"Specificare una stringa per il parametro {self.__grassetto__('nome_subsection')}")

      self.file_latex += "\\subsection{" + nome_subsection + "} \n"

   def add_subsubsection(self,nome_subsubsection:str):
      """ Aggiungere una subsubsection con il suo titolo"""

      if not isinstance(nome_subsubsection,str):
         raise TypeError(f"Specificare una stringa per il parametro {self.__grassetto__('nome_subsubsection')}")

      self.file_latex += "\\subsubsection{" + nome_subsubsection + "} \n"

   def add_table(self,
                 dataframe : pd.DataFrame(),
                 include_index : bool = True, 
                 intestazione_tabella : str = ""):
      """Converte un pandas dataframe in una tabella LaTeX
         Input:
            - dataframe: dataframe da convertire in una tabella LaTeX
            - include_index (opzionale, default=True): se includere l'indice della tabella o meno 
            - intestazione_tabella (opzionale): se aggiungere un titolo alla tabella
      """

      # region parametri di controllo input
      if not isinstance(include_index,bool):
         raise TypeError(f"Specificare una booleano per il parametro {self.__grassetto__('include_index')}")

      if not isinstance(intestazione_tabella,str):
         raise TypeError(f"Specificare una stringa per il parametro {self.__grassetto__('intestazione_tabella')}")


      if not isinstance(dataframe, pd.DataFrame):
         raise TypeError(f"La variabile {self.__grassetto__('dataframe')} deve essere un pandas DataFrame")
      # endregion

      # region Arrotondare al terzo decimale se la colonna è un float, sennò lasciare così
      for row in dataframe.columns:

         valori = dataframe[row].values
         tipi_valori = list(set([type(val) for val in valori]))

         if len(tipi_valori) == 1:

            if tipi_valori[0] == np.float64:
               dataframe[row] = dataframe[row].map(lambda x: f"{x:.3f}" if isinstance(x, float) else x)
      # endregion  

      # Conversione del DataFrame in formato LaTeX
      column_format = '|l'* (dataframe.shape[1]+1) + "|" # stabilire il formato delle colonne
      latex_output = dataframe.to_latex(index=include_index, escape=False, longtable=False,column_format=column_format)

      # region Tutte le colonne in grassetto
      for colonna in dataframe.columns:
         # Aggiungere personalizzazioni al codice LaTeX
         correzione_colonna = "\\textbf{" + colonna + "}"
         latex_output = latex_output.replace(colonna, correzione_colonna)

      latex_output = latex_output.replace("_"," ")
      latex_output = latex_output.replace('\\toprule', '\\hline')  # Sostituisci top rule
      latex_output = latex_output.replace('\\midrule', '\\hline')  # Sostituisci mid rule
      latex_output = latex_output.replace('\\bottomrule', '\\hline')  # Sostituisci bottom rule
      # endregion

      # region Aggiungere la tabella al file LaTeX
      if len(intestazione_tabella) > 0:
         latex_table = '\n\n\\begin{table}[h!]\n\\centering\n\\caption{' + intestazione_tabella + '} '+ latex_output +' \n\n\n\\end{table}'


      else:
         latex_table = '\n\n\\begin{table}[h!]\n\\centering'+ latex_output +'\n\n\n\\end{table}'

      latex_table = latex_table.replace("#","\\#")
      latex_table = latex_table.replace("_","\\_")

      self.file_latex += '\n'
      self.file_latex +=  latex_table
      self.file_latex += '\n'
      # endregion 

   def export(self,
              path : str,
              knit : bool = False
            ):
      """ Esportare il file LaTeX
          Input:
           - path: path + nome del file (con estensione) ;
           - knit (opzionale, default False): se False crea solo il file .tex, se True fa anche il knit.
      """
      if not isinstance(path,str):
         raise TypeError(f"Il {self.__grassetto__('path')} deve essere una stringa")

      self.file_latex += "\end{document}"

      # Separare dal path la cartella ove si vuole salvare ed il nome del file
      percorso_cartella = re.sub(r"/[^/]+$", "", path) # Nome Cartella
      nome_file =  os.path.basename(path)              # Nome File LaTeX 

      # 1. Salvare il percorso file attuale, utile per poi potervici ritornare.
      first_path = os.getcwd()  
 
      # 2. Spostarsi al percorso file della cartella ove si vuole salvare il file.
      os.chdir(percorso_cartella)      

      # 3. Creare una cartella intitola LaTeX se non esiste già.
      if 'LaTeX' not in os.listdir():
         os.mkdir('LaTeX')   
         
      # 4. Spostarsi nella cartella LaTeX
      os.chdir('LaTeX')      
      
      # 5. Creare il file latex se non esiste già
      if nome_file not in os.listdir():
         with open(f"{percorso_cartella}/LaTeX/{nome_file}",'w+') as file:
            file.write(self.file_latex)
            
         if knit: # Se Knit = True si procede al knit del file tex e quindi alla creazione del pdf
            with open(os.devnull, 'w') as FNULL:
               subprocess.run(["pdflatex", f"{percorso_cartella}/LaTeX/{nome_file}"], stdout=FNULL, stderr=FNULL)
      else: # Se il file esiste già:
         
         # Copiare il nome file
         new_nome_file = nome_file 
         
         # Iterare all'infinito finchè non si trova un nome che vada bene.
         i = 2 # I nuovi nomi saranno quello iniziale con l'aggiunta di una costante
         while True: 
            new_nome_file = nome_file.replace(".tex","") # Rimuovere l'estensione
            new_nome_file = f"{new_nome_file}_{i}.tex" # Aggiornare il nome del nuovo file: aggiungo COSTANTE ed l'estensione
            
            # Se il nuovo file non è presente nella directory salvarlo
            if new_nome_file not in os.listdir():
               with open(f"{percorso_cartella}/LaTeX/{new_nome_file}",'w+') as file:
                  file.write(self.file_latex)
               
               if knit: # Se Knit = True si procede al knit del file tex e quindi alla creazione del pdf
                  with open(os.devnull, 'w') as FNULL:
                     subprocess.run(["pdflatex", f"{percorso_cartella}/LaTeX/{new_nome_file}"], stdout=FNULL, stderr=FNULL)
               # Descrivere l'accaduto e printare il nuovo nome
               print(f"File {nome_file} già presente, salvato con nome: {new_nome_file}")
               break # Uscire dal while loop
            
            i +=1 # Aggiornare la costante
  

      # 6. Ritornare alla directory originaria per evitare errori.
      os.chdir(first_path)

if __name__ == "__main__":
   # Aggiungere il pacchetto allo script
   #sys.path.append('/Users/mattia/Desktop/Università/Data Science in Python/__pacchetti__')
   #from LaTeX import LaTeX # Importare la classe dedicata ai file LaTeX

   # Per creare un oggetto LaTeX
   doc = LaTeX(formato="V",
               titolo_report="Report semestrale",
               indice=True,
               autore="Mattia Manna",
               data="\\today",
               geometry="left=1cm"
   )

   # Aggiungere una section
   doc.add_section("Nome section")

   # Aggiungere una subsection
   doc.add_subsection("Nome subsection")

   # Aggiungere una subsubsection
   doc.add_subsection("Nome subsubsection")

   # Aggiungere testo
   doc.add_text("Testo da inserire")

   # Nuova pagina
   doc.add_newpage()
   doc.add_section("Nuova sezione e tabella")

   # Inizializzare un dataframe per provare la funzione "add_table"
   df = pd.DataFrame({"Colonna 1":[1,2,3],"Colonna 2":[3,4,5]})

   # Aggiungere un pandas dataframe sotto forma di tabella
   doc.add_table(df,include_index=False,intestazione_tabella="Tabella di prova")

   # Esportare il file LaTeX, va specificato il percorso file
   doc.export("/Users/mattia/Desktop/Università/Magistrale/Tesi/Python Data Leakage GSE183635/fileprova.tex")