from datetime import datetime
def current_time():
   """Richiamata fornisce la data attuale, anno - mese - giorno - ora - minuti
      Può essere utile quando dei print/file hanno bisogno di essere datati
    """

   # Current timestamp with date, hour, and minute
   current_time = datetime.now()

   # Extracting specific components
   date = current_time.date()        # Date (YYYY-MM-DD)
   hour = current_time.hour          # Hour (0-23)
   minute = current_time.minute      # Minute (0-59)

   # Printing the result
   #print("Current Date:", date)
   #print("Hour:", hour)
   #print("Minute:", minute)

   # Formatting timestamp into a string
   formatted_time = current_time.strftime("%Y-%m-%d %H:%M")
   #print("Formatted Timestamp:", formatted_time)

   return formatted_time


if __name__ == "__main__":
   print(current_time())
