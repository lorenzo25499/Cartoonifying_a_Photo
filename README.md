
# CARTOONIFYING_A_PHOTO

Questo progetto è stato testato con la versione Python 3.10.4.

Per l'uso si consiglia una versione almeno pari a quella scritta sopra.


Per questo progetto sono neccesarie le seguenti librerie:
- OpenCV
- Numpy


Per installarle è necesario digitare da riga di comando:
- $ pip install opencv-python
- $ py -m pip install numpy

Oppure:
- $ pip install opencv-python==versione
- $ py -m pip install numpy

Per eseguire il codice digitare da riga di comando della directory il seguente comando:
- $ py main.py input/<nome_immagine>.jpg


Le immagini per testare il codice sono nella cartella "input"

Il programma fa uso di diversi steps per creare l'effetto desiderato:
- Applicazione di un median filter all'immagine data in input per ridurne il rumore
- Conversione dell' immagine in scala di grigi
- Applicazione di un rilevatore di bordi: in questo caso è stato scelto il Canny Edge Detector:
          1- Applicazione di un gaussian filter per ridurre il rumore.
          2- Calcolo del gradiente con i Sobel filters.
          3- Assottigliamento dei bordi con la NMS (Non Maximum Suppression).
          4- Thresholding.
          5- Hysteresis.

- Dilatazione dei bordi trovati
- Riduzione delle dimensioni dell' immagine
- Applicazione di un bilateral filter sull' immagine a colori ridotta
- Applicazione della bilinear resize  per riportare l' immagine alle dimensioni originali
- Applicazione di un median filter all'immagine
- Quantize Color per dare l'effetto "cartoon"
- Sovrapporre i bordi trovati con il Canny Edge Detector sull' immagine a colori per avere l' effetto desiderato
          
          
**l'esecuzione richiede circa 45 min dovendo controllare ogni singolo pixel**
