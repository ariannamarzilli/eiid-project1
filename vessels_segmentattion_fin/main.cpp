#include "dataset.h"
#include "eiid.h"


int main() {

    /*
     *  Prima di eseguire il programma, inserire i seguenti pecorsi:
     *  - path della cartella contenente il dataset scelto;
     *  - path di una cartella in cui salvare le immagini risultato;
     *  - path di una cartella in cui salvare le immagini risultato in pseudocolori;
     *  - specificare nella funzione vessels::vessels_segmentatio il dataset scelto tra {DRIVE, CHASEDB1, STARE}.
     */

    std::string dataset_path = std::string("/Users/Mariangela/Desktop/Università/Magistrale/1 anno - 2 semestre/EIID/AIA-Retinal-Vessel-Segmentation/datasets/DRIVE");
    std::string visual_results_path = std::string("/Users/Mariangela/Desktop/Università/Magistrale/1 anno - 2 semestre/EIID/AIA-Retinal-Vessel-Segmentation/datasets/DRIVE/results");
    std::string binary_results_path = std::string("/Users/Mariangela/Desktop/Università/Magistrale/1 anno - 2 semestre/EIID/AIA-Retinal-Vessel-Segmentation/datasets/DRIVE/results_binarization");

    vessels::vessels_segmentation(DRIVE, dataset_path, visual_results_path, binary_results_path);
    return 0;
}