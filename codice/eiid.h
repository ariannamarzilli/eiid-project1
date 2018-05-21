//
// Created by Mariangela Evangelista on 20/05/18.
//

#ifndef PROVA_EIID_H
#define PROVA_EIID_H


#include <opencv2/core/mat.hpp>

namespace eiid {

    /**
    * Realizza l'operazione di ricostruzione morfologica tramite dilatazione.
    * Come primo step, viene costruito il marker mediante un'operazione di erosione con un
    * elemento strutturale width x height. A seguire, vengono applicate successive
    * dilatazioni con un elemento strutturale 3x3 finchè il marker non subisce più alcuna modifica.
    *
    * @param image Immagine su cui applicare la ricostruzione morfologica
    * @param width Larghezza dell'elemento strutturale da utilizzare per l'erosione
    * @param height lunghezza dell'elemento strutturale da utilizzare per l'erosione
    * @return Il marker finale
    */
    cv::Mat get_marker_from_reconstruction(const cv::Mat image, int width, int height);


    /**
    * Implementa la somma di (white) top-hat sull'immagine image. La white top-hat
    * enfatizza strutture più luminose del vicinato. L'immagine image passata in ingresso
    * deve essere tale da avere le strutture da enfatizzare più scure rispetto al vicinato.
    * Come primo step l'immagine in ingresso 'image' viene invertita e, in base all'angolazione
    * angle scelta, sono creati esattamente un numero di elementi strutturali pari a angle
    * ruotati tra di loro e con una distanza angolare tra essi pari a 360°/angle. Per ciascun
    * elemento strutturale viene applicata la trasformata top-hat sull'immagine invertita,
    * sommando poi ciascun risultato.
    *
    * @param image Immagine su cui applicare la somma di (white) top-hat
    * @param width Ampiezza dello SE da utilizzare per la trasformata top hat
    * @param height Altezza dello SE da utilizzare per la trasformata top hat
    * @param angle Determina il numero di SE da utilizzare per implementare la somma di top hat
    * @return Il risulto della somma di white top hat
    */
    cv::Mat sum_top_hat(const cv::Mat& image, int width, int height, int angle);


    /**
     * Rende nere tutte le strutture in image al di fuori della maschera.
     *
     * @param image L'immagine su cui eliminare eventuali strutture al di fuori della maschera
     * @param mask La maschera associata ad image
     * @return Un'immagine in cui sono rimosse le strutture al di fuori della maschera
     */
    cv::Mat external_bound_remove(cv::Mat image, cv::Mat mask);

    /**
     * Realizza la segmentazione di vasi sanguigni mediante il seguente algoritmo:
     *
     * - denoising mediante l'algoritmo di Non Local Means Denoising;
     * - eliminazione di piccole regioni luminose appartenenti al background e omogenizzazione delle varie
     *   strutture mediante la ricostruzione morfologica tramite dilatazione;
     * - somma di white top hat;
     * - denoising con l'algoritmo di Non local Means Denoising;
     * - binarizzazione mediante otsu;
     * - eliminazione di eventuali strutture formatesi sul contorno del FOV mediante erosione della maschera.
     *
     * La funzione stampa inoltre l'accuracy ottenuta sul dataset scelto
     *
     * @param dataset Dataset di immagini su cui applicare la segmentazione dei vasi sanguigni. È un intero
     * che identifica univocamente un dataset (definizione in dataset.h).
     */
    void vessels_segmentation(int dataset);
}


#endif //PROVA_EIID_H
