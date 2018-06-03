//
// Created by Mariangela Evangelista on 20/05/18.
//

#ifndef PROVA_EIID_H
#define PROVA_EIID_H


#include <opencv2/core/mat.hpp>

namespace vessels {


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
    * Implementa la somma di bottom-hat sull'immagine image. La black top-hat
    * enfatizza strutture meno luminose rispetto al vicinato.
    * In base all'angolazione angle scelta, sono creati esattamente un numero di elementi
    * strutturali pari a angle ruotati tra di loro e con una distanza angolare tra essi pari
    * a 360°/angle. Per ciascun elemento strutturale viene applicata la trasformata bottom-hat
    * sull'immagine, sommando poi ciascun risultato.
    *
    * @param image Immagine su cui applicare la somma di black top-hat
    * @param width Ampiezza dello SE da utilizzare per la trasformata bottom hat
    * @param height Altezza dello SE da utilizzare per la trasformata bottom hat
    * @param angle Determina il numero di SE da utilizzare per implementare la somma di bottom hat
    * @return Il risulto della somma di bottom hat
    */
    cv::Mat sum_buttom_hat(const cv::Mat& image, int width, int height, int angle);


    /**
     * Rende nere tutte le strutture in image al di fuori della maschera.
     *
     * @param image L'immagine su cui eliminare eventuali strutture al di fuori della maschera
     * @param mask La maschera associata ad image
     * @return Un'immagine in cui sono rimosse le strutture al di fuori della maschera
     */
    cv::Mat external_bound_remove(cv::Mat image, cv::Mat mask);


    /**
     * * Realizza la segmentazione di vasi sanguigni mediante il seguente algoritmo:
     *
     * - denoising mediante l'algoritmo di Non Local Means Denoising;
     * - eliminazione di piccole regioni luminose appartenenti al background e omogenizzazione delle varie
     *   strutture mediante la ricostruzione morfologica tramite dilatazione;
     * - somma di bottom hat;
     * - denoising con l'algoritmo di Non local Means Denoising;
     * - binarizzazione mediante Otsu;
     * - eliminazione di eventuali strutture formatesi sul contorno del FOV mediante erosione della maschera;
     * - eliminazione di eventuali falsi positivi;
     * - hole filling.
     *
     * La funzione stampa inoltre l'accuracy ottenuta sul dataset scelto e salva le immagini risultato
     * (binarizzate e in pseudocolori) in corrispondenza dei due path passati in ingresso.
     *
     * @param dataset Dataset di immagini su cui applicare la segmentazione dei vasi sanguigni. È un intero
     * che identifica univocamente un dataset (definizione in dataset.h).
     * @param dataset_path Path del dataset
     * @param visual_results_path Path della cartella in sui salvare le immagini in pseudocolori
     * @param binary_results_path Path della cartella in cui salvare le immagini binarizzare
     */
    void vessels_segmentation(int dataset, std::string dataset_path, std::string visual_results_path, std::string binary_results_path);


    /**
     * Viene effettuato il calcolo delle componenti connesse e, per ogni contorno trovato
     * viene calcolata area e circolarità. I contorni aventi area inferiore ad 'area' e
     * circolarità superiore a 'circularity' sono eliminati (associando ad ogni punto interno
     * al contorno il livello di grigio 0). Successivamente, sempre mediante il calcolo delle
     * componenti connesse, si individuano contorni aventi una certa area massima e ai punti
     * interni a questi contorni viene associato un livello di grigio 255. Viene così implementato
     * il filling di eventuali buchi contenuti all'interno dei vasi sanguigni.
     *
     * @param image Immagine su cui effettuare il post-processing
     * @param area Area massima che devono avere i contorni da eliminare
     * @param circularity Circolarità minima che devono avere i contorni per essere eliminati
     * @return Immagine dopo il post-processing (8-bit)
     */
    cv::Mat post_processing(cv::Mat image, double area, double circularity);

}


#endif //PROVA_EIID_H
