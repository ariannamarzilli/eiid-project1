//
// Created by Mariangela Evangelista on 20/05/18.
//

#include <opencv2/core/mat.hpp>
#include <opencv/cv.hpp>
#include "utility.h"
#include "parameters.h"
#include "3rdparty/aiacommon/aiaConfig.h"
#include "3rdparty/ucascommon/ucasMathUtils.h"

namespace vessels {

    cv::Mat get_marker_from_reconstruction(const cv::Mat image, int width, int height) {

        // Si ottiene il marker mediante erosione
        cv::Mat eroded;
        cv::morphologyEx(image, eroded, CV_MOP_ERODE, cv::getStructuringElement(CV_SHAPE_RECT, cv::Size(width, height)));

        // Si effettua una dilatazione iterativa dell'immagine erosa fin tanto che la dilatazione non apporta alcuna
        // modifica. In particolare l'immagine erosa funge da marker, mentre l'immagine 'image' funge da maschera.
        cv::Mat marker, marker_prev;
        marker = eroded.clone();

        do {
            // È necessario salvare il marker dell'iterazione precedente affinchè sia possibile determinare se il
            // marker attuale è cambiato o meno
            marker_prev = marker.clone();

            // dilatazione geodesica
            cv::morphologyEx(marker, marker, CV_MOP_DILATE, cv::getStructuringElement(CV_SHAPE_RECT, cv::Size(3,3)));
            marker = cv::min(marker, image);

        } while( cv::countNonZero(marker - marker_prev) > 0);

        return marker;
    }

    cv::Mat sum_buttom_hat(const cv::Mat image, int width, int height, int angle) {

        cv::Mat sum_bottom_hat, bottom_hat_transform;

        // Inizializzazione della matrice in cui verrà salvato la somma dei risultati delle bottom-hat transform
        sum_bottom_hat = cv::Mat::zeros(image.size(), 0);

        // Memorizzazione degli elementi strutturali da utilizzare per la somma di black top hat
        std::vector<cv::Mat> tilted_SE;
        tilted_SE = createTiltedStructuringElements(width, height, angle);

        // Per ogni elemento strutturale, viene applicata la trasformata black-hat. Il risultato viene sommato
        // ai risultati precedentemente ottenuti
        for (int i = 0; i < tilted_SE.size(); i++) {
            cv::morphologyEx(image, bottom_hat_transform, cv::MORPH_BLACKHAT, tilted_SE[i]);
            sum_bottom_hat = sum_bottom_hat + bottom_hat_transform;
        }

        return sum_bottom_hat;
    }

    cv::Mat external_bound_remove(cv::Mat image, cv::Mat mask) {

        cv::Mat img = image.clone();

        // Per ciascun pixel in posizione (x,y) nella maschera mask con livello di grigio 0,
        // il pixel corrispondente nell'immagine image viene posto a 0.
        for (int y = 0; y < img.rows; y++) {
            unsigned char* y_row_image = img.ptr<unsigned char>(y);
            unsigned char* y_row_mask = mask.ptr<unsigned char>(y);

            for (int x = 0; x < img.cols; x++) {

                if (y_row_mask[x] == 0) {

                    y_row_image[x] = 0;
                }
            }
        }

        return img;
    }

    cv::Mat post_processing(cv::Mat image, double area, double circularity, double area_hole_filling) {


        /// ELIMINATE LITTLE REGIONS
        cv::Mat img = image.clone();
        std::vector < std::vector <cv::Point> > contours;

        // trovo tutti i contorni nell'immagine
        cv::findContours(img, contours, CV_RETR_LIST, CV_CHAIN_APPROX_NONE);

        for (int i = 0; i < contours.size(); i++) {

            // per ogni contorno individuato viene calcolata area e circolarità
            double a = cv::contourArea(contours[i]);
            double p = cv::arcLength(contours[i], true);
            double c = 4*ucas::PI*area/ (p*p);


            // se l'area del contorno considerato è inferiore ad 'area' e la circolarità è maggiore
            // di 'circularity' tutta la regione interna al contorno viene eliminata
            if (a < area && c > circularity) {

                cv::drawContours(img, contours, i, cv::Scalar(0, 0, 0), CV_FILLED);
            }

        }

        contours.clear();

        ///FILLING
        // Vengono trovati tutti i contorni (grazie all'elaborazione precedente, la maggior parte
        // delle regioni caratterizzate da falsi positivi sono state eliminate, mentre le regioni
        // costituite da falsi negativi sono ancora presenti: sono queste ultime ad essere individuate).

        cv::findContours(img, contours, CV_RETR_LIST, CV_CHAIN_APPROX_NONE);

        for (int i = 0; i < contours.size(); i++) {

            // Per ogni contorno individuato, viene calcolata l'area e, se questa risulta essere
            // inferiore ad 'area_hole_filling', viene riempita di bianco.
            double a = cv::contourArea(contours[i]);

            if (a < area_hole_filling) {

                cv::drawContours(img, contours, i, cv::Scalar(255, 255, 255), CV_FILLED);

            }

        }

        return img;
    }

    void vessels_segmentation(int dataset, std::string dataset_path, std::string visual_results_path, std::string binary_results_path){

        // definizione variabili
        std::string image_extension, mask_extension, ground_truth_extension;
        int se_size_for_mask_erosion, se_size_for_reconstruction, se_width_for_sum_top_hat, se_height_for_sum_top_hat, rotation_angle_for_sum_top_hat, denoising_parameter;
        double area, circularity, area_hole_filling;


        // Inizializzazione dei parametri per le elaborazioni delle immagini in base al dataset scelto
        switch (dataset) {

            case  DRIVE :
                image_extension = DRIVE_EXTENSION;
                mask_extension = DRIVE_EXTENSION;
                ground_truth_extension = DRIVE_EXTENSION;
                se_size_for_mask_erosion = DRIVE_SE_SIZE_MASK_EROSION;
                se_size_for_reconstruction = DRIVE_SE_SIZE_RECONSTRUCTION_EROSION;
                se_width_for_sum_top_hat = DRIVE_SE_LENGHT_SUMTOPHAT;
                se_height_for_sum_top_hat = DRIVE_SE_HEIGHT_SUMTOPHAT;
                rotation_angle_for_sum_top_hat = DRIVE_ANGLES_SUMTOPHAT;
                denoising_parameter = DRIVE_DENOISING_2;
                area = DRIVE_AREA;
                circularity = DRIVE_CIRCULARITY;
                area_hole_filling = DRIVE_AREA_HOLE_FILLING;
                break;

            case CHASEDB1 :
                image_extension = CHASEDB1_IMAGES_EXTENSION;
                mask_extension = CHASEDB1_MASKS_EXTENSION;
                ground_truth_extension = CHASEDB1_TRUTHS_EXTENSION;
                se_size_for_mask_erosion = CHASEDB1_SE_SIZE_MASK_EROSION;
                se_size_for_reconstruction = CHASEDB1_SE_SIZE_RECONSTRUCTION_EROSION;
                se_width_for_sum_top_hat = CHASEDB1_SE_LENGHT_SUMTOPHAT;
                se_height_for_sum_top_hat = CHASEDB1_SE_HEIGHT_SUMTOPHAT;
                rotation_angle_for_sum_top_hat = CHASEDB1_ANGLES_SUMTOPHAT;
                denoising_parameter = CHASEDB1_DENOISING_2;
                area = CHASEDB1_AREA;
                circularity = CHASEDB1_CIRCULARITY;
                area_hole_filling = CHASEDB1_AREA_HOLE_FILLING;
                break;

            case STARE :
                image_extension = STARE_EXTENSION;
                mask_extension = STARE_MASKS_EXTENSION;
                ground_truth_extension = STARE_TRUTHS_EXTENSION;
                se_size_for_mask_erosion = STARE_SE_SIZE_MASK_EROSION;
                se_size_for_reconstruction = STARE_SE_SIZE_RECONSTRUCTION_EROSION;
                se_width_for_sum_top_hat = STARE_SE_LENGHT_SUMTOPHAT;
                se_height_for_sum_top_hat = STARE_SE_HEIGHT_SUMTOPHAT;
                rotation_angle_for_sum_top_hat = STARE_ANGLES_SUMTOPHAT;
                denoising_parameter = STARE_DENOISING_2;
                area = STARE_AREA;
                circularity = STARE_CIRCULARITY;
                area_hole_filling = STARE_AREA_HOLE_FILLING;
                break;

            default:
                printf("Dataset errato\n");
                return;

        }

        // Caricamento delle immagini
        std::vector <cv::Mat> images = getImagesInFolder(dataset_path + "/" + "images", image_extension);
        std::vector <cv::Mat> truths_multi_channel = getImagesInFolder(dataset_path + "/" + "groundtruths", ground_truth_extension);
        std::vector <cv::Mat> masks_multi_channel  = getImagesInFolder(dataset_path + "/" "masks", mask_extension);
        std::vector <cv::Mat> masks;
        std::vector <cv::Mat> truths;
        std::vector <cv::Mat> results;


        for (int i = 0; i < images.size(); i++) {


            ///GREEN CHANNEL EXTRACTION
            // Estrazione del canale verde dell'immagine. Tra i 3 canali, quello verde
            // presenta il maggior contrasto tra sfondo e vasi.

            std::vector<cv::Mat> channels;
            cv::split(images[i], channels);
            cv::Mat green_channel = channels[1].clone();


            green_channel.convertTo(green_channel, CV_8U);
            truths_multi_channel[i].convertTo(truths_multi_channel  [i], CV_8U);

            channels.clear();
            cv::split(masks_multi_channel[i], channels);
            masks.push_back(channels[0]);

            channels.clear();
            cv::split(truths_multi_channel[i], channels);
            truths.push_back(channels[0]);


            ///NOISE REDUCTION
            // Eliminazione del rumore di sfondo preservando le strutture contenute nell'immagine (essendo il non local
            // means edge-preserving

            cv::Mat denoised;
            cv::fastNlMeansDenoising(green_channel, denoised, 3);


            ///ELIMINATE IRREGULAR BRIGHT REGIONS
            // Eliminazione delle componenti più luminose dell'immagine, omogenizzando le strutture
            // che compongono l'immagine stessa (vasi sanguigni e sfondo) tramite il metodo di
            // riscostruzione morfologica mediante dilatazione

            cv::Mat marker;
            marker = get_marker_from_reconstruction(denoised, se_size_for_reconstruction, se_size_for_reconstruction);


            ///SUM TOP-HAT
            // Enfatizzazione dei vasi sanguigni

            cv::Mat sum_of_top_hat = sum_buttom_hat(marker, se_width_for_sum_top_hat, se_height_for_sum_top_hat, rotation_angle_for_sum_top_hat);


            ///NOISE REDUCTION
            cv::fastNlMeansDenoising(sum_of_top_hat, sum_of_top_hat, denoising_parameter);


            ///THRESHOLDING
            // Binarizzazione mediante l'algoritmo di otsu

            cv::Mat thresholded;
            cv::threshold(sum_of_top_hat, thresholded, 0, 255, CV_THRESH_BINARY | CV_THRESH_OTSU);


            ///MASK EROSION
            // Erosione della maschera. Questo passaggio ha lo scopo di restringere il fov in maniera tale da poter
            // eliminare nel passo successivo dei falsi positivi che si trovano sul contorno del fov stesso

            cv::Mat mask_eroded;
            cv::morphologyEx(masks[i], mask_eroded, CV_MOP_ERODE, cv::getStructuringElement(CV_SHAPE_ELLIPSE, cv::Size(se_size_for_mask_erosion, se_size_for_mask_erosion)));
            thresholded = external_bound_remove(thresholded, mask_eroded);

            /// Eliminazione di eventuali piccole regioni classificate erroneamente come vasi e successivo hole filling
            cv::Mat result = post_processing(thresholded, area, circularity, area_hole_filling);


            results.push_back(result);

            printf("%d\n", i + 1);

        }

        // Calcolo dell'accuracy e generazione di immmagini in pseudocolori che evidenziano TP, TN, FP e FN.


        std::vector <cv::Mat> visual_results;
        double ACC = accuracy(results, truths, masks, &visual_results);
        printf("Accuracy = %.2f%%\n", ACC*100);

        // Salvataggio delle immagini binarizzate e delle immagini in pseudocolori

        for (int i = 0; i < visual_results.size(); i++) {

            std::string path_name_visual_results = visual_results_path + "/" + std::to_string(i+1) + ".tif";
            cv::imwrite(path_name_visual_results, visual_results[i]);


            std::string path_name_results = binary_results_path + "/" + std::to_string(i+1) + ".tif";
            cv::imwrite(path_name_results, results[i]);
        }


    }

};

