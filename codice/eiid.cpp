//
// Created by Mariangela Evangelista on 20/05/18.
//

#include <opencv2/core/mat.hpp>
#include <opencv/cv.hpp>
#include "utility.h"
#include "parameters.h"
#include "3rdparty/aiacommon/aiaConfig.h"

namespace eiid {

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

            // geodesic dilation ( = dilation + pointwise minimum with mask)
            cv::morphologyEx(marker, marker, CV_MOP_DILATE, cv::getStructuringElement(CV_SHAPE_RECT, cv::Size(3,3)));
            marker = cv::min(marker, image);

        } while( cv::countNonZero(marker - marker_prev) > 0);

        return marker;
    }


    cv::Mat sum_top_hat(const cv::Mat image, int width, int height, int angle) {


        cv::Mat image_reverse, top_hat, destination, top_hat_transform, result;

        // Inversione dell'immagine image
        image_reverse = 255 - image;

        // Definizione della matrice in cui verrà salvato la somma dei risultati della top-hat transform
        top_hat = cv::Mat::zeros(image.size(), 0);

        // Memorizzazione degli elementi strutturali da utilizzare per la somma di white top hat
        std::vector<cv::Mat> tilted_SE;
        tilted_SE = createTiltedStructuringElements(width, height, angle);

        // Per ogni elemento strutturale, viene applicata la trasformata top-hat. Il risultato viene sommato
        // ai risultati precedentemente ottenuti
        for (int i = 0; i < tilted_SE.size(); i++) {
            cv::morphologyEx(image_reverse, destination, cv::MORPH_TOPHAT, tilted_SE[i]);
            top_hat = top_hat + destination;
        }

        top_hat_transform = image_reverse - top_hat;
        result = image_reverse - top_hat_transform;

        return result;
    }

    cv::Mat sum_buttom_hat(const cv::Mat image, int width, int height, int angle) {


        cv::Mat image_reverse, top_hat, destination, top_hat_transform, result;

        // Inversione dell'immagine image
        //image_reverse = 255 - image;
        image_reverse = image.clone();

        // Definizione della matrice in cui verrà salvato la somma dei risultati della top-hat transform
        top_hat = cv::Mat::zeros(image.size(), 0);

        // Memorizzazione degli elementi strutturali da utilizzare per la somma di white top hat
        std::vector<cv::Mat> tilted_SE;
        tilted_SE = createTiltedStructuringElements(width, height, angle);

        // Per ogni elemento strutturale, viene applicata la trasformata top-hat. Il risultato viene sommato
        // ai risultati precedentemente ottenuti
        for (int i = 0; i < tilted_SE.size(); i++) {
            cv::morphologyEx(image_reverse, destination, cv::MORPH_BLACKHAT, tilted_SE[i]);
            top_hat = top_hat + destination;
        }

        //top_hat_transform = image_reverse - top_hat;
        //result = image_reverse - top_hat_transform;

        return top_hat;
    }


    cv::Mat external_bound_remove(cv::Mat image, cv::Mat mask) {

        cv::Mat image_copy = image.clone();

        // Per ciascun pixel in posizione (x,y) nella maschera mask con livello di grigio 0,
        // il pixel corrispondente nell'immagine image viene posto a 0.
        for (int y = 0; y < image_copy.rows; y++) {
            unsigned char* y_row_image = image_copy.ptr<unsigned char>(y);
            unsigned char* y_row_mask = mask.ptr<unsigned char>(y);

            for (int x = 0; x < image_copy.cols; x++) {

                if (y_row_mask[x] == 0) {

                    y_row_image[x] = 0;
                }
            }
        }

        return image_copy;
    }


    void vessels_segmentation(int dataset){

        std::string path, image_extension, mask_extension, ground_truth_extension, visual_results_path, binarized_results_path;
        int se_size_for_mask_erosion, se_size_for_reconstruction, se_width_for_sum_top_hat, se_height_for_sum_top_hat, rotation_angle_for_sum_top_hat;

        // Inizializzazione dei parametri per le elaborazioni delle immagini in base al dataset scelto

        switch (dataset) {

            case  DRIVE :
                path = DRIVE_PATH;
                image_extension = DRIVE_EXTENSION;
                mask_extension = DRIVE_EXTENSION;
                ground_truth_extension = DRIVE_EXTENSION;
                visual_results_path = DRIVE_VIS_RESULTS_PATH;
                binarized_results_path = DRIVE_BIN_RESULTS_PATH;
                se_size_for_mask_erosion = DRIVE_SE_SIZE_MASK_EROSION;
                se_size_for_reconstruction = DRIVE_SE_SIZE_RECONSTRUCTION_EROSION;
                se_width_for_sum_top_hat = DRIVE_SE_LENGHT_SUMTOPHAT;
                se_height_for_sum_top_hat = DRIVE_SE_HEIGHT_SUMTOPHAT;
                rotation_angle_for_sum_top_hat = DRIVE_ANGLES_SUMTOPHAT;
                break;

            case CHASEDB1 :
                path = CHASEDB1_PATH;
                image_extension = CHASEDB1_IMAGES_EXTENSION;
                mask_extension = CHASEDB1_MASKS_EXTENSION;
                ground_truth_extension = CHASEDB1_TRUTHS_EXTENSION;
                visual_results_path = CHASEDB1_VIS_RESULTS_PATH;
                binarized_results_path = CHASEDB1_BIN_RESULTS_PATH;
                se_size_for_mask_erosion = CHASEDB1_SE_SIZE_MASK_EROSION;
                se_size_for_reconstruction = CHASEDB1_SE_SIZE_RECONSTRUCTION_EROSION;
                se_width_for_sum_top_hat = CHASEDB1_SE_LENGHT_SUMTOPHAT;
                se_height_for_sum_top_hat = CHASEDB1_SE_HEIGHT_SUMTOPHAT;
                rotation_angle_for_sum_top_hat = CHASEDB1_ANGLES_SUMTOPHAT;
                break;

            case STARE :
                path = STARE_PATH;
                image_extension = STARE_EXTENSION;
                mask_extension = STARE_MASKS_EXTENSION;
                ground_truth_extension = STARE_TRUTHS_EXTENSION;
                visual_results_path = STARE_VIS_RESULTS_PATH;
                binarized_results_path = STARE_BIN_RESULTS_PATH;
                se_size_for_mask_erosion = STARE_SE_SIZE_MASK_EROSION;
                se_size_for_reconstruction = STARE_SE_SIZE_RECONSTRUCTION_EROSION;
                se_width_for_sum_top_hat = STARE_SE_LENGHT_SUMTOPHAT;
                se_height_for_sum_top_hat = STARE_SE_HEIGHT_SUMTOPHAT;
                rotation_angle_for_sum_top_hat = STARE_ANGLES_SUMTOPHAT;
                break;

            default:
                printf("Dataset errato\n");
                return;

        }

        // Caricamento delle immagini
        std::vector <cv::Mat> images = getImagesInFolder(path + "images", image_extension);
        std::vector <cv::Mat> truthsMultiChannel = getImagesInFolder(path + "groundtruths", ground_truth_extension);
        std::vector <cv::Mat> masksMultiChannel  = getImagesInFolder(path + "masks", mask_extension);
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
            //aia::imshow("green channel", green_channel);


            std::string p1= "/Users/Mariangela/Desktop/Università/Magistrale/1 anno - 2 semestre/EIID/AIA-Retinal-Vessel-Segmentation/datasets/DRIVE/canale_rosso/canalerosso" + std::to_string(i+1) + ".tif";
          //  cv::imwrite(p1, channels[0]);
            p1= "/Users/Mariangela/Desktop/Università/Magistrale/1 anno - 2 semestre/EIID/AIA-Retinal-Vessel-Segmentation/datasets/DRIVE/canale_verde/canaleverde" + std::to_string(i+1) + ".tif";
           // cv::imwrite(p1, channels[1]);
            p1= "/Users/Mariangela/Desktop/Università/Magistrale/1 anno - 2 semestre/EIID/AIA-Retinal-Vessel-Segmentation/datasets/DRIVE/canale_blu/canaleblu" + std::to_string(i+1) + ".tif";
            //cv::imwrite(p1, channels[2]);

            green_channel.convertTo(green_channel, CV_8U);
            //truths[i].convertTo(truths[i], CV_8U);
            //aia::imshow("green channel", green_channel);


            channels.clear();
            cv::split(masksMultiChannel[i], channels);
            masks.push_back(channels[0]);

            channels.clear();
            cv::split(truthsMultiChannel[i], channels);
            truths.push_back(channels[0]);

            //aia::imshow("green channel", green_channel);



            ///NOISE REDUCTION
            // Eliminazione del rumore di sfondo preservando le strutture contenute nell'immagine (essendo il non local
            // means edge-preserving

            cv::Mat denoised;
            cv::fastNlMeansDenoising(green_channel, denoised, 3); aia::imshow("denoising", denoised);

            p1= "/Users/Mariangela/Desktop/Università/Magistrale/1 anno - 2 semestre/EIID/AIA-Retinal-Vessel-Segmentation/datasets/DRIVE/denoising1/denoising1" + std::to_string(i+1) + ".tif";
            //cv::imwrite(p1, denoised);


            ///ELIMINATE IRREGULAR BRIGHT REGIONS
            // Eliminazione delle componenti più luminose dell'immagine, omogenizzando le strutture
            // che compongono l'immagine stessa (vasi sanguigni e sfondo) tramite il metodo di
            // riscostruzione morfologica mediante dilatazione


            cv::Mat marker;
            marker = get_marker_from_reconstruction(denoised, se_size_for_reconstruction, se_size_for_reconstruction);
            aia::imshow("No bright regions", marker);


            p1= "/Users/Mariangela/Desktop/Università/Magistrale/1 anno - 2 semestre/EIID/AIA-Retinal-Vessel-Segmentation/datasets/DRIVE/marker/marker" + std::to_string(i+1) + ".tif";
            //cv::imwrite(p1, marker);


            ///SUM TOP-HAT
            // Enfatizzazione dei vasi sanguigni

            //cv::Mat sum_of_top_hat = sum_top_hat(marker, se_width_for_sum_top_hat, se_height_for_sum_top_hat, rotation_angle_for_sum_top_hat);
            cv::Mat sum_of_top_hat = sum_buttom_hat(marker, se_width_for_sum_top_hat, se_height_for_sum_top_hat, rotation_angle_for_sum_top_hat);
            aia::imshow("top hat", sum_of_top_hat);

            p1= "/Users/Mariangela/Desktop/Università/Magistrale/1 anno - 2 semestre/EIID/AIA-Retinal-Vessel-Segmentation/datasets/DRIVE/top_hat/top_hat" + std::to_string(i+1) + ".tif";
            cv::imwrite(p1, sum_of_top_hat);


            ///NOISE REDUCTION

            cv::fastNlMeansDenoising(sum_of_top_hat, sum_of_top_hat, 15);
            aia::imshow("noise reduction 2", sum_of_top_hat);

            p1= "/Users/Mariangela/Desktop/Università/Magistrale/1 anno - 2 semestre/EIID/AIA-Retinal-Vessel-Segmentation/datasets/DRIVE/denoising2/denoising_due" + std::to_string(i+1) + ".tif";
            cv::imwrite(p1, sum_of_top_hat);


            ///THRESHOLDING
            // Binarizzazione mediante l'algoritmo di otsu

            cv::Mat thresholded;
            cv::threshold(sum_of_top_hat, thresholded, 0, 255, CV_THRESH_BINARY | CV_THRESH_OTSU);
            //aia::imshow("threshold", thresholded);

            p1= "/Users/Mariangela/Desktop/Università/Magistrale/1 anno - 2 semestre/EIID/AIA-Retinal-Vessel-Segmentation/datasets/DRIVE/con_contorno/con_contorno" + std::to_string(i+1) + ".tif";
            //cv::imwrite(p1, thresholded);


            ///MASK EROSION
            // Erosione della maschera. Questo passaggio ha lo scopo di restringere il fov in maniera tale da poter
            // eliminare nel passo successivo dei falsi positivi che si trovano sul contorno del fov stesso

            cv::Mat mask_eroded;
            cv::morphologyEx(masks[i], mask_eroded, CV_MOP_ERODE, cv::getStructuringElement(CV_SHAPE_ELLIPSE, cv::Size(se_size_for_mask_erosion, se_size_for_mask_erosion)));


            thresholded = external_bound_remove(thresholded, mask_eroded);


            results.push_back(thresholded);

            printf("%d\n", i + 1);
        }

        // Calcolo dell'accuracy e generazione di immmagini in pseudocolori che mettono in evidenza TP, TN, FP e FN.

        std::vector <cv::Mat> visual_results;
        double ACC = accuracy(results, truths, masks, &visual_results);
        printf("Accuracy = %.2f%%\n", ACC*100);

        // Salvataggio delle immagini binarizzate e delle immagini in pseudocolori

        for (int i = 0; i < visual_results.size(); i++) {

            std::string path_name_visual_results = visual_results_path + std::to_string(i+1) + ".tif";
            cv::imwrite(path_name_visual_results, visual_results[i]);


            std::string path_name_results = binarized_results_path + std::to_string(i+1) + ".tif";
            cv::imwrite(path_name_results, results[i]);
        }


    }

};