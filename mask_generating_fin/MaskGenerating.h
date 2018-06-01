#ifndef MASKGENERATING_H
#define MASKGENERATING_H

// include aia and ucas utility functions
	#include "aiaConfig.h"
	#include "ucasConfig.h"
	#include <vector>
	#include <opencv2\photo.hpp>

	// include my project functions
	#include "functions.h"

namespace eiid {

	// Dataset path
	#define PATH "C:/Users/pc/Documents/Universita/primo_anno_secondo_semestre/EIID/project/AIA-Retinal-Vessel-Segmentation/datasets/STARE/"
	
	// Soglia per la selezione dei contorni nell'algoritmo di Canny
	#define CANNY_THRESHOLD 50
	
	// Variabile di comodo per la memorizzazione delle immagini maschera
	
	/**
    * Metodo per la generazione di immagini maschere per un dataset.
    * Come primo step, vengono caricate le immagini presenti nel dataset, quindi vengono generate le 
    * maschere per tutte le immagini caricate. In fine le maschere risultato vengono memorizzate. 
    *
    * @param str Stringa che defnisce la posizione del dataset
    */ 
	void generateMasksDataset(std::string str);
	
	
	
	/**
    * Metodo per la generazione di un'immagine singola maschera.
    * Inizialmente viene effettuata l'individuazione dei contorni di image tramite l'algoritmo di Canny e la
    * loro estrazioe tramite il calcolo delle componenti connesse. Quindi viene estratto il controno a lunghezza
    * massima (bordo esterno) e chiuso manualmente (in caso di imperfezioni dell'immagine. In fine il contorno estratto
	* viene riempito.
	*
    * @param image immagine di cui estrarre la maschera
	* @return immagine maschera
    */ 
	cv::Mat maskGenerating(const cv::Mat& image);
	
	
	/**
    * Metodo per l'individuazione del controno a dimensione massima.
    * Viene valutata la lunghezza di ogni singolo controno e viene selezionata l'indice di quello massimo
	*
    * @param contours vettore di contorni (vettri di punti)
	* @return posizione del contorno massimo
    */
	int maxContourn(std::vector<std::vector<cv::Point>> contours);
	
	
	/**
    * Metodo per il riempimento di un controno (realizza la maschera effettiva).
    * Inizialmente creo un'immagine che conterrà il risultato finale. Quindi per ogni pixel dell'immagine su detta valuto se
    * è interno o esterno al contorno. Se è interno viene posto a 255 (bianco) se è esterno viene posto  0 (nero)
	*
    * @param img immagine usata per definire le dimensioni dell'immagine in uscita
	* @param contours controno da riempirsi
	* @return immagine binaria del controno riempito (bianca nel contorno nera fuori)
    */ 
	cv::Mat createMask(const cv::Mat& img, std::vector<cv::Point> contours);
	
	
	/**
    * Metodo per l'indviduazione dei punt estrami di un controno .
    * Itero su tutti i punti del contorno finchè non trovo i punti estremi. Se tali punti sono stati trovati vengono insriti in
    * interestsPoint e nClose è posto a true altrimenti nClose è posto a false
	*
    * @param conturs controno che bisogna analizzare (è chiuso o meno?)
	* @param interestPoints vettore di punti che contiene gli estremi del contorno aperto
	* @param nClose informa se il controno è chiuso o meno	
    */ 
	void closeContourn(std::vector<cv::Point> contours, std::vector<cv::Point> &interestPoints, bool &nClose, cv::Mat img);
	
	// retrieves and loads all images within the given folder and having the given file extension
	std::vector < cv::Mat > getImagesInFolder(std::string folder, std::string ext = ".tif", bool force_gray = false) throw (aia::error);
}

#endif // MASKGENERATING_H