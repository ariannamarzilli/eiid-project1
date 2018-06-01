#include "MaskGenerating.h"

namespace eiid {

	// Variabile di comodo per la memorizzazione delle immagini maschera
	int contMasks = 0;
	
	// retrieves and loads all images within the given folder and having the given file extension
	std::vector< cv::Mat >getImagesInFolder(std::string folder, std::string ext , bool force_gray) throw (aia::error)
	{
		// check folders exist
		if(!ucas::isDirectory(folder))
			throw aia::error(aia::strprintf("in getImagesInFolder(): cannot open folder at \"%s\"", folder.c_str()));

		// get all files within folder
		std::vector < std::string > files;
		cv::glob(folder, files);

		// open files that contains 'ext'
		std::vector < cv::Mat > images;
		for(auto & f : files)
		{
			if(f.find(ext) == std::string::npos)
				continue;

			images.push_back(cv::imread(f, force_gray ? CV_LOAD_IMAGE_GRAYSCALE : CV_LOAD_IMAGE_UNCHANGED));
		}

		return images;
	}
	
	
	
	void closeContourn(std::vector<cv::Point> contours, std::vector<cv::Point> &interestPoints, bool &nClose, cv::Mat img) {

		cv::Point prevPos = contours[0];
		cv::Point nextPos = contours[2];
		int cont = 0;

		// Itero partendo dal secondo elemnto del contorno e analizzo la posizione precedente e successiva a quella attuale: se tali punti sono gli stassi
		// ho individuato un estremo, quindi memorizzo tale posizione ed incremento un contatore. Trovato il secondo punto il contatore sarà a 2 e il ciclo trìermina.
		// L'iterazione si basa sul modo con cui il metodo cv::findContours(...) individua i punti del contorno.
		for (int i = 1; i < contours.size() - 2; i++) {

		
			if (nextPos == prevPos) {

				interestPoints.push_back(contours[i]); //std::cout << contours[i] << std::endl;
				cont++;

				if(cont > 2) {

					nClose = false;
					break;
				}
			}

			nextPos = contours[i + 2];
			prevPos = contours[i];
		}

		// Se cont = 2 ho trovato un contorno aperto

		if (cont == 2) {

			nClose = true;
		}
	}
	
	
	
	cv::Mat createMask(const cv::Mat& img, std::vector<cv::Point> contours) {

	cv::Mat resultBin = img.clone();

	// Per ogni pixel dell'immagine se è contenuto nel contorno lo rendo bianco altrimenti lo rendo nero

	for (int y = 0; y < resultBin.rows; y++) {

		unsigned char* yRow = resultBin.ptr<unsigned char>(y);

		for (int x = 0; x < resultBin.cols; x++) {

			if (cv::pointPolygonTest(contours, cv::Point2f(x, y), false) >= 0) {

				yRow[x] = 255;
			} else {

				yRow[x] = 0;
				
			}
		}
	}

	return resultBin;
}



int maxContourn(std::vector<std::vector<cv::Point>> contours) {

	int pos = 0; 

	// Per ogni controno valuto se la sua lughezza è maggiore di quella del massimo attuale. In
	// caso affermativo aggiorno pos
	
	for (int i = 1; i < contours.size(); i++) {

		if(cv::arcLength(contours[i], false) > cv::arcLength(contours[pos], false)) {

			pos = i;
		}
	}

	return pos;
}


cv::Mat maskGenerating(const cv::Mat& image) {

	cv::Mat img = image.clone(); //aia::imshow("img", img);
	
	//aia::imshow("Image", img); cv::imwrite("C:/Users/pc/Desktop/Image.png", img);
	
	//aia::imshow("Morphological Smoothing", img); cv::imwrite("C:/Users/pc/Desktop/rimmaginiRelazione/vessel/mask/STARE/smooth6.png", img);


	// Applicazione dell'algoritmo di Canny, quindi viene effettuato lo smoothing dell'immagine tramite un filtro gaussiano standard
	// (media zero e varianza 1 sia lungho la direzione x che quella y) e in seguito estratti i contorni tramite la funzione cv::Canny
	
	cv::Mat imageEdges; int thresholdCanny = CANNY_THRESHOLD;//aia::imshow("img", img, true); 
	cv::GaussianBlur(img, img, cv::Size(7,7), 1,1); //aia::imshow("gaussian", img);cv::imwrite("C:/Users/pc/Desktop/smooth.jpg", img);
	cv::Canny(img, imageEdges, thresholdCanny, 3 * thresholdCanny); //cv::imwrite("C:/Users/pc/Desktop/canny.jpg", imageEdges); 
	
	//aia::imshow(" ", imageEdges); cv::imwrite("C:/Users/pc/Desktop/rimmaginiRelazione/vessel/mask/STARE/imageEdges6.png", imageEdges);

	// A seguito dell'algoritmo di Canny, l'immagine ottenuta è composta da bordi di spessore unitario. Quindi vengono individuati 
	// e viene selezionato quello di lunghezza massima
	std::vector<std::vector<cv::Point>> contoursEdges;
	cv::findContours(imageEdges, contoursEdges, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_NONE);
	int rightContourn = maxContourn(contoursEdges); cv::Mat dummy = cv::Mat(img.rows, img.cols, CV_8U, cv::Scalar(0));
	cv::drawContours(dummy, contoursEdges, rightContourn, cv::Scalar(255), 1); 
	
	//aia::imshow(" ", dummy);  cv::imwrite("C:/Users/pc/Desktop/imageEdgesSingle2.png", dummy);

	// Può capitare che il controno sia aperto (l'immagine di partenza è stata acquisita non perfettamente). Va chiuso.
	std::vector<cv::Point> interestPoints; bool toClose = false;
	closeContourn(contoursEdges[rightContourn], interestPoints, toClose, imageEdges);

	if (toClose == true) {

		cv::line(imageEdges, interestPoints[0], interestPoints[1], cv::Scalar(255), 1);
		//aia::imshow(" ", imageEdges);

		// Se il contorno risultava aperto in precedenza va determinato di nuovo

		contoursEdges.clear();

		cv::findContours(imageEdges, contoursEdges, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_NONE);
		rightContourn = maxContourn(contoursEdges); 
	}

	// cv::Mat dummy3 = image.clone(); cv::cvtColor(dummy3, dummy3, CV_GRAY2BGR);

	//cv::drawContours(dummy3, contoursEdges, rightContourn, cv::Scalar(0, 0, 255), 2);		
	//aia::imshow(" ", dummy3);  cv::imwrite("C:/Users/pc/Desktop/imageEdgesSingleclose.png", dummy3);

	// Ottenuto il controno finale vine riempito per avere un'immagine maschera
	
	cv::Mat bin = createMask(img, contoursEdges[rightContourn]); 

	//aia::imshow("bin", bin); cv::imwrite("C:/Users/pc/Desktop/bin6.png", bin);
	
	contMasks++;
	return bin;
}

void generateMasksDataset(std::string str) {

	// Carico le immagini del dataset
	std::vector <cv::Mat> images = getImagesInFolder(str + "images", ".ppm");
	std::vector<cv::Mat> masks; 

	for (int i = 0; i < images.size(); i++) {
		
		std::vector<cv::Mat> channels; //aia::imshow("img", images[i]); cv::imwrite("C:/Users/pc/Desktop/originalImage.png", images[i]);
		// Estraggo il canale verde dell'immagine
		cv::split(images[i], channels); 
		cv::Mat red_channel = channels[2]; 
		red_channel.convertTo(red_channel, CV_8U);

		// Memorizzo la maschera ricavata
		masks.push_back(maskGenerating(red_channel));

		// Salvo l'immagine (uso la variabile cont per memorizzare le stringhe in maniera corretta per la
		// fase di impiego della maschera, ovvero la sementazione dei vasi sanguigni)
		if(contMasks < 10) {
		std::string pathName = str + "masks2/0" + std::to_string(i+1) + ".tif";
		cv::imwrite(pathName, masks[i]);
		} else {
		std::string pathName = str + "masks2/" + std::to_string(i+1) + ".tif";
		cv::imwrite(pathName, masks[i]);
		}

		printf("Image saved %d\n", i + 1);
	}
}
	
}