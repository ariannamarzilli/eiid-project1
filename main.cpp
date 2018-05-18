// include aia and ucas utility functions
#include "aiaConfig.h"
#include "ucasConfig.h"
#include <opencv2\photo.hpp>


// include my project functions
#include "functions.h"

#define PATH "C:/Users/pc/Documents/Universita/primo_anno_secondo_semestre/EIID/project/AIA-Retinal-Vessel-Segmentation/datasets/STARE/"
#define CANNY_THRESHOLD 30

// retrieves and loads all images within the given folder and having the given file extension
std::vector < cv::Mat > getImagesInFolder(std::string folder, std::string ext = ".tif", bool force_gray = false) throw (aia::error)
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


// Accuracy (ACC) is defined as:
// ACC = (True positives + True negatives)/(number of samples)
// i.e., as the ratio between the number of correctly classified samples (in our case, pixels)
// and the total number of samples (pixels)
double accuracy(
        std::vector <cv::Mat> & segmented_images,		// (INPUT)  segmentation results we want to evaluate (1 or more images, treated as binary)
        std::vector <cv::Mat> & groundtruth_images,     // (INPUT)  reference/manual/groundtruth segmentation images
        std::vector <cv::Mat> & mask_images,			// (INPUT)  mask images to restrict the performance evaluation within a certain region
        std::vector <cv::Mat> * visual_results = 0		// (OUTPUT) (optional) false color images displaying the comparison between automated segmentation results and groundtruth
        //          True positives = blue, True negatives = gray, False positives = yellow, False negatives = red
) throw (aia::error)
{
    // (a lot of) checks (to avoid undesired crashes of the application!)
    if(segmented_images.empty())
        throw aia::error("in accuracy(): the set of segmented images is empty");
    if(groundtruth_images.size() != segmented_images.size())
        throw aia::error(aia::strprintf("in accuracy(): the number of groundtruth images (%d) is different than the number of segmented images (%d)", groundtruth_images.size(), segmented_images.size()));
    if(mask_images.size() != segmented_images.size())
        throw aia::error(aia::strprintf("in accuracy(): the number of mask images (%d) is different than the number of segmented images (%d)", mask_images.size(), segmented_images.size()));
    for(size_t i=0; i<segmented_images.size(); i++)
    {
        if(segmented_images[i].depth() != CV_8U || segmented_images[i].channels() != 1)
            throw aia::error(aia::strprintf("in accuracy(): segmented image #%d is not a 8-bit single channel images (bitdepth = %d, nchannels = %d)", i, ucas::imdepth(segmented_images[i].depth()), segmented_images[i].channels()));
        if(!segmented_images[i].data)
            throw aia::error(aia::strprintf("in accuracy(): segmented image #%d has invalid data", i));
        if(groundtruth_images[i].depth() != CV_8U || groundtruth_images[i].channels() != 1)
            throw aia::error(aia::strprintf("in accuracy(): groundtruth image #%d is not a 8-bit single channel images (bitdepth = %d, nchannels = %d)", i, ucas::imdepth(groundtruth_images[i].depth()), groundtruth_images[i].channels()));
        if(!groundtruth_images[i].data)
            throw aia::error(aia::strprintf("in accuracy(): groundtruth image #%d has invalid data", i));
        if(mask_images[i].depth() != CV_8U || mask_images[i].channels() != 1)
            throw aia::error(aia::strprintf("in accuracy(): mask image #%d is not a 8-bit single channel images (bitdepth = %d, nchannels = %d)", i, ucas::imdepth(mask_images[i].depth()), mask_images[i].channels()));
        if(!mask_images[i].data)
            throw aia::error(aia::strprintf("in accuracy(): mask image #%d has invalid data", i));
        if(segmented_images[i].rows != groundtruth_images[i].rows || segmented_images[i].cols != groundtruth_images[i].cols)
            throw aia::error(aia::strprintf("in accuracy(): image size mismatch between %d-th segmented (%d x %d) and groundtruth (%d x %d) images", i, segmented_images[i].rows, segmented_images[i].cols, groundtruth_images[i].rows, groundtruth_images[i].cols));
        if(segmented_images[i].rows != mask_images[i].rows || segmented_images[i].cols != mask_images[i].cols)
            throw aia::error(aia::strprintf("in accuracy(): image size mismatch between %d-th segmented (%d x %d) and mask (%d x %d) images", i, segmented_images[i].rows, segmented_images[i].cols, mask_images[i].rows, mask_images[i].cols));
    }

    // clear previously computed visual results if any
    if(visual_results)
        visual_results->clear();

    // True positives (TP), True negatives (TN), and total number N of pixels are all we need
    double TP = 0, TN = 0, N = 0;

    // examine one image at the time
    for(size_t i=0; i<segmented_images.size(); i++)
    {
        // the caller did not ask to calculate visual results
        // accuracy calculation is easier...
        if(visual_results == 0)
        {
            for(int y=0; y<segmented_images[i].rows; y++)
            {
                aia::uint8* segData = segmented_images[i].ptr<aia::uint8>(y);
                aia::uint8* gndData = groundtruth_images[i].ptr<aia::uint8>(y);
                aia::uint8* mskData = mask_images[i].ptr<aia::uint8>(y);

                for(int x=0; x<segmented_images[i].cols; x++)
                {
                    if(mskData[x])
                    {
                        N++;		// found a new sample within the mask

                        if(segData[x] && gndData[x])
                            TP++;	// found a true positive: segmentation result and groundtruth match (both are positive)
                        else if(!segData[x] && !gndData[x])
                            TN++;	// found a true negative: segmentation result and groundtruth match (both are negative)
                    }
                }
            }
        }
        else
        {
            // prepare visual result (3-channel BGR image initialized to black = (0,0,0) )
            cv::Mat visualResult = cv::Mat(segmented_images[i].size(), CV_8UC3, cv::Scalar(0,0,0));

            for(int y=0; y<segmented_images[i].rows; y++)
            {
                aia::uint8* segData = segmented_images[i].ptr<aia::uint8>(y);
                aia::uint8* gndData = groundtruth_images[i].ptr<aia::uint8>(y);
                aia::uint8* mskData = mask_images[i].ptr<aia::uint8>(y);
                aia::uint8* visData = visualResult.ptr<aia::uint8>(y);

                for(int x=0; x<segmented_images[i].cols; x++)
                {
                    if(mskData[x])
                    {
                        N++;		// found a new sample within the mask

                        if(segData[x] && gndData[x])
                        {
                            TP++;	// found a true positive: segmentation result and groundtruth match (both are positive)

                            // mark with blue
                            visData[3*x + 0 ] = 255;
                            visData[3*x + 1 ] = 0;
                            visData[3*x + 2 ] = 0;
                        }
                        else if(!segData[x] && !gndData[x])
                        {
                            TN++;	// found a true negative: segmentation result and groundtruth match (both are negative)

                            // mark with gray
                            visData[3*x + 0 ] = 128;
                            visData[3*x + 1 ] = 128;
                            visData[3*x + 2 ] = 128;
                        }
                        else if(segData[x] && !gndData[x])
                        {
                            // found a false positive

                            // mark with yellow
                            visData[3*x + 0 ] = 0;
                            visData[3*x + 1 ] = 255;
                            visData[3*x + 2 ] = 255;
                        }
                        else
                        {
                            // found a false positive

                            // mark with red
                            visData[3*x + 0 ] = 0;
                            visData[3*x + 1 ] = 0;
                            visData[3*x + 2 ] = 255;
                        }
                    }
                }
            }

            visual_results->push_back(visualResult);
        }
    }

    return (TP + TN) / N;	// according to the definition of Accuracy
}


cv::Mat rotate(cv::Mat src, double angle) {

	cv::Mat dst;
	cv::Point2f pt(src.cols/2., src.rows/2.);
	cv::Mat r = cv::getRotationMatrix2D(pt, angle, 1.0);
	cv::warpAffine(src, dst, r, cv::Size(src.cols, src.rows));
	return dst;
}

namespace eiid {

	using namespace cv;

void equalizeHistWithMask(const Mat1b& src, Mat& dst, Mat1b mask = Mat1b())
{
    int cnz = countNonZero(mask);
    if (mask.empty() || ( cnz == src.rows*src.cols))
    {
        equalizeHist(src, dst);
        return;
    }

    dst = src.clone();

    // Histogram
    vector<int> hist(256,0);
    for (int r = 0; r < src.rows; ++r) {
        for (int c = 0; c < src.cols; ++c) {
            if (mask(r, c)) {
                hist[src(r, c)]++;
            }
        }
    }

    // Cumulative histogram
    float scale = 255.f / float(cnz);
    vector<uchar> lut(256);
    int sum = 0;
    for (int i = 0; i < hist.size(); ++i) {
        sum += hist[i];
        lut[i] = saturate_cast<uchar>(sum * scale);
    }

    // Apply equalization
    for (int r = 0; r < src.rows; ++r) {
        for (int c = 0; c < src.cols; ++c) {
            if (mask(r, c)) {
                dst.at<unsigned char>(r, c) = lut[src(r,c)];
            }
        }
    }
}
}

// Metodo per l'individuazione del contorno relativo al fondo oculare
// Input : vettore di contorni (vettore di vettore di punti)
// Output : posizione del contorno di interesse

int findGoodContourn(std::vector<std::vector<cv::Point>> contours) {

	int goodCont;

	// Avalizzo i contorni: se uno ha una lunghezza maggiore di 1000 unità (termine eurstico, il contorno da considerare deve essete "molto grande") lo considero
	for (int i = 0; i < contours.size(); i++) {

		if (cv::arcLength(contours[i], true) > 1000) {

			goodCont = i;
		}
	}

	/*int goodCont = 0; int maxLength = cv::arcLength(contours[goodCont], true);

	for (int i = 0; i < contours.size(); i++) {

		if (cv::arcLength(contours[i], true) > maxLength) {

			maxLength = cv::arcLength(contours[i], true);
			goodCont = i;
		}
	}*/

	return goodCont;
}

// Metodo per l'indviduazione dei punt estrami di un controno 

// Input : vettore di contorni attuali
// Output : Punti estremi del contorno no chiuso
//        : Variabile booleana che specifica se un contorno è chiuso o meno

void closeContourn(std::vector<cv::Point> contours, std::vector<cv::Point> &interestPoints, bool &close, cv::Mat img) {

	cv::Point prevPos = contours[0];
	cv::Point nextPos = contours[2];
	int cont = 0;

	// Itero parendo dal secondo elemnto del contorno e analizzo la posizione precedente e successiva a quella attuale: se tali punti sono gli stassi
	// ho individuato un estremo, quindi memorizzo tale posizione ed incremento un contatore. Trovato il secondo punto il contatore sarà a 2 e il ciclo trìermina.
	// L'iterazione si basa sul modo con cui il metodo cv::findContours(...) individua i punti del contorno.
	for (int i = 1; i < contours.size() - 2; i++) {

	
		if (nextPos == prevPos) {

			interestPoints.push_back(contours[i]); //std::cout << contours[i] << std::endl;
			cont++;

			if(cont > 2) {

				close = false;
				break;
			}
		}

		nextPos = contours[i + 2];
		prevPos = contours[i];
	}

	// Se cont = 2 ho trovato un contorno aperto

	if (cont == 2) {

		close = true;
	}
}

// Metodo per la costruzione di una maschera
// Input : immagine di partenza (per le dimesioni)
//       : contorno del fondo oculare
// Output : immagine maschera binaria

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

	for (int i = 1; i < contours.size(); i++) {

		if(cv::arcLength(contours[i], false) > cv::arcLength(contours[pos], false)) {

			pos = i;
		}
	}

	return pos;
}

// Metodo per la generazione di un'immagine maschera

// Input : immagine di cui estrarre la maschera
// Output : immagine maschera

cv::Mat maskGenerating(const cv::Mat& image) {

	cv::Mat img = image.clone(); //aia::imshow("Image", img); 

	// MORPHOLOGICAL SMOOTHING 
	// Viene applicato un morphological smoothing per ridurre l'intensità dei vasi sanguignei pur mantenendo il contorno della maschera.

	int k = 11;

	cv::morphologyEx(img, img, CV_MOP_OPEN, cv::getStructuringElement(CV_SHAPE_RECT, cv::Size(k,k)));
	cv::morphologyEx(img, img, CV_MOP_CLOSE, cv::getStructuringElement(CV_SHAPE_RECT, cv::Size(k,k)));
	//aia::imshow("Morphological Smoothing", img);

	cv::Mat imageEdges; int thresholdCanny = CANNY_THRESHOLD;
	cv::GaussianBlur(img, img, cv::Size(3,3), 3,3);
	cv::Canny(img, imageEdges, thresholdCanny, 3 * thresholdCanny);
	//aia::imshow(" ", imageEdges);

	std::vector<std::vector<cv::Point>> contoursEdges;
	cv::findContours(imageEdges, contoursEdges, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_NONE);
	int rightContourn = maxContourn(contoursEdges);
	//cv::drawContours(img, contoursEdges, rightContourn, cv::Scalar(255), 2); //aia::imshow(" ", img);

	std::vector<cv::Point> interestPoints; bool toClose = false;
	closeContourn(contoursEdges[rightContourn], interestPoints, toClose, imageEdges);

	if (toClose == true) {

		cv::line(imageEdges, interestPoints[0], interestPoints[1], cv::Scalar(255), 1);
		//aia::imshow(" ", img);

		// IF THE CONTORN WAS OPEN I MUST RE-COMPUTE THE CONTOURS

		contoursEdges.clear();

		cv::findContours(imageEdges, contoursEdges, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_NONE);
		rightContourn = findGoodContourn(contoursEdges); 
	}

	cv::Mat bin = createMask(img, contoursEdges[rightContourn]);
	
	return bin;

	//cv::fillConvexPoly(bin, contoursEdges[rightContourn], cv::Scalar(255)); aia::imshow("1", bin);

	/*
	// GRADIENT IMAGE
	// Mi ricavo l'immagine gradiente dalla quale verrà effettuata l'estrazione dei contorni

	cv::Mat gradientImage = img.clone();
	cv::morphologyEx(img, gradientImage, CV_MOP_GRADIENT, cv::getStructuringElement(CV_SHAPE_RECT, cv::Size(3, 3)));

	// CANNY
	// Applico il metodo di Canny per estrarre il contorno dominate della maschera. Non è stato applicato prima un gaussian blure, come previsto
	// dal metodo, data la struttra "semplice" dell'immagine. (Sono contorni chiari su sfondo scuro)

	cv::Mat imageEdges; int thresholdCanny = CANNY_THRESHOLD;
	//cv::GaussianBlur(gradientImage, gradientImage, cv::Size(3,3), 3,3);
	cv::Canny(gradientImage, imageEdges, thresholdCanny, 3 * thresholdCanny);
	//aia::imshow(" ", imageEdges);

	// GRADIENT IMAGE BINARIZED
	// Applico un thresholding secondo il metodo Otsu sull'immagine gradiente di partenza.

	cv::Mat gradientImageBin;
	cv::threshold(gradientImage, gradientImageBin, 0, 255, CV_THRESH_BINARY | CV_THRESH_OTSU);
	//aia::imshow("Gradient image", gradientImageBin);	

	// FIND CONTOURS
	// Estraggo i contorni sia dall'immagine ricavata tramite l'algoritmo di Canny si da quella ricavata tramite la binarizzazione dell'immagine gradiente.

	std::vector<std::vector<cv::Point>> contoursBin;
	std::vector<std::vector<cv::Point>> contoursEdges;
	cv::findContours(gradientImageBin, contoursBin, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_NONE);
	cv::findContours(imageEdges, contoursEdges, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_NONE);

	int goodContBin = findGoodContourn(contoursBin);
	int goodContEdge = findGoodContourn(contoursEdges); 

	// CLOSE OPEN CONTOURS
	// Può succedere che, a causa del modo in cui è stata acquisita l'imagine, non si ottenga un contono del fondo oculare chiuso. E' necessario
	// un metodo che chida il contorno individuato
	
	std::vector<cv::Point> interestPoints; bool toClose = false;
	closeContourn(contoursEdges[goodContEdge], interestPoints, toClose, imageEdges);

	if (toClose == true) {

		cv::line(imageEdges, interestPoints[0], interestPoints[1], cv::Scalar(255), 1);
		//aia::imshow(" ", img);

		// IF THE CONTORN WAS OPEN I MUST RE-COMPUTE THE CONTOURS

		contoursEdges.clear();

		cv::findContours(imageEdges, contoursEdges, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_NONE);
		goodContEdge = findGoodContourn(contoursEdges); 
	}

	// POINT POLYGON TEST
	// Costruisco l'immagine maschera a partire dai contorni estratti in precedenza.

	cv::Mat resultBin = cv::Mat(img.rows, img.cols, CV_8U);
	resultBin = createMask(img, contoursBin[goodContBin]);

	//aia::imshow("solo bin", resultBin, false);

	cv::Mat resultEdge = cv::Mat(img.rows, img.cols, CV_8U);
	resultEdge = createMask(img, contoursEdges[goodContEdge]);

	//aia::imshow("bn + edge", resultEdge, false);

	// L'immagine risultane è la somma delle due maschere estratte. Questo perchè, data la soglia di threshold, il risuato otteuto con Canny può produrre
	// contorni che si chidono su loro stessi per cui il contono non verrebbe considerato chiuso "chiuso" cosa che con capita per le immagini binarie. Di contro,
	// in altri casi, sono i contorni delle immagini binarie a non essere ben definiti per cui è necesario usare i risultati di Canny. Aumentare o ridurre la
	// soglia di threshold pùo portare a risulatai non corretti.
	
	img =  resultBin + resultEdge;
	//aia::imshow("Mask", img);
	*/
	//return img;
}

// Metodo per la generazione di immagni maschere per un dataset

// Input : stringa che defnisce la posizione del dataset
// Output : maschere del dataset

void generateMasksDataset(std::string str) {

	std::vector <cv::Mat> images = getImagesInFolder(str + "images", ".ppm");
	std::vector<cv::Mat> masks; 

	for (int i = 0; i < images.size(); i++) {
		
		std::vector<cv::Mat> channels;

		// Estraggo il canale verde dell'immagine
		cv::split(images[i], channels);
		cv::Mat red_channel = channels[2];
		red_channel.convertTo(red_channel, CV_8U);

		// Memorizzo la maschera ricavata
		masks.push_back(maskGenerating(red_channel));

		// Salvo l'immagine
		std::string pathName = str + "masks2/" + std::to_string(i+1) + ".tif";
		cv::imwrite(pathName, masks[i]);

		printf("Image saved %d\n", i + 1);
	}
}

int main() 
{
	try {

		// MASK GENERATING

		generateMasksDataset(std::string(PATH));

		// 1 MASK GENERATING TEST
		
		/*cv::Mat image = cv::imread(std::string(PATH) + std::string("images/im0044.ppm"), CV_LOAD_IMAGE_UNCHANGED);
		std::vector<cv::Mat> channels;

		cv::split(image, channels);
		cv::Mat red_channel = channels[2];
		red_channel.convertTo(red_channel, CV_8U);

		cv::Mat mask = maskGenerating(red_channel);*/

		return 1;

	}
	catch (aia::error &ex)
	{
		std::cout << "EXCEPTION thrown by " << ex.getSource() << "source :\n\t|=> " << ex.what() << std::endl;
	}
	catch (ucas::Error &ex)
	{
		std::cout << "EXCEPTION thrown by unknown source :\n\t|=> " << ex.what() << std::endl;
	}
}

