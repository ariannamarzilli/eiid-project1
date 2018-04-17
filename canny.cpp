// include aia and ucas utility functions
#include "aiaConfig.h"
#include "ucasConfig.h"
#include <opencv2\photo.hpp>


// include my project functions
#include "functions.h"

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

int main() 
{
	try {
		cv::Mat image = cv::imread("C:/Users/utente/Downloads/AIA-Retinal-Vessel-Segmentation-20180409T200855Z-001/AIA-Retinal-Vessel-Segmentation/datasets/DRIVE/images/01.tif", CV_LOAD_IMAGE_UNCHANGED);
		cv::Mat mask = cv::imread("C:/Users/utente/Downloads/AIA-Retinal-Vessel-Segmentation-20180409T200855Z-001/AIA-Retinal-Vessel-Segmentation/datasets/DRIVE/masks/01_mask.tif", CV_LOAD_IMAGE_GRAYSCALE);
		cv::Mat ground_truth = cv::imread("C:/Users/utente/Downloads/AIA-Retinal-Vessel-Segmentation-20180409T200855Z-001/AIA-Retinal-Vessel-Segmentation/datasets/DRIVE/groundtruths/01_manual1.tif", CV_LOAD_IMAGE_GRAYSCALE);

		//cv::Mat image = cv::imread("C:/Users/utente/Desktop/01.tif (green).tif", CV_LOAD_IMAGE_UNCHANGED);// DA SOSTITUIRE
		std::vector<cv::Mat> channels;

		cv::split(image, channels);
		//cv::Mat green_channel = channels[1];
		cv::Mat green_channel = channels[0]; // d SOSTITUIRE
		green_channel.convertTo(green_channel, CV_8U);
		mask.convertTo(mask, CV_8U);
		ground_truth.convertTo(ground_truth, CV_8U);

		aia::imshow("Image", green_channel);

		// NOISE REDUCTION

		cv::fastNlMeansDenoising(green_channel, green_channel, 0.40*15.0, 3, 21);

		aia::imshow("Denoised image", green_channel);

		// TOP-HAT TRANSFORM

		cv::Mat green_channel_reverse = 255 - green_channel;

		//aia::imshow("green_channel_reverse", green_channel_reverse);

		cv::Mat dest, THTransform;

		cv::morphologyEx(green_channel_reverse, dest, cv::MORPH_TOPHAT, cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(15, 15)));

		THTransform = green_channel_reverse - dest; 

		cv::Mat foundRegion = 10  * (green_channel_reverse - THTransform);

		aia::imshow("found region", foundRegion);

		// usare, eventualmente, THTransform come marker e l'immagine di partenza come maschera

		// BINARIZZAZIONE

		// Rendo l'esterno del fondo retinico completamente nero

		for (int y = 0; y < foundRegion.rows; y++) {

			unsigned char* yRowImage = foundRegion.ptr<unsigned char>(y);
			unsigned char* yRowMask = mask.ptr<unsigned char>(y);

			for (int x = 0; x < foundRegion.cols; x++) {

				if (yRowMask[x] != 255) {

					yRowImage[x] = 0;
				}
			}
		}

		aia::imshow("found Region (tutto sfondo nero)", foundRegion);

		// CANNY BINARIZATION
		
		int sigmaCannyX10 = 10;			// similar to 'sigmaGradX10'
		int thresholdCanny = 20;
		cv::Mat imgEdges;              // we will store here the binary image after edge detection to be displayed

		// calculate gaussian kernel size so that 99% of data are under the gaussian 
		int n = 6* (sigmaCannyX10/10.0);
		if(n % 2 == 0)
			n++;
		// if 'sigmaCannyX10' is valid, we apply gaussian smoothing
		if(sigmaCannyX10 > 0)
			cv::GaussianBlur(foundRegion, imgEdges, cv::Size(n,n), (sigmaCannyX10/10.0), (sigmaCannyX10/10.0));
		// otherwise we simply clone the image as it is
		else
			imgEdges = foundRegion.clone();

		// NOTE: OpenCV Canny function does not include gaussian smoothing: that's why we must did it before calling cv::Canny
		cv::Canny(imgEdges, imgEdges, thresholdCanny, 3*thresholdCanny);
		//                                            /\
		//                                            || suggested by Canny: 2 * low threshold <= high threshold <= 3 * low threshold

		aia::imshow("Edge detection (Canny)", imgEdges);

		// SMOOTHING EDGE PRESERVING

		/*cv::fastNlMeansDenoising(foundRegion, foundRegion, 0.40*15.0, 3, 21);

		aia::imshow("Found Region (smoothed)", foundRegion);

		 // Calcolo l'istogramma della regione interna al fondo retinico

		std::vector<int> histogram_in_FOV(256);
		float mask_element = 0;

		for (int y = 0; y < foundRegion.rows; y++) {

			unsigned char* y_row_image = foundRegion.ptr<unsigned char>(y);
			unsigned char* y_row_image_mask = mask.ptr<unsigned char>(y);

			for (int x = 0; x < foundRegion.cols; x++) {

				if (y_row_image_mask[x] != 0) {

					histogram_in_FOV[y_row_image[x]]++;
					mask_element = mask_element + 1;
				}
			}
		}

		// Binarizzo tramite otsu

		cv::Mat binarized;
		int T = ucas::getOtsuAutoThreshold(histogram_in_FOV);

		cv::threshold(foundRegion, binarized, T, 255, CV_THRESH_BINARY);*/

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

