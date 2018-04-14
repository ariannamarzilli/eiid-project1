// include aia and ucas utility functions
#include "aiaConfig.h"
#include "ucasConfig.h"

// include my project functions
#include "functions.h"

cv::Mat rotate(cv::Mat src, double angle) {

	cv::Mat dst;
	cv::Point2f pt(src.cols/2., src.rows/2.);
	cv::Mat r = cv::getRotationMatrix2D(pt, angle, 1.0);
	cv::warpAffine(src, dst, r, cv::Size(src.cols, src.rows));
	return dst;
}

int main() 
{
	try {
		cv::Mat image = cv::imread("C:/Users/utente/Downloads/AIA-Retinal-Vessel-Segmentation-20180409T200855Z-001/AIA-Retinal-Vessel-Segmentation/datasets/DRIVE/images/01.tif", CV_LOAD_IMAGE_UNCHANGED);
		cv::Mat mask = cv::imread("C:/Users/utente/Downloads/AIA-Retinal-Vessel-Segmentation-20180409T200855Z-001/AIA-Retinal-Vessel-Segmentation/datasets/DRIVE/masks/01_mask.tif", CV_LOAD_IMAGE_GRAYSCALE);
		cv::Mat ground_truth = cv::imread("C:/Users/utente/Downloads/AIA-Retinal-Vessel-Segmentation-20180409T200855Z-001/AIA-Retinal-Vessel-Segmentation/datasets/DRIVE/groundtruths/01_manual1.tif", CV_LOAD_IMAGE_GRAYSCALE);

		std::vector<cv::Mat> channels;

		cv::split(image, channels);
		cv::Mat green_channel = channels[1];

		green_channel.convertTo(green_channel, CV_8U);
		mask.convertTo(mask, CV_8U);
		ground_truth.convertTo(ground_truth, CV_8U);

		// NOISE REDUCTION

		cv::fastNlMeansDenoising(green_channel, green_channel, 3);

		// TOP-HAT TRANSFORM

		cv::Mat green_channel_reverse = 255 - green_channel;

		aia::imshow("green_channel_reverse", green_channel_reverse);

		cv::Mat dest, THTransform;

		cv::morphologyEx(green_channel_reverse, dest, cv::MORPH_TOPHAT, cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(20, 20)));

		THTransform = green_channel_reverse - dest;

		cv::Mat foundRegion = 10 * (green_channel_reverse - THTransform);

		aia::imshow("found region", foundRegion);


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

