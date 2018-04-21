// include aia and ucas utility functions
#include "aiaConfig.h"
#include "ucasConfig.h"
#include <opencv2\photo.hpp>


// include my project functions
#include "functions.h"

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

int findGoodContourn(std::vector<std::vector<cv::Point>> contours) {

	int goodCont;

	for (int i = 0; i < contours.size(); i++) {

		if (cv::arcLength(contours[i], true) > 1000) {

			goodCont = i;
		}
	}

	return goodCont;
}

void closeContourn(std::vector<cv::Point> contours, cv::Point interestPoints[], bool &close) {

	cv::Point prevPos = contours[0];
	cv::Point nextPos = contours[2];
	int cont = 0;

	for (int i = 1; i < contours.size() - 2; i++) {

		if (nextPos == prevPos) {

			interestPoints[cont] = contours[i];

			//std::cout << std::endl << "Interest point: " << interestPoints[cont] << std::endl << std::endl;
			cont++;
		}

		//cv::circle(img, cv::Point(contoursEdges[goodContEdge][i]), 3, cv::Scalar(0), 2);
		//std::cout << "Prev:" << prevPos  << " ---- " << "Now" << contoursEdges[goodContEdge][i]  << " ---- "<< "Next" << nextPos << std::endl;
		//aia::imshow(" ", img);

		nextPos = contours[i + 2];
		prevPos = contours[i];
	}

	if (cont == 2) {

		close = true;
	}
}

cv::Mat createMask(const cv::Mat& img, std::vector<cv::Point> contours) {

	cv::Mat resultBin = img.clone();

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

cv::Mat maskGenerating(const cv::Mat& image) {

	cv::Mat img = image.clone(); 

	// MORPHOLOGICAL SMOOTHING

	int k = 11;

	cv::morphologyEx(img, img, CV_MOP_OPEN, cv::getStructuringElement(CV_SHAPE_RECT, cv::Size(k,k)));
	cv::morphologyEx(img, img, CV_MOP_CLOSE, cv::getStructuringElement(CV_SHAPE_RECT, cv::Size(k,k)));
	//aia::imshow("Morphological Smoothing", img);

	// GRADIENT IMAGE

	cv::Mat gradientImage = img.clone();
	cv::morphologyEx(img, gradientImage, CV_MOP_GRADIENT, cv::getStructuringElement(CV_SHAPE_RECT, cv::Size(3, 3)));

	// CANNY

	cv::Mat imageEdges; int thresholdCanny = 55;

	cv::Canny(gradientImage, imageEdges, thresholdCanny, 3 * thresholdCanny);
	//aia::imshow(" ", imageEdges);

	// GRADIENT IMAGE BINARIZED

	cv::Mat gradientImageBin;
	cv::threshold(gradientImage, gradientImageBin, 0, 255, CV_THRESH_BINARY | CV_THRESH_OTSU);
	//aia::imshow("Gradient image", gradientImageBin);	

	// FIND CONTOURS

	std::vector<std::vector<cv::Point>> contoursBin;
	std::vector<std::vector<cv::Point>> contoursEdges;
	cv::findContours(gradientImageBin, contoursBin, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_NONE);
	cv::findContours(imageEdges, contoursEdges, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_NONE);

	std::cout << contoursBin.size() << std::endl;

	int goodContBin = findGoodContourn(contoursBin);
	int goodContEdge = findGoodContourn(contoursEdges); 

	/*for (int i = 0; i < contoursBin.size(); i++) {

		if (cv::arcLength(contoursBin[i], true) > 1000) {

			goodContBin = i;
		}
	}

	for (int i = 0; i < contoursEdges.size(); i++) {

		if (cv::arcLength(contoursEdges[i], true) > 1000) {

			goodContEdge = i;
		}
	}*/

	cv::Point interestPoints[2]; bool close = false;
	closeContourn(contoursEdges[goodContEdge], interestPoints, close);

	/*cv::Point prevPos = contoursEdges[goodContEdge][0];
	cv::Point nextPos = contoursEdges[goodContEdge][2];

	cv::Point interestPoints[2]; int cont = 0;

	for (int i = 1; i < contoursEdges[goodContEdge].size() - 2; i++) {

		if (nextPos == prevPos) {

			interestPoints[cont] = contoursEdges[goodContEdge][i];

			//std::cout << std::endl << "Interest point: " << interestPoints[cont] << std::endl << std::endl;
			cont++;
		}

		//cv::circle(img, cv::Point(contoursEdges[goodContEdge][i]), 3, cv::Scalar(0), 2);
		//std::cout << "Prev:" << prevPos  << " ---- " << "Now" << contoursEdges[goodContEdge][i]  << " ---- "<< "Next" << nextPos << std::endl;
		//aia::imshow(" ", img);

		nextPos = contoursEdges[goodContEdge][i + 2];
		prevPos = contoursEdges[goodContEdge][i];

	}*/

	if (close == true) {

		cv::line(imageEdges, interestPoints[0], interestPoints[1], cv::Scalar(255), 1);
		//aia::imshow(" ", img);

		// RE-COMPUTE THE CONTOURS

		contoursEdges.clear();

		cv::findContours(imageEdges, contoursEdges, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_NONE);
		goodContEdge = findGoodContourn(contoursEdges); 
	}

	/*for (int i = 0; i < contoursEdges.size(); i++) {

		if (cv::arcLength(contoursEdges[i], true) > 1000) {

			goodContEdge = i;
		}
	}*/	

	// POINT POLYGON TEST
	cv::Mat resultBin = cv::Mat(img.rows, img.cols, CV_8U);
	resultBin = createMask(img, contoursBin[goodContBin]);

	/*for (int y = 0; y < img.rows; y++) {

		unsigned char* yRow = resultBin.ptr<unsigned char>(y);

		for (int x = 0; x < img.cols; x++) {

			if (cv::pointPolygonTest(contoursBin[goodContBin], cv::Point2f(x, y), false) >= 0) {

				yRow[x] = 255;
			} else {

				yRow[x] = 0;
				
			}
		}
	}*/

	aia::imshow("solo bin", resultBin, false);

	cv::Mat resultEdge = cv::Mat(img.rows, img.cols, CV_8U);
	resultEdge = createMask(img, contoursEdges[goodContEdge]);

	/*for (int y = 0; y < img.rows; y++) {

		unsigned char* yRow = resultEdge.ptr<unsigned char>(y);

		for (int x = 0; x < img.cols; x++) {

			if (cv::pointPolygonTest(contoursEdges[goodContEdge], cv::Point2f(x, y), false) >= 0) {

				yRow[x] = 255;
			} else {

				yRow[x] = 0;
				
			}
		}
	}*/

	aia::imshow("bn + edge", resultEdge, false);
	img = resultBin + resultEdge;
	aia::imshow("Mask", img);

	return img;
}

void generateMasksDataset(std::string str) {

	std::vector <cv::Mat> images = getImagesInFolder(str + "images", ".ppm");
	std::vector<cv::Mat> masks; 

	//std::cout << str + "masks/" + std::to_string(1) + ".tif" << std::endl; getchar();

	for (int i = 0; i < images.size(); i++) {

		std::vector<cv::Mat> channels;

		cv::split(images[i], channels);
		cv::Mat red_channel = channels[2];
		red_channel.convertTo(red_channel, CV_8U);

		masks.push_back(maskGenerating(red_channel));

		std::string pathName = str + "masks/" + std::to_string(i+1) + ".tif";
		cv::imwrite(pathName, masks[i]);

		printf("Image saved %d\n", i + 1);
	}
}

int main() 
{
	try {

		// MASK GENERATING

		//generateMasksDataset(std::string("C:/Users/Simone/Desktop/AIA-Retinal-Vessel-Segmentation/datasets/STARE/"));
		//getchar();
		// 1 MASK GENERATING TEST
		
		cv::Mat image = cv::imread("C:/Users/Simone/Desktop/AIA-Retinal-Vessel-Segmentation/datasets/STARE/images/im0162.ppm", CV_LOAD_IMAGE_UNCHANGED);
		std::vector<cv::Mat> channels;

		cv::split(image, channels);
		cv::Mat red_channel = channels[2];
		red_channel.convertTo(red_channel, CV_8U);

		cv::Mat mask = maskGenerating(red_channel); getchar();

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