//
// Created by Mariangela Evangelista on 20/05/18.
//

#ifndef PROVA_UTILITY_H
#define PROVA_UTILITY_H


#include <opencv2/core/mat.hpp>
#include "3rdparty/ucascommon/ucasExceptions.h"
#include "3rdparty/aiacommon/aiaConfig.h"


cv::Mat rotate90(cv::Mat img, int step);


// create 'n' rectangular Structuring Elements (SEs) at different orientations spanning the whole 360°
// return vector of 'width' x 'width' uint8 binary images with non-black pixels being the SE
// parameter width: SE width (must be odd)
// parameter height: SE height (must be odd)
// parameter n: number of SEs
std::vector<cv::Mat> createTiltedStructuringElements(int width, int height, int n) throw (ucas::Error);


// retrieves and loads all images within the given folder and having the given file extension
std::vector < cv::Mat > getImagesInFolder(std::string folder, std::string ext = ".tif", bool force_gray = false) throw (aia::error);

// Accuracy (ACC) is defined as:
// ACC = (True positives + True negatives)/(number of samples)
// i.e., as the ratio between the number of correctly classified samples (in our case, pixels)
// and the total number of samples (pixels)
double accuracy(
        std::vector <cv::Mat> & segmented_images,		// (INPUT)  segmentation results we want to evaluate (1 or more images, treated as binary)
        std::vector <cv::Mat> & groundtruth_images,     // (INPUT)  reference/manual/groundtruth segmentation images
        std::vector <cv::Mat> & mask_images,			// (INPUT)  mask images to restrict the performance evaluation within a certain region
        std::vector <cv::Mat> * visual_results = 0		// (OUTPUT) (optional) false color images displaying the comparison between automated segmentation results and groundtruth

        // True positives = blue, True negatives = gray, False positives = yellow, False negatives = red
) throw (aia::error);

#endif //PROVA_UTILITY_H
