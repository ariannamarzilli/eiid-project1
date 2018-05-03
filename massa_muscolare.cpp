#include <opencv2/core/mat.hpp>
#include <opencv/cv.hpp>
#include <iostream>
#include "3rdparty/ucascommon/ucasExceptions.h"
#include "3rdparty/ucascommon/ucasStringUtils.h"
#include "3rdparty/aiacommon/aiaConfig.h"
#include "3rdparty/ucascommon/ucasFileUtils.h"

#define DATASET_PATH "/Users/Mariangela/Desktop/AIA-Pectoral-Muscle-Segmentation/dataset/"
#define DATASET_VIS_RESULTS_PATH "/Users/Mariangela/Desktop/AIA-Pectoral-Muscle-Segmentation/dataset/results/"

std::vector<cv::Mat> createTiltedStructuringElements(int width, int height, int n) throw (ucas::Error) {
    // check preconditions
    if( width%2 == 0 )
        throw ucas::Error(ucas::strprintf("Structuring element width (%d) is not odd", width));
    if( height%2 == 0 )
        throw ucas::Error(ucas::strprintf("Structuring element height (%d) is not odd", height));

    // draw base SE along x-axis
    cv::Mat base(width, width, CV_8U, cv::Scalar(0));
    // workaround: cv::line does not work properly when thickness > 1. So we draw line by line.
    for(int k=width/2-height/2; k<=width/2+height/2; k++)
        cv::line(base, cv::Point(0,k), cv::Point(width, k), cv::Scalar(255));

    // compute rotated SEs
    std::vector <cv::Mat> SEs;
    SEs.push_back(base);
    double angle_step = 180.0/n;
    for(int k=1; k<n; k++)
    {
        cv::Mat SE;
        cv::warpAffine(base, SE, cv::getRotationMatrix2D(cv::Point2f(base.cols/2.0f, base.rows/2.0f), k*angle_step, 1.0), cv::Size(width, width), CV_INTER_NN);
        SEs.push_back(SE);
    }

    return SEs;
}

// retrieves and loads all images within the given folder and having the given file extension
std::vector < cv::Mat > getImagesInFolder(std::string folder, std::string ext = ".tif", bool force_gray = false) throw (aia::error)
{
    // check folders exist
    if(!ucas::isDirectory(folder))
        throw aia::error(aia::strprintf("in getImagesInFolder(): cannot open folder at \"%s\"", folder.c_str()));

    // get all files within folder
    std::vector < cv::String > files;
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

        // True positives = blue, True negatives = gray, False positives = yellow, False negatives = red
) throw (aia::error)
{
    /*
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

    }*/


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



int main() {



    try {

        //Load images
        std::string datasetPath = DATASET_PATH;
        std::vector <cv::Mat> images; //getImagesInFolder(datasetPath + "images", "tif");
        std::vector <cv::Mat> truths; //= //getImagesInFolder(datasetPath + "groundtruths", "tif");
        std::vector <cv::Mat> masks; //getImagesInFolder(datasetPath + "masks", "png");
        std::vector <cv::Mat> results; bool toFlip = false;
        cv::Mat dummy = cv::imread(std::string(DATASET_PATH) + std::string("images/20586986_6c613a14b80a8591_MG_L_ML_ANON.tif"), CV_LOAD_IMAGE_UNCHANGED);
        images.push_back(dummy);
        dummy = cv::imread(std::string(DATASET_PATH) + std::string("masks/20586986_6c613a14b80a8591_MG_L_ML_ANON.mask.png"));
        masks.push_back(dummy);

        printf("Load done\n");

        for (int i = 0; i < images.size(); i++) {

            // Convert the image in 8 bit

            /*cv::normalize(img, img, 0, 255, cv::NORM_MINMAX);
            img.convertTo(img, CV_8U);
            cv::normalize(mask, mask, 0, 255, cv::NORM_MINMAX);
            mask.convertTo(mask, CV_8U);

            int xPos; float angle = 70;

            unsigned char *yRow = mask.ptr<unsigned char>(10);

            for (int x = 0; x < mask.cols -1; x++) {

                if (yRow[x] == 255 && yRow[x + 1] == 0) {

                    xPos = x;
                    break;
                }
            }

            xPos = 3*xPos/4;
            int yPos = xPos * tan(angle * ucas::PI / 180);
            cv::Point p1(xPos, 0), p2(0, yPos);

            cv::Mat newImage = cv::Mat(img.rows, img.cols, CV_8U, cv::Scalar(0));

            for (int y = 0; y < yPos; y++) {

                unsigned char *yRow = img.ptr<unsigned char>(y);
                unsigned char *yRowNew = newImage.ptr<unsigned char>(y);

                for (int x = 0; x < xPos; x++) {

                    yRowNew[x] = yRow[x];
                }
            }

            roi = img(cv::Rect(0, 0, xPos, yPos));

            // Equalize
            gamma = 255;

            cv::Mat equalized;

            cv::equalizeHist(roi, equalized);

            float c = std::pow(L - 1, 1 - gamma/100.0f);

            for (int y = 0; y < equalized.rows; y++) {
                unsigned char *yRow = equalized.ptr<unsigned char>(y);

                for (int x = 0; x < equalized.cols; x++) {
                    yRow[x] = c * std::pow(yRow[x], gamma/100.0f);
                }
            }
        roi = equalized.clone();

        cv::fastNlMeansDenoising(roi, roi, 20);

        cv::threshold(roi, roi, 0, 255, CV_THRESH_BINARY | CV_THRESH_OTSU);
            aia::imshow("new", roi);*/

            cv::Mat img = images[i].clone(); printf("%d", img.channels());getchar();
            if (masks[i].at<unsigned char>(30, 30) == 0) {

                cv::flip(img, img , 1);
                cv::flip(masks[i], masks[i], 1);
                toFlip = true;
            }

            cv::normalize(img, img, 0, 255, cv::NORM_MINMAX);
            img.convertTo(img, CV_8U);

            //Equalize
            cv::Mat equalized = img.clone();
            int k = 10;

            cv::equalizeHist(img, equalized);
            //aia::imshow("Equalized", equalized, false, 0.23);

            // Non linear trasformation
            int L = 256;
            float gamma = 180.0f;
            float c = std::pow(L - 1, 1 - gamma / 100.0f);

            for (int y = 0; y < equalized.rows; y++) {
                unsigned char *yRow = equalized.ptr < unsigned char > (y);

                for (int x = 0; x < equalized.cols; x++) {
                    yRow[x] = c * std::pow(yRow[x], gamma / 100.0f);
                }
            }

            //aia::imshow("Gamma correction", equalized, false, 0.23);

            //Smoothing
            cv::Mat smooth = equalized.clone();

            cv::morphologyEx(equalized, smooth, CV_MOP_OPEN, cv::getStructuringElement(CV_SHAPE_RECT, cv::Size(k, k)));
            cv::morphologyEx(equalized, smooth, CV_MOP_CLOSE, cv::getStructuringElement(CV_SHAPE_RECT, cv::Size(k, k)));
            //aia::imshow("Smoothed", smooth, false, 0.23);

            //Mean Shift
            cv::Mat meanShift = smooth.clone();
            int hs = 10, hr = 30;
            cv::Mat multiChannelSmooth;

            cv::cvtColor(smooth, multiChannelSmooth, CV_GRAY2BGR);
            cv::pyrMeanShiftFiltering(multiChannelSmooth, meanShift, hs, hr, 0);
            cv::cvtColor(meanShift, meanShift, CV_BGR2GRAY);
            //aia::imshow("Mean Shift", meanShift, false, 0.23);

            //Binarize
            cv::Mat bin = meanShift.clone();

            cv::threshold(meanShift, bin, 0, 255, CV_THRESH_BINARY | CV_THRESH_OTSU);
            //aia::imshow("Binarized", bin, false, 0.23);

            cv::morphologyEx(bin, bin, CV_MOP_OPEN, cv::getStructuringElement(CV_SHAPE_RECT, cv::Size(25, 25)));
            //aia::imshow("OPENED", bin, false, 0.23);

            std::vector <std::vector<cv::Point> > contours;
            cv::Mat binCopy = bin.clone();

            cv::findContours(binCopy, contours, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_NONE);
            //cv::cvtColor(bin, bin, CV_GRAY2BGR);

            /*for (int i = 0; i < contours.size(); i++) {

                cv::drawContours(bin, contours, i, cv::Scalar(0, 0, 255), 2);
                aia::imshow("Contours", bin, true);
            }

            cv::cvtColor(bin, bin, CV_BGR2GRAY);*/
            int goodContourn = 0;

            for (int i = 0; i < contours.size(); i++) {

                if (cv::pointPolygonTest(contours[i], cv::Point(5, 5), false) >= 0) {

                    goodContourn = i;
                }
            }

            cv::fillConvexPoly(bin, contours[goodContourn], cv::Scalar(255));

            for (int y = 0; y < bin.rows / 2; y++) {

                unsigned char *yRow = bin.ptr < unsigned char > (y);
                unsigned char *yRowMask = masks[i].ptr < unsigned char > (y);

                for (int x = 0; x < (3 * bin.cols) / 4; x++) {

                    if (yRowMask[x] == 255) {

                        if (cv::pointPolygonTest(contours[goodContourn], cv::Point(x, y), false) < 0) {

                            yRow[x] = 0;
                        }
                    }
                }
            }

            aia::imshow("Result", bin, true, 0.23);
            printf("%d\n", i + 1);

            if (toFlip) {

                cv::flip(bin, bin, 1);
                cv::flip(masks[i], masks[i], 1);
                toFlip = false;
            }

            results.push_back(bin);
        }

        std::vector <cv::Mat> visual_results;
        double ACC = accuracy(results, truths, masks, &visual_results);
        printf("Accuracy = %.2f%%\n", ACC*100);

        for (int i = 0; i < visual_results.size(); i++) {

            std::string pathNameVisualResults = DATASET_VIS_RESULTS_PATH + std::to_string(i + 1) + ".tif";
            cv::imwrite(pathNameVisualResults, visual_results[i]);
        }
        return 0;
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