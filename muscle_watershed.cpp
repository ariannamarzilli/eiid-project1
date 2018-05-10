#include <opencv2/core/hal/interface.h>
#include <opencv2/core/types.hpp>
#include <opencv/cv.hpp>
#include <opencv2/core/mat.hpp>
#include <iostream>
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

{/*
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

*/

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

std::vector<std::vector <cv::Point> > myWatershed(cv::Mat img, cv::Mat origin) {

    float scaling_factor = 1; int radius = 5; int offset = 50;
    //aia::imshow("Mammogram", img, true, scaling_factor);

    //generating internal marker
    cv::Mat internal_markers = cv::Mat(img.rows, img.cols, CV_8U, cv::Scalar(0));
    cv::circle(internal_markers, cv::Point(radius , radius), radius, cv::Scalar(255), -1);
   // aia::imshow("Internal marker", internal_markers, true, scaling_factor);

    //generating external marker
    cv::Mat external_marker = cv::Mat(img.rows, img.cols, CV_8U, cv::Scalar(0));
    /*cv::circle(external_marker, cv::Point(img.cols -20, img.rows -20), radius, cv::Scalar(255), -1);
    cv::circle(external_marker, cv::Point(img.cols -20, img.rows / 2), radius, cv::Scalar(255), -1);
    cv::circle(external_marker, cv::Point(img.cols -20, 20), radius, cv::Scalar(255), -1);*/
    //cv::line(external_marker, cv::Point(img.cols -radius, img.rows -radius), cv::Point(img.cols -radius, radius), cv::Scalar(255), radius);
    //cv::line(external_marker, cv::Point(img.cols -radius, img.rows -radius), cv::Point(radius, img.rows -radius), cv::Scalar(255), radius);
    cv::line(external_marker, cv::Point(img.cols -radius + offset, radius), cv::Point(radius + offset, img.rows -radius), cv::Scalar(255), radius);
    //aia::imshow("External marker", external_marker, true, scaling_factor);

    // build the marker image to be inputted to the whatershed
    cv::Mat markers(img.rows, img.cols, CV_32S, cv::Scalar(0));
    std::vector < std::vector <cv::Point> > contours;
    cv::findContours(internal_markers, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_NONE);
    for(int i=0; i<contours.size(); i++) {
        cv::drawContours(markers, contours, i, cv::Scalar(i+1), CV_FILLED);
    }
    markers.setTo(cv::Scalar(contours.size()+1), external_marker);

    // Visualize the markers
    cv::Mat markers_image = markers.clone();
    cv::normalize(markers_image, markers_image, 0, 255, cv::NORM_MINMAX);
    markers_image.convertTo(markers_image, CV_8U);
   // aia::imshow("Markers image", markers_image, true, scaling_factor);

    cv::cvtColor(img, img, CV_GRAY2BGR);
    cv::watershed(img, markers);
    cv::cvtColor(img, img, CV_BGR2GRAY);
    // marker = -1 where there are dams
    // marker + 1 = 0 where there are dams, != 0 in the rest of the image
    markers += 1;
    markers.convertTo(markers, CV_8U);
    cv::threshold(markers, markers, 0, 255, CV_THRESH_BINARY_INV);
    //aia::imshow("Dams = region contours", markers, true, scaling_factor);
    std::vector<std::vector<cv::Point> > contour;
    cv::findContours(markers, contour, CV_RETR_LIST,CV_CHAIN_APPROX_NONE);
    cv::cvtColor(origin, origin, CV_GRAY2BGR);

    /*for (int i = 0; i <contour.size(); i++) {

        cv::drawContours(origin, contour, i, cv::Scalar(0, 0, 255), 2);
    }*/
    cv::drawContours(origin, contour, 1, cv::Scalar(0, 0, 255), 2);
  //  aia::imshow("Result", origin);
    return contour;
}


cv::Mat roiGen(cv::Mat img, cv::Mat mask) {

    int xPos, yPos;

    unsigned char* ythRow = mask.ptr<unsigned char>(10);

    for(int x=10; x<img.cols; x++){

        if (ythRow[x]==0) {

            xPos = x;
            break;
        }
    }

    for(int y=img.rows-1; y>0; y--){

        if (mask.at<unsigned char>(y, xPos) == 255) {

            yPos = y;
            break;
        }
    }

    if (yPos < 100)  {

        yPos = 3*img.rows/4;
    }

    return img(cv::Rect(0, 0, xPos, yPos));
}

int main() {

    try {

        //Load images

        std::string datasetPath = DATASET_PATH;
        std::vector <cv::Mat> images = getImagesInFolder(datasetPath + "images", "tif");
        std::vector <cv::Mat> truths = getImagesInFolder(datasetPath + "groundtruths", "tif");
        std::vector <cv::Mat> masks = getImagesInFolder(datasetPath + "masks", "png");
        std::vector <cv::Mat> results; bool toFlip = false;

        //cv::Mat dummy = cv::imread(std::string(DATASET_PATH) + std::string("images/24055464_ac3185e18ffdc7b6_MG_R_ML_ANON.tif"), CV_LOAD_IMAGE_UNCHANGED);
        //images.push_back(dummy);
        //dummy = cv::imread(std::string(DATASET_PATH) + std::string("masks/24055464_ac3185e18ffdc7b6_MG_R_ML_ANON.mask.png"), CV_LOAD_IMAGE_UNCHANGED);
        //masks.push_back(dummy);
        printf("Load done\n");

        for (int i = 0; i < images.size(); i++) {

            ///Flipping

            cv::Mat img = images[i].clone(); float resizeFactor = 0.2;

            if (masks[i].at<unsigned char>(30, 30) == 0) {

                cv::flip(img, img , 1);
                cv::flip(masks[i], masks[i], 1);
                toFlip = true;
            }

            cv::normalize(img, img, 0, 255, cv::NORM_MINMAX);
            img.convertTo(img, CV_8U);

            //aia::imshow("originale", img, true,0.20);

            ///Finding ROI


            cv::Mat roi = roiGen(img, masks[i]);
            //std::cout <<"cols: " <<roi.cols <<" rows: " <<roi.rows <<std::endl;
            cv::resize(roi, roi, cv::Size(resizeFactor * roi.cols, resizeFactor * roi.rows));
            // aia::imshow("roi", roi, true, 0.20);

            //Mean Shift Filter
            cv::Mat origin = roi.clone();
            cv::fastNlMeansDenoising(roi, roi, 15);
            int hs = 16, hr = 16;
            //aia::imshow("roi", roi);
            cv::cvtColor(roi, roi, CV_GRAY2BGR);
            cv::pyrMeanShiftFiltering(roi, roi, hs, hr, 0);
            cv::cvtColor(roi, roi, CV_BGR2GRAY);
            //aia::imshow("mean shift1", roi);

            //separateRegions(roi);

            std::vector<std::vector<cv::Point> > contours = myWatershed(roi, origin);
            cv::Mat bin = cv::Mat(origin.rows, origin.cols, CV_8U, cv::Scalar(0));

            for (int i = 0; i < contours.size(); i++) {

                cv::drawContours(bin, contours, i, cv::Scalar(255), 1);
            }

//            aia::imshow(" contours", bin);

            for (int i = 0; i < contours[1].size(); i++) {

                cv::Point p = contours[1][i];

                for (int x = 0; x < p.x; x++) {

                    bin.at<unsigned char>(p.y, x) = 255;
                }
            }
           // aia::imshow("prima resize", bin);
//	aia::imshow("contours", bin);

            cv::resize(bin, bin, cv::Size(roi.cols/resizeFactor,  roi.rows/resizeFactor));
            std::cout <<"cols: " <<bin.cols <<" rows: " <<bin.rows <<std::endl;
            //aia::imshow("resize", bin);
            cv::Mat result = cv::Mat(img.rows, img.cols, CV_8U, cv::Scalar(0));

            for (int y = 0; y < bin.rows; y++) {

                unsigned char *yRowBin = bin.ptr<unsigned char>(y);
                unsigned char *yRowRes = result.ptr<unsigned char>(y);

                for (int x = 0; x < bin.cols; x++) {

                    if (yRowBin[x] == 255) {

                        yRowRes[x] = 255;
                    }
                }
            }

            if (toFlip) {

                cv::flip(result, result, 1);
                cv::flip(masks[i], masks[i], 1);
                toFlip = false;
            }

            //cv::imwrite("C:/Users/giorgio/Desktop/bbb.tif", result);
            printf("%d \n", i);
            results.push_back(result);
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
