#include <opencv2/core/mat.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv/cv.hpp>
#include <iostream>
#include "3rdparty/aiacommon/aiaConfig.h"
#include "3rdparty/ucascommon/ucasExceptions.h"
#include "3rdparty/ucascommon/ucasFileUtils.h"
#include "3rdparty/ucascommon/ucasImageUtils.h"

// include my project functions


#ifndef CV_CAST_8U
#define CV_CAST_8U(t) (uchar)(!((t) & ~255) ? (t) : (t) > 0 ? 255 : 0)
#endif


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


int main()
{
    try {
        cv::Mat image = cv::imread("/Users/Mariangela/Desktop/Università/Magistrale/1 anno - 2 semestre/EIID/AIA-Retinal-Vessel-Segmentation/datasets/DRIVE/images/01.tif", CV_LOAD_IMAGE_UNCHANGED);
        cv::Mat mask = cv::imread("/Users/Mariangela/Desktop/Università/Magistrale/1 anno - 2 semestre/EIID/AIA-Retinal-Vessel-Segmentation/datasets/DRIVE/masks/01_mask.tif", CV_LOAD_IMAGE_UNCHANGED);
        cv::Mat ground_truth = cv::imread("/Users/Mariangela/Desktop/Università/Magistrale/1 anno - 2 semestre/EIID/AIA-Retinal-Vessel-Segmentation/datasets/DRIVE/images/01.tif", CV_LOAD_IMAGE_UNCHANGED);

        std::vector<cv::Mat> channels;

        cv::split(image, channels);
        cv::Mat green_channel = channels[1];

        green_channel.convertTo(green_channel, CV_8U);
        ground_truth.convertTo(ground_truth, CV_8U);


        // NOISE REDUCTION

        cv::fastNlMeansDenoising(green_channel, green_channel, 3);

        aia::imshow("denoising", green_channel);

        // TOP-HAT TRANSFORM

        cv::Mat green_channel_reverse = 255 - green_channel;

        //aia::imshow("green_channel_reverse", green_channel_reverse);

        cv::Mat dest, THTransform;

        cv::morphologyEx(green_channel_reverse, dest, cv::MORPH_TOPHAT, cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(20, 20)));

        THTransform = green_channel_reverse - dest;

        cv::Mat foundRegion = 10 * (green_channel_reverse - THTransform);

        aia::imshow("found region", THTransform);


        // SHARPENING
/*
        int sharpening_factor_x10 = 5;

        float k = sharpening_factor_x10/10.0f;

        cv::Mat laplacian(3, 3, CV_32F);
        laplacian.at<float> (0,0) = -k;
        laplacian.at<float> (0,1) = -k;
        laplacian.at<float> (0,2) = -k;
        laplacian.at<float> (1,0) = -k;
        laplacian.at<float> (1,1) = 1 + 8 * k;
        laplacian.at<float> (1,2) = -k;
        laplacian.at<float> (2,0) = -k;
        laplacian.at<float> (2,1) = -k;
        laplacian.at<float> (2,2) = -k;

        // a better option is to cut the peaks directly by storing
        // the result in 8U
        cv::filter2D(foundRegion, foundRegion, CV_8U, laplacian);

        aia::imshow("sharpening", foundRegion);
*/



        // CLAHE
/*
         int block_size = 8;

         // instance and apply CLAHE algorithm
         cv::Ptr<cv::CLAHE> clahe = cv::createCLAHE();
         clahe->setTilesGridSize(cv::Size(foundRegion.cols/block_size, foundRegion.rows/block_size));
         clahe->apply(foundRegion, foundRegion);

        aia::imshow("CLAHE", foundRegion);
*/

        // RECONSTRUCTION
        cv::Mat tophat = foundRegion.clone();
        cv::Mat marker;
        cv::morphologyEx(green_channel_reverse, marker, CV_MOP_OPEN, cv::getStructuringElement(CV_SHAPE_ELLIPSE, cv::Size(50,50)));
        aia::imshow("Marker", marker);
        //cv::imwrite(std::string(EXAMPLE_IMAGES_PATH) + "/galaxy_marker.jpg", marker);


        cv::Mat marker_prev;
        int it = 0;
        do
        {
            // make a backup copy of the previous marker
            marker_prev = marker.clone();

            // geodesic dilation ( = dilation + pointwise minimum with mask)
            cv::morphologyEx(marker, marker, CV_MOP_DILATE, cv::getStructuringElement(CV_SHAPE_RECT, cv::Size(3,3)));
            marker = cv::min(marker, green_channel_reverse);

            // display reconstruction in progress
            printf("it = %d\n", ++it);
            //aia::imshow("marker", marker);

        }
        while( cv::countNonZero(marker - marker_prev) > 0) ;

        aia::imshow("Reconstructed", marker);
        foundRegion = (green_channel_reverse - marker);
        aia::imshow("G - reconstructed", foundRegion);

        foundRegion = 0.30 * tophat + 0.70 * foundRegion;

        aia::imshow("somma pesata", foundRegion);

        // ESTRAZIONE FOV

        for (int y = 0; y < foundRegion.rows; y++) {

            unsigned char* yRowFound = foundRegion.ptr<unsigned char>(y);
            unsigned char* yRowMask = mask.ptr<unsigned char>(y);

            for (int x = 0; x < foundRegion.cols; x++) {

                if (yRowMask[x] != 255) {

                    yRowFound[x] = 0;
                }
            }
        }

        aia::imshow("internal mask", foundRegion);

        // NOISE REDUCTION

        cv::fastNlMeansDenoising(foundRegion, foundRegion, 3);
        aia::imshow("noise reduction", foundRegion);


        //THRESHOLDING

        cv::Mat thresholded;

        cv::threshold(foundRegion, thresholded, 0, 255, CV_THRESH_BINARY | CV_THRESH_OTSU);

        aia::imshow("binarizzazione", thresholded);

        //aia::imshow("opening", thresholded);

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

