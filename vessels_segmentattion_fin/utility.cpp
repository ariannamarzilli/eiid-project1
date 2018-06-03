//
// Created by Mariangela Evangelista on 20/05/18.
//

#include <opencv2/core/mat.hpp>
#include <opencv2/core.hpp>
#include <opencv/cv.hpp>
#include "3rdparty/ucascommon/ucasExceptions.h"
#include "3rdparty/ucascommon/ucasStringUtils.h"
#include "3rdparty/ucascommon/ucasFileUtils.h"
#include "3rdparty/aiacommon/aiaConfig.h"

#include "utility.h"

cv::Mat rotate90(cv::Mat img, int step)
{
    cv::Mat img_rot;

    // adjust step in case it is negative
    if(step < 0)
        step = -step;
    // adjust step in case it exceeds 4
    step = step%4;

    // no rotation
    if(step == 0)
        img_rot = img;
        // 90° CW rotation
    else if(step == 1)
    {
        cv::transpose(img, img_rot);
        cv::flip(img_rot, img_rot, 1);
    }
        // 180° CW rotation
    else if(step == 2)
        cv::flip(img, img_rot, -1);
        // 270° CW rotation
    else if(step == 3)
    {
        cv::transpose(img, img_rot);
        cv::flip(img_rot, img_rot, 0);
    }

    return img_rot;
}


std::vector<cv::Mat> createTiltedStructuringElements(int width, int height, int n) throw (ucas::Error) {
    // check preconditions
    if (width % 2 == 0) {
        printf("errore\n");
        throw ucas::Error(ucas::strprintf("Structuring element width (%d) is not odd", width));
    }
    if (height % 2 == 0) {
        printf("errore\n");
        throw ucas::Error(ucas::strprintf("Structuring element height (%d) is not odd", height));
    }
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


std::vector < cv::Mat > getImagesInFolder(std::string folder, std::string ext, bool force_gray) throw (aia::error)
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


double accuracy(
        std::vector <cv::Mat> & segmented_images,		// (INPUT)  segmentation results we want to evaluate (1 or more images, treated as binary)
        std::vector <cv::Mat> & groundtruth_images,     // (INPUT)  reference/manual/groundtruth segmentation images
        std::vector <cv::Mat> & mask_images,			// (INPUT)  mask images to restrict the performance evaluation within a certain region
        std::vector <cv::Mat> * visual_results      	// (OUTPUT) (optional) false color images displaying the comparison between automated segmentation results and groundtruth

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