#include <iostream>
#include <vector>
#include <opencv2/core/mat.hpp>
#include <opencv/cxmisc.h>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgcodecs/imgcodecs_c.h>
#include <opencv/cv.hpp>
#include <list>
#include "3rdparty/aiacommon/aiaConfig.h"
#include "3rdparty/ucascommon/ucasExceptions.h"
#include "3rdparty/ucascommon/ucasStringUtils.h"
#include "3rdparty/ucascommon/ucasFileUtils.h"

#define DRIVE_PATH "/Users/Mariangela/Desktop/Università/Magistrale/1 anno - 2 semestre/EIID/AIA-Retinal-Vessel-Segmentation/datasets/DRIVE/"
#define DRIVE_EXTENSION ".tif"
#define DRIVE_SE_SIZE_MASK_EROSION 8
#define DRIVE_SE_SIZE_RECONSTRUCTION_EROSION 9
#define DRIVE_SE_SIZE_SUMTOPHAT 17
#define DRIVE_VIS_RESULTS_PATH "/Users/Mariangela/Desktop/Università/Magistrale/1 anno - 2 semestre/EIID/AIA-Retinal-Vessel-Segmentation/datasets/DRIVE/results/"
#define DRIVE_BIN_RESULTS_PATH "/Users/Mariangela/Desktop/Università/Magistrale/1 anno - 2 semestre/EIID/AIA-Retinal-Vessel-Segmentation/datasets/DRIVE/results_binarization/"
#define DRIVE_ROI 100

#define CHASEDB1_PATH "/Users/Mariangela/Desktop/Università/Magistrale/1 anno - 2 semestre/EIID/AIA-Retinal-Vessel-Segmentation/datasets/CHASEDB1/"
#define CHASEDB1_IMAGES_EXTENSION ".jpg"
#define CHASEDB1_MASKS_EXTENSION ".png"
#define CHASEDB1_TRUTHS_EXTENSION ".png"
#define CHASEDB1_SE_SIZE_MASK_EROSION 20
#define CHASEDB1_SE_SIZE_RECONSTRUCTION_EROSION 37
#define CHASEDB1_SE_SIZE_SUMTOPHAT 15
#define CHASEDB1_VIS_RESULTS_PATH "/Users/Mariangela/Desktop/Università/Magistrale/1 anno - 2 semestre/EIID/AIA-Retinal-Vessel-Segmentation/datasets/CHASEDB1/results/"
#define CHASEDB1_BIN_RESULTS_PATH "/Users/Mariangela/Desktop/Università/Magistrale/1 anno - 2 semestre/EIID/AIA-Retinal-Vessel-Segmentation/datasets/CHASEDB1/results_binarization/"


#define STARE_PATH "/Users/Mariangela/Desktop/Università/Magistrale/1 anno - 2 semestre/EIID/AIA-Retinal-Vessel-Segmentation/datasets/STARE/"
#define STARE_EXTENSION ".ppm"
#define STARE_MASKS_EXTENSION ".tif"
#define STARE_TRUTHS_EXTENSION ".ppm"
#define STARE_SE_SIZE_MASK_EROSION 15
#define STARE_SE_SIZE_RECONSTRUCTION_EROSION 37
#define STARE_SE_SIZE_SUMTOPHAT 15
#define STARE_VIS_RESULTS_PATH "/Users/Mariangela/Desktop/Università/Magistrale/1 anno - 2 semestre/EIID/AIA-Retinal-Vessel-Segmentation/datasets/STARE/results/"
#define STARE_BIN_RESULTS_PATH "/Users/Mariangela/Desktop/Università/Magistrale/1 anno - 2 semestre/EIID/AIA-Retinal-Vessel-Segmentation/datasets/STARE/results_binarization/"


// utility function that rotates 'img' by step*90°
// step = 0 --> no rotation
// step = 1 --> 90° CW rotation
// step = 2 --> 180° CW rotation
// step = 3 --> 270° CW rotation
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


// create 'n' rectangular Structuring Elements (SEs) at different orientations spanning the whole 360°
// return vector of 'width' x 'width' uint8 binary images with non-black pixels being the SE
// parameter width: SE width (must be odd)
// parameter height: SE height (must be odd)
// parameter n: number of SEs
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

//prende in input l'immagine, restituisce una lista di Point, ogni Point rappresenta un end point
std::list<cv::Point> endPoint(cv::Mat mat){

    cv::Mat img = mat.clone();
    std::list<cv::Point> endPointList;

    try
    {
        //aia::imshow("original image", img);

        // STEP 1: SKELETONIZATION
        // to perform skeletonization, we use two 'edge'-like SEs
        // along with their rotated versions
        std::vector<cv::Mat> skel_SEs;

        cv::Mat se0(3, 3, CV_32F);
        se0.at<float> (0,0) = -1;
        se0.at<float> (0,1) = -1;
        se0.at<float> (0,2) = -1;
        se0.at<float> (1,0) = 0;
        se0.at<float> (1,1) = 1;
        se0.at<float> (1,2) = 0;
        se0.at<float> (2,0) = 1;
        se0.at<float> (2,1) = 1;
        se0.at<float> (2,2) = 1;
        
        skel_SEs.push_back(se0);

        cv::Mat se1(3, 3, CV_32F);
        se1.at<float> (0,0) = 0;
        se1.at<float> (0,1) = -1;
        se1.at<float> (0,2) = -1;
        se1.at<float> (1,0) = 1;
        se1.at<float> (1,1) = 1;
        se1.at<float> (1,2) = -1;
        se1.at<float> (2,0) = 1;
        se1.at<float> (2,1) = 1;
        se1.at<float> (2,2) = 0;
        
        skel_SEs.push_back(se1);

        // skeletonization is based on iterative thinning using hit-or-miss transform
        cv::Mat skeleton = img.clone();
        cv::Mat skeleton_prev;
        do
        {
            // we need to memorize both previous and current versions of the skeleton
            // in order to detect if no more changes occurred (convergence)
            skeleton_prev = skeleton.clone();

            for(int i=0; i<skel_SEs.size(); i++)
            {
                // perform all 90° rotations so that thinning is anisotropic
                for(int j=0; j<4; j++)
                {
                    cv::Mat hitormiss;
                    cv::morphologyEx(skeleton, hitormiss, cv::MORPH_HITMISS, rotate90(skel_SEs[i], j));
                    skeleton -= hitormiss;
                }
            }

            // display intermediate results with a delay of 200ms between two iterations
            //cv::imshow("skeletonization", skeleton);
            if (cv::waitKey(200)>=0)
                cv::destroyWindow("skeletonization");
        }
        while (cv::countNonZero(skeleton_prev - skeleton) > 0);	// convergence = no more changes
        //aia::imshow("skeleton", skeleton);
        //cv::imwrite("C:/work/skeleton.png", skeleton);

        // STEP 2: PRUNING (to remove spurious junctions generated by skeletonization)
        // to perform pruning, we use an 'endpoint'-like SE
        // along with its rotated versions
        std::vector <cv::Mat> prun_SEs;

        cv::Mat se2(3, 3, CV_32F);
        se2.at<float> (0,0) = 0;
        se2.at<float> (0,1) = 0;
        se2.at<float> (0,2) = 0;
        se2.at<float> (1,0) = -1;
        se2.at<float> (1,1) = 1;
        se2.at<float> (1,2) = -1;
        se2.at<float> (2,0) = -1;
        se2.at<float> (2,1) = -1;
        se2.at<float> (2,2) = -1;
        
        prun_SEs.push_back(se2);

        // pruning is based on iterative subtractions (like thinning) of the
        // endpoint structures detected using the hit-or-miss transform
        cv::Mat pruned = skeleton.clone();
        int pruning_iterations = 10;		// too many pruning iterations will destroy the tree;
        // we only need to remove spurious junctions generated
        // by skeletonization, which are usually small
        for(int k=0; k<pruning_iterations; k++)
        {
            for(int i=0; i<prun_SEs.size(); i++)
            {
                // perform all 90° rotations so that pruning is anisotropic
                for(int j=0; j<4; j++)
                {
                    cv::Mat hitormiss;
                    cv::morphologyEx(pruned, hitormiss, cv::MORPH_HITMISS, rotate90(prun_SEs[i], j));
                    pruned -= hitormiss;
                }
            }

            // display intermediate results with a delay of 200ms between two iterations
            //cv::imshow("pruning", pruned);
            if (cv::waitKey(200)>=0)
                cv::destroyWindow("pruning");
        }
        //aia::imshow("pruned", pruned);

        // STEP 3: detection of endpoints
        std::vector <cv::Mat> jun_SEs;

        cv::Mat se3(3, 3, CV_32F);
        se3.at<float> (0,0) = -1;
        se3.at<float> (0,1) = -1;
        se3.at<float> (0,2) = -1;
        se3.at<float> (1,0) = -1;
        se3.at<float> (1,1) = 1;
        se3.at<float> (1,2) = 1;
        se3.at<float> (2,0) = -1;
        se3.at<float> (2,1) = -1;
        se3.at<float> (2,2) = -1;
        
        jun_SEs.push_back(se3);

        // endpoint detection is the union of multiple hit-or-miss transforms
        // (i.e. one hit-or-miss for each junction-pattern to be detected)
        cv::Mat junctions(img.rows, img.cols, CV_8U, cv::Scalar(0));
        for(int i=0; i<jun_SEs.size(); i++)
        {
            // perform all 90° rotations so that junction detection is anisotropic
            for(int j=0; j<4; j++)
            {
                cv::Mat hitormiss;
                cv::morphologyEx(pruned, hitormiss, cv::MORPH_HITMISS, rotate90(jun_SEs[i], j));
                junctions += hitormiss;
                for(int y=0; y < hitormiss.rows; y++){
                    unsigned char* yRowFound = hitormiss.ptr<unsigned char>(y);
                    for (int x=0; x < hitormiss.cols; x++){
                        if (yRowFound[x] == 255){
                            cv::Point p;
                            p.x = x; p.y = y;
                            endPointList.push_back(p);
                        }
                    }
                }
            }
        }
        //aia::imshow("endpoints", junctions);


        // overlay endpoints as red circles on the original image
        // endpoints are single pixels in the 'endpoints' image
        // we use dilation with a circle-like SE
        cv::dilate(junctions, junctions, cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(7,7)));
        cv::cvtColor(img, img, cv::COLOR_GRAY2BGR);
        img.setTo(cv::Scalar(0, 0, 255), junctions);
       // aia::imshow("result", img);

        return endPointList;
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

void fun() {

    cv::Mat img = cv::imread("/Users/Mariangela/Desktop/roi.tif", CV_LOAD_IMAGE_GRAYSCALE);

    cv::fastNlMeansDenoising(img, img);

    cv::Ptr<cv::CLAHE> clahe = cv::createCLAHE();
    clahe -> setTilesGridSize(cv::Size(img.cols/7, img.rows/7));
    clahe -> apply(img, img);

    cv::morphologyEx(img, img, CV_MOP_ERODE, cv::getStructuringElement(CV_SHAPE_ELLIPSE, cv::Size(3,3)));

    cv::threshold(img, img, 0, 255, CV_THRESH_BINARY_INV | CV_THRESH_OTSU);

    aia::imshow("CLAHE", img);
}

int main() {

    //fun();
    //getchar();

    std::string datasetPath = DRIVE_PATH;
    std::vector <cv::Mat> images = getImagesInFolder(datasetPath + "images",DRIVE_EXTENSION);
    std::vector <cv::Mat> truthsMultiChannel = getImagesInFolder(datasetPath + "groundtruths", DRIVE_EXTENSION);
    std::vector <cv::Mat> masksMultiChannel  = getImagesInFolder(datasetPath + "masks", DRIVE_EXTENSION);
    std::vector <cv::Mat> masks;
    std::vector <cv::Mat> truths;
    std::vector <cv::Mat> results;

    for (int i = 0; i < images.size(); i++) {


        ///GREEN CHANNEL EXTRACTION
        std::vector<cv::Mat> channels;
        cv::split(images[i], channels);
        cv::Mat green_channel = channels[1].clone(); cv::Mat greenChannelOri = green_channel.clone();
        //aia::imshow("green channel", green_channel);

        green_channel.convertTo(green_channel, CV_8U);
        //truths[i].convertTo(truths[i], CV_8U);
        //aia::imshow("green channel", green_channel);

        channels.clear();
        cv::split(masksMultiChannel[i], channels);
        masks.push_back(channels[0]);

        channels.clear();
        cv::split(truthsMultiChannel[i], channels);
        truths.push_back(channels[0]);



        ///ELIMINATE IRREGULAR BRIGHT REGIONS

        //1°: EROSION
        cv::Mat eroded;
        cv::morphologyEx(green_channel, eroded, CV_MOP_ERODE, cv::getStructuringElement(CV_SHAPE_RECT, cv::Size(DRIVE_SE_SIZE_RECONSTRUCTION_EROSION,DRIVE_SE_SIZE_RECONSTRUCTION_EROSION)));
        //aia::imshow("erosion", eroded);



        //2°: DILATATION OF THE ERODED IMAGE ITERATIVELY
        // - eroded image -> marker
        // - origimal image -> mask

        cv::Mat marker = eroded.clone();
        cv::Mat marker_prev;
        cv::Mat reconstruction = green_channel.clone();
        int it = 0;
        do
        {
            // make a backup copy of the previous marker
            marker_prev = marker.clone();

            // geodesic dilation ( = dilation + pointwise minimum with mask)
            cv::morphologyEx(marker, marker, CV_MOP_DILATE, cv::getStructuringElement(CV_SHAPE_RECT, cv::Size(3,3)));
            marker = cv::min(marker, green_channel);

        }
        while( cv::countNonZero(marker - marker_prev) > 0) ;

        //aia::imshow("Reconstructed", marker);
        reconstruction = (reconstruction - marker);
        //aia::imshow("G - reconstructed", reconstruction);

        //green_channel = green_channel - reconstruction;

        green_channel = marker.clone();



        ///NOISE REDUCTION
        cv::fastNlMeansDenoising(green_channel, green_channel, 3); cv::Mat denoised = green_channel.clone();



        ///SUM TOP-HAT
        cv::Mat green_channel_reverse = 255 - green_channel;
        cv::Mat top_hat, THTransform;
        top_hat = cv::Mat::zeros(images[i].size(), 0);

        std::vector<cv::Mat> tiltedSE ;
        tiltedSE = createTiltedStructuringElements(3, 9, 18);
        cv::Mat dest;

        for (int i = 0; i < tiltedSE.size(); i++) {
            cv::morphologyEx(green_channel_reverse, dest, cv::MORPH_TOPHAT, tiltedSE[i]);
            top_hat = top_hat + dest;
        }

        THTransform = green_channel_reverse - top_hat;
        cv::Mat foundRegion = (green_channel_reverse - THTransform);
        //aia::imshow("top hat", top_hat);
        //aia::imshow("THTransform", THTransform);
        aia::imshow("foundRegion", foundRegion);
        //cv::imwrite("/Users/Mariangela/Desktop/Università/Magistrale/1 anno - 2 semestre/EIID/AIA-Retinal-Vessel-Segmentation/datasets/DRIVE/sum top hat 17 17/img.tif", foundRegion);
        //getchar();

        ///NOISE REDUCTION
        cv::fastNlMeansDenoising(foundRegion, foundRegion, 3);
        //aia::imshow("noise reduction 2", foundRegion);



        ///MASK EROSION
        cv::morphologyEx(masks[i], masks[i], CV_MOP_ERODE, cv::getStructuringElement(CV_SHAPE_ELLIPSE, cv::Size(DRIVE_SE_SIZE_MASK_EROSION, DRIVE_SE_SIZE_MASK_EROSION)));




        ///ESTRAZIONE FOV
        for (int y = 0; y < foundRegion.rows; y++) {
            unsigned char* yRowFound = foundRegion.ptr<unsigned char>(y);
            unsigned char* yRowMask = masks[i].ptr<unsigned char>(y);

            for (int x = 0; x < foundRegion.cols; x++) {

                if (yRowMask[x] == 0) {

                    yRowFound[x] = 0;
                }
            }
        }



        ///THRESHOLDING
        cv::Mat thresholded;
        cv::threshold(foundRegion, thresholded, 0, 255, CV_THRESH_BINARY | CV_THRESH_OTSU);
      //  aia::imshow("binarizzazione", thresholded);

        results.push_back(thresholded);


        ///ENDPOINT FINDER

        std::list<cv::Point> l = endPoint(thresholded);




        for (cv::Point p : l) {

            if (!(p.x - DRIVE_ROI/2 < 0 || p.y - DRIVE_ROI/2 < 0 || p.x + DRIVE_ROI/2 >= images[i].cols || p.y + DRIVE_ROI/2 >= images[i].rows)) {

                cv::Mat gc = greenChannelOri.clone();

                cv::Mat img = gc(cv::Rect(p.x - DRIVE_ROI/2, p.y - DRIVE_ROI/2, DRIVE_ROI, DRIVE_ROI));
                cv::Mat binRoi = thresholded(cv::Rect(p.x - DRIVE_ROI/2, p.y - DRIVE_ROI/2, DRIVE_ROI, DRIVE_ROI));

                aia::imshow("img", img, false);

                cv::fastNlMeansDenoising(img, img);

                cv::Ptr<cv::CLAHE> clahe = cv::createCLAHE();
                clahe -> setTilesGridSize(cv::Size(img.cols/4, img.rows/4));
                clahe -> apply(img, img);

                cv::morphologyEx(img, img, CV_MOP_ERODE, cv::getStructuringElement(CV_SHAPE_ELLIPSE, cv::Size(3,3)));

                cv::threshold(img, img, 0, 255, CV_THRESH_BINARY_INV | CV_THRESH_OTSU);

                binRoi = binRoi + img;

                //aia::imshow("AAA", binRoi);
                //aia::imshow("img", img);
            }
        }

        std::cout << i << std::endl;

        aia::imshow("binarizzazione", thresholded);
    }



    std::vector <cv::Mat> visual_results;
    double ACC = accuracy(results, truths, masks, &visual_results);
    printf("Accuracy = %.2f%%\n", ACC*100);

    for (int i = 0; i < visual_results.size(); i++) {

        std::string pathNameVisualResults = DRIVE_VIS_RESULTS_PATH + std::to_string(i+1) + ".tif";
        cv::imwrite(pathNameVisualResults, visual_results[i]);


        std::string pathNameResults = DRIVE_BIN_RESULTS_PATH + std::to_string(i+1) + ".tif";
        cv::imwrite(pathNameResults, results[i]);


    }

    return 0;
}