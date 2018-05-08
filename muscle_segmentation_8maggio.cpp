#include <iostream>
#include <vector>
#include "3rdparty/ucascommon/ucasExceptions.h"
#include "3rdparty/ucascommon/ucasStringUtils.h"
#include "3rdparty/aiacommon/aiaConfig.h"
#include "3rdparty/ucascommon/ucasFileUtils.h"

#define DATASET_PATH "/Users/Mariangela/Desktop/AIA-Pectoral-Muscle-Segmentation/dataset/"
#define DATASET_VIS_RESULTS_PATH "/Users/Mariangela/Desktop/AIA-Pectoral-Muscle-Segmentation/dataset/results/"


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
        cv::Mat dummy = cv::imread(std::string(DATASET_PATH) + std::string("images/53586388_dda3c6969a34ff8e_MG_R_ML_ANON.tif"), CV_LOAD_IMAGE_UNCHANGED);
        images.push_back(dummy);
        dummy = cv::imread(std::string(DATASET_PATH) + std::string("masks/53586388_dda3c6969a34ff8e_MG_R_ML_ANON.mask.png"), CV_LOAD_IMAGE_UNCHANGED);
        masks.push_back(dummy);

        printf("Load done\n");

        for (int i = 0; i < images.size(); i++) {


            ///Flipping
            cv::Mat img = images[i].clone();
            if (masks[i].at<unsigned char>(30, 30) == 0) {

                cv::flip(img, img , 1);
                cv::flip(masks[i], masks[i], 1);
                toFlip = true;
            }

            cv::normalize(img, img, 0, 255, cv::NORM_MINMAX);
            img.convertTo(img, CV_8U);
            cv::Mat original;
            cv::resize(img, original, cv::Size(700, 800));
            aia::imshow("originale", original);



            ///Finding ROI
            int xPos, yPos;
            unsigned char* ythRow = masks[i].ptr<unsigned char>(10);
            for(int x=10; x<img.cols; x++){
                if (ythRow[x]==0) {
                    xPos = x;
                    break;
                }
            }

            for(int y=img.rows-1; y>0; y--){
                if (masks[i].at<unsigned char>(y, xPos) == 255) {
                    yPos = y;
                    break;
                }
            }
            printf("%d\n", xPos);
            printf("%d\n", yPos);
            cv::Mat roi = img(cv::Rect(0, 0, xPos, yPos));
            cv::resize(roi, roi, cv::Size(xPos/4, yPos/4));
            aia::imshow("roi", roi);



            ///Mean Shift
            cv::Mat meanShift = roi.clone();
            int hs = 20, hr = 20;
            cv::Mat multiChannelSmooth;

            cv::cvtColor(meanShift, multiChannelSmooth, CV_GRAY2BGR);
            cv::pyrMeanShiftFiltering(multiChannelSmooth, meanShift, hs, hr, 0);
            cv::cvtColor(meanShift, meanShift, CV_BGR2GRAY);
            aia::imshow("Mean Shift", meanShift);



            ///Enhancement
            cv::Mat equalized = meanShift.clone();
            int L = 256;
            float gamma = 150.0f;
            float c = std::pow(L - 1, 1 - gamma / 100.0f);

            for (int y = 0; y < equalized.rows; y++) {
                unsigned char *yRow = equalized.ptr < unsigned char > (y);

                for (int x = 0; x < equalized.cols; x++) {
                    yRow[x] = c * std::pow(yRow[x], gamma / 100.0f);
                }
            }
            aia::imshow("Gamma correction", equalized);



            ///Gaussian blur
            cv::Mat blur;
            cv::GaussianBlur(equalized, blur, cv::Size(0,0), (1.0), (1.0));
            aia::imshow("gaussian blur", blur);



            ///Edge detection
            cv::Mat edges;

            //Sobel
            /*cv::Mat dx, dy;
            cv::Mat abs_dx, abs_dy;

            // Gradient X
            cv::Sobel( meanShift, dx, CV_8U, 1, 0, 3);
            convertScaleAbs( dx, abs_dx);

            // Gradient Y
            cv::Sobel( meanShift, dy, CV_8U, 0, 1, 3);
            convertScaleAbs( dy, abs_dy);

            cv::addWeighted( abs_dx, 0.3, abs_dy, 0.7, 0, edges);
            aia::imshow("Edges - sobel", 10*edges);

            //Laplacian
            cv::Laplacian(meanShift, edges, CV_8U, 3, 1, 0);
            aia::imshow("Edges - laplacian", edges);*/

            //Canny
            cv::Canny(blur, edges, 15, 45);
            aia::imshow("Edges - canny", edges);



            ///Hough Lines - codice internet
            // Declare the output variables
            cv::Mat src, dst, cdst;
            src = edges.clone();

            // Copy edges to the images that will display the results in BGR
            cvtColor(src, cdst, CV_GRAY2BGR);
            cdst = cdst.clone();

            //Standard Line Transform
            /*std::vector<cv::Vec2f> lines;
            HoughLines(dst, lines, 1, 1, 10, 0, 0 );

            std::cout <<"#linee: " <<lines.size() <<std::endl;

            for( size_t i = 0; i < lines.size(); i++ )
            {
                float rho = lines[i][0], theta = lines[i][1];
                cv::Point pt1, pt2;
                double a = cos(theta), b = sin(theta);
                double x0 = a*rho, y0 = b*rho;
                pt1.x = cvRound(x0 + 1000*(-b));
                pt1.y = cvRound(y0 + 1000*(a));
                pt2.x = cvRound(x0 - 1000*(-b));
                pt2.y = cvRound(y0 - 1000*(a));
                line( cdst, pt1, pt2, cv::Scalar(0,0,255), 3, CV_AA);
            }*/


            // Probabilistic Line Transform
            std::vector<cv::Vec4i> linesP; // will hold the results of the detection
            HoughLinesP(dst, linesP, 30, CV_PI/180, 1); // runs the actual detection

            std::cout <<"numero linee: " <<linesP.size() <<std::endl;

            // Draw the lines
            for( size_t i = 0; i < linesP.size(); i++ )
            {
                cv::Vec4i l = linesP[i];
                line( cdst, cv::Point(l[0], l[1]), cv::Point(l[2], l[3]), cv::Scalar(0,0,255), 3, cv::LINE_AA);
            }
            // Show results
            aia::imshow("Source", src);
            aia::imshow("Detected Lines (in red) - Standard Hough Line Transform", cdst);
            //aia::imshow("Detected Lines (in red) - Probabilistic Line Transform", cdstP);





            ///Binarize
            cv::Mat bin = blur.clone();
            cv::threshold(bin, bin, 0, 255, CV_THRESH_BINARY | CV_THRESH_OTSU);
            //cv::resize(bin, resized, cv::Size(700, 800));
            aia::imshow("Binarized", bin);



            std::vector <std::vector<cv::Point> > contours;
            cv::Mat binCopy = bin.clone();

            cv::findContours(binCopy, contours, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_NONE);

            int goodContourn = 0;

            for (int i = 0; i < contours.size(); i++) {

                if (cv::pointPolygonTest(contours[i], cv::Point(10, 10), false) >= 0) {

                    goodContourn = i;
                }
            }

            cv::fillConvexPoly(bin, contours[goodContourn], cv::Scalar(255));

            for (int y = 0; y < bin.rows ; y++) {

                unsigned char *yRow = bin.ptr < unsigned char > (y);
                unsigned char *yRowMask = masks[i].ptr < unsigned char > (y);

                for (int x = 0; x < bin.cols; x++) {

                    if (yRowMask[x] == 255) {

                        if (cv::pointPolygonTest(contours[goodContourn], cv::Point(x, y), false) < 0) {

                            yRow[x] = 0;
                        }
                    }
                }
            }

            //cv::resize(bin, resized, cv::Size(700, 800));
            aia::imshow("Result", bin);




            ///
            /*for (int y = 0; y < bin.rows; y++) {
                for (int x = 0; x < bin.cols; x++) {
                    if (final.at<unsigned char>(y,x) == 255) {
                        //printf("bianco!\n");
                        original.at<unsigned char>(y,x) = 255;
                    }
                }
            }
            cv::resize(original, original, cv::Size(700, 800));
            aia::imshow("boh", original);*/



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
    return 0;
}