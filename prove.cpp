#include <iostream>
#include <opencv2/core/mat.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv/cv.hpp>
#include "3rdparty/aiacommon/aiaConfig.h"
#include "3rdparty/ucascommon/ucasImageUtils.h"

#ifndef CV_CAST_8U
#define CV_CAST_8U(t) (uchar)(!((t) & ~255) ? (t) : (t) > 0 ? 255 : 0)
#endif


namespace eiid {

    int gammaX100 = 200;
    int L = 256;

    // effettua l'equalizzazione dell'immagine srcarr
    void cvEqualizeHist( const CvArr* srcarr, CvArr* dstarr, CvMat* mask )
    {
        using namespace cv;

        CvMat sstub, *src = cvGetMat(srcarr, &sstub);
        CvMat dstub, *dst = cvGetMat(dstarr, &dstub);

        CV_Assert( CV_ARE_SIZES_EQ(src, dst) && CV_ARE_TYPES_EQ(src, dst) &&
                   CV_MAT_TYPE(src->type) == CV_8UC1 );

        CV_Assert( CV_ARE_SIZES_EQ(src, mask) && CV_MAT_TYPE(mask->type) == CV_8UC1);


        int height = src->rows;
        int width = src->cols;
        CvSize size(width, height);

        //CvSize size = cvGetMatSize(src);

        if( CV_IS_MAT_CONT(src->type & dst->type) )
        {
            size.width *= size.height;
            size.height = 1;
        }
        int x, y;
        const int hist_sz = 256;
        int hist[hist_sz];
        memset(hist, 0, sizeof(hist));

        for( y = 0; y < size.height; y++ )
        {
            const uchar* sptr = src->data.ptr + src->step*y;
            const uchar* mptr = mask->data.ptr + mask->step*y;
            for( x = 0; x < size.width; x++ )
                if (mptr[x]) hist[sptr[x]]++;
        }

        float scale = 255.f/(cvCountNonZero(mask));
        int sum = 0;
        uchar lut[hist_sz+1];

        for( int i = 0; i < hist_sz; i++ )
        {
            sum += hist[i];
            int val = cvRound(sum*scale);
            lut[i] = CV_CAST_8U(val);
        }

        lut[0] = 0;
        cvSetZero(dst);
        for( y = 0; y < size.height; y++ )
        {
            const uchar* sptr = src->data.ptr + src->step*y;
            const uchar* mptr = mask->data.ptr + mask->step*y;
            uchar * dptr = dst->data.ptr + dst->step*y;
            for( x = 0; x < size.width; x++ )
                if (mptr[x]) dptr[x] = lut[sptr[x]];
        }
    }

}




int main() {

    cv::Mat image;
    cv::Mat mask;
    cv::Mat hist;

    std::string path = "/Users/Mariangela/Desktop/Università/Magistrale/1 anno - 2 semestre/EIID/AIA-Retinal-Vessel-Segmentation/datasets/DRIVE/images/01.tif";
    image = cv::imread(path);
    mask = cv::imread("/Users/Mariangela/Desktop/Università/Magistrale/1 anno - 2 semestre/EIID/AIA-Retinal-Vessel-Segmentation/datasets/DRIVE/masks/01_mask.tif", CV_LOAD_IMAGE_GRAYSCALE);

    cv::Mat green_channel;
    std::vector <cv::Mat> channels;
    cv::split(image, channels);

    green_channel = channels[1];
    green_channel.convertTo(green_channel, CV_8U);
    mask.convertTo(mask, CV_8U);
    cv::threshold(mask, mask, 0, 255, CV_THRESH_OTSU);


    /*---------------------------------
     *      gamma transformation
     */
/*
    float c = std::pow(L-1, 1-gammaX100/100.0f);
    if(green_channel.depth() == CV_8U)
    {

        for(int y=0; y<green_channel.rows; y++)
        {
            unsigned char* data_row_mask = mask.ptr<unsigned char>(y);
            unsigned char* data_row = green_channel.ptr<unsigned char>(y);
            for(int x=0; x<green_channel.cols; x++)
                if (data_row_mask[x] == 255) {
                    data_row[x] = c*std::pow(data_row[x], gammaX100/100.0f);
                }

        }
    }

    cv::namedWindow("enhanced");
    aia::imshow("enhanced", green_channel, false);
    aia::imshow("default", green_channel_copy);
    aia::imshow("hist", hist);

 */


    /*----------------------------------
     *       Non local means denoising
     */

    cv::Mat noise_green_channel;
    cv::Mat denoised_green_channel;
    green_channel.copyTo(noise_green_channel);


    /*
    noise_green_channel = green_channel(cv::Rect(396, 396, 100, 100));
    aia::imshow("roi", noise_green_channel, false);
*/
    cv::fastNlMeansDenoising(noise_green_channel, denoised_green_channel, 3);



    /*---------------------------------
     *      Equalizzazione istogramma
     */
/*
    IplImage src = denoised_green_channel;

    IplImage* dest;
    dest = cvCreateImage(cvSize(green_channel.cols,green_channel.rows),8,1);

    CvMat cvMask = mask;

    eiid::cvEqualizeHist(&src, dest, &cvMask);

    cv::Mat dstMat = cv::cvarrToMat(dest);

    aia::imshow("equalizzazione hist", dstMat, false);
*/
    cv::Mat dstMat = denoised_green_channel;
    //cv::equalizeHist(denoised_green_channel, dstMat);


   /*----------------------------------
    *       Sharpening
    */

/*
    // sharpening
    cv::Mat sharpened;
    int sharpening_factor_x10 = 50;

    float k = sharpening_factor_x10/10.0f;
    //(cv::Mat_<float>(3, 3) << -k, -k, -k,  -k, 1 + 8 * k,  -k, -k,  -k, -k);

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

    // if we store the result in a float image and then normalize,
    // the true image content will be compressed in the centermost
    // region of the histogram (=image becomes gray). This is because
    // of the negative (<0) and positive (>255) peaks generated by
    // sharpening at the edges (see lecture slides)
    //cv::filter2D(sharpened, sharpened, CV_32F, laplacian);
    //cv::normalize(sharpened, sharpened, 0, 255, cv::NORM_MINMAX);
    //sharpened.convertTo(sharpened, CV_8U);

    // a better option is to cut the peaks directly by storing
    // the result in 8U
    cv::filter2D(denoised_green_channel, sharpened, CV_8U, laplacian);

    aia::imshow("sharpening", sharpened, false);

 */

   /*-------------------------------
    *       Laplaciano
    */
/*
    cv::Mat image_laplace;
    cv::Mat laplacian(3, 3, CV_32F);
    laplacian.at<float> (0,0) = 1;
    laplacian.at<float> (0,1) = 1;
    laplacian.at<float> (0,2) = 1;
    laplacian.at<float> (1,0) = 1;
    laplacian.at<float> (1,1) = -8;
    laplacian.at<float> (1,2) = 1;
    laplacian.at<float> (2,0) = 1;
    laplacian.at<float> (2,1) = 1;
    laplacian.at<float> (2,2) = 1;

    cv::filter2D(dstMat, image_laplace, CV_8U, laplacian);

    aia::imshow("laplaciano", image_laplace);
*/

    /*------------------------------
     *      Sobel
     */

    cv::Mat dx, dy, mag;
    cv::Sobel(dstMat, dx, CV_32F, 1, 0);
    cv::Sobel(dstMat, dy, CV_32F, 0, 1);

    cv::magnitude(dx, dy, mag);

    // now we can normalize in [0,255] and convert back to 8U
    cv::normalize(mag, mag, 0, 255, cv::NORM_MINMAX);
    mag.convertTo(mag, CV_8U);

    aia::imshow("sobel", 3*mag, false);

    mag = 3*mag;

    //cv::imwrite("/Users/Mariangela/Desktop/nlm_eq_sob.png", mag);

    cv::threshold(mag, mag, 0, 255, CV_THRESH_BINARY | CV_THRESH_OTSU);
    aia::imshow("threshold", mag);



    return 0;
}