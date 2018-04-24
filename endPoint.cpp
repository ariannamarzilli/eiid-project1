// include aia and ucas utility functions
#include "aiaConfig.h"
#include "ucasConfig.h"

#include "functions.h"
#include <list>
namespace
{
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
}

//prende in input l'immagine, restituisce una lista di Point, ogni Point rappresenta un end point
std::list<cv::Point> endPoint(cv::Mat mat){

	cv::Mat img = mat.clone();
	std::list<cv::Point> endPointList;

	try
	{	
		ucas::imshow("original image", img);

		// STEP 1: SKELETONIZATION
		// to perform skeletonization, we use two 'edge'-like SEs
		// along with their rotated versions
		std::vector<cv::Mat> skel_SEs;
		skel_SEs.push_back((cv::Mat_<char>(3,3) << 
			-1, -1, -1, 
			 0,  1,  0,
			 1,  1,  1 ));
		skel_SEs.push_back((cv::Mat_<char>(3,3) << 
			 0, -1, -1, 
			 1,  1, -1,
			 1,  1,  0 ));

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
			cv::imshow("skeletonization", skeleton);
			if (cv::waitKey(200)>=0)
				cv::destroyWindow("skeletonization");
		}
		while (cv::countNonZero(skeleton_prev - skeleton) > 0);	// convergence = no more changes
		ucas::imshow("skeleton", skeleton);
		cv::imwrite("C:/work/skeleton.png", skeleton);

		// STEP 2: PRUNING (to remove spurious junctions generated by skeletonization)
		// to perform pruning, we use an 'endpoint'-like SE
		// along with its rotated versions
		std::vector <cv::Mat> prun_SEs;
		prun_SEs.push_back((cv::Mat_<char>(3,3) << 
			 0,  0,  0, 
			-1,  1, -1,
			-1, -1, -1 ));

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
			cv::imshow("pruning", pruned);
			if (cv::waitKey(200)>=0)
				cv::destroyWindow("pruning");
		}
		ucas::imshow("pruned", pruned);

		// STEP 3: detection of endpoints
		std::vector <cv::Mat> jun_SEs;
		jun_SEs.push_back((cv::Mat_<char>(3,3) << 
			 -1, -1,  -1, 
			 -1,  1,  1,
			-1,  -1, -1 ));

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
		ucas::imshow("endpoints", junctions);


		// overlay endpoints as red circles on the original image
		// endpoints are single pixels in the 'endpoints' image
		// we use dilation with a circle-like SE
		cv::dilate(junctions, junctions, cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(7,7)));
		cv::cvtColor(img, img, cv::COLOR_GRAY2BGR);
		img.setTo(cv::Scalar(0, 0, 255), junctions);
		ucas::imshow("result", img);

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


int main() 
{
	try
	{	
		// load the binary vessel tree image
		cv::Mat img = cv::imread("C:/Users/giorgio/Desktop/1.tif", CV_LOAD_IMAGE_GRAYSCALE);
		if(!img.data)
			throw ucas::Error("cannot load image");
		std::list<cv::Point> l = endPoint(img);
		for (cv::Point p : l){
			printf("endpoint x:%d y:%d\n",p.x, p.y);
		}

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
