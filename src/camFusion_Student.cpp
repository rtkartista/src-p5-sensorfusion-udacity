
#include <iostream>
#include <algorithm>
#include <numeric>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include "camFusion.hpp"
#include "dataStructures.h"

using namespace std;


// Create groups of Lidar points whose projection into the camera falls into the same bounding box
void clusterLidarWithROI(std::vector<BoundingBox> &boundingBoxes, std::vector<LidarPoint> &lidarPoints, float shrinkFactor, cv::Mat &P_rect_xx, cv::Mat &R_rect_xx, cv::Mat &RT)
{
    // loop over all Lidar points and associate them to a 2D bounding box
    cv::Mat X(4, 1, cv::DataType<double>::type);
    cv::Mat Y(3, 1, cv::DataType<double>::type);

    for (auto it1 = lidarPoints.begin(); it1 != lidarPoints.end(); ++it1)
    {
        // assemble vector for matrix-vector-multiplication
        X.at<double>(0, 0) = it1->x;
        X.at<double>(1, 0) = it1->y;
        X.at<double>(2, 0) = it1->z;
        X.at<double>(3, 0) = 1;

        // project Lidar point into camera
        Y = P_rect_xx * R_rect_xx * RT * X;
        cv::Point pt;
        // pixel coordinates
        pt.x = Y.at<double>(0, 0) / Y.at<double>(2, 0); 
        pt.y = Y.at<double>(1, 0) / Y.at<double>(2, 0); 

        vector<vector<BoundingBox>::iterator> enclosingBoxes; // pointers to all bounding boxes which enclose the current Lidar point
        for (vector<BoundingBox>::iterator it2 = boundingBoxes.begin(); it2 != boundingBoxes.end(); ++it2)
        {
            // shrink current bounding box slightly to avoid having too many outlier points around the edges
            cv::Rect smallerBox;
            smallerBox.x = (*it2).roi.x + shrinkFactor * (*it2).roi.width / 2.0;
            smallerBox.y = (*it2).roi.y + shrinkFactor * (*it2).roi.height / 2.0;
            smallerBox.width = (*it2).roi.width * (1 - shrinkFactor);
            smallerBox.height = (*it2).roi.height * (1 - shrinkFactor);

            // check wether point is within current bounding box
            if (smallerBox.contains(pt))
            {
                enclosingBoxes.push_back(it2);
            }

        } // eof loop over all bounding boxes

        // check wether point has been enclosed by one or by multiple boxes
        if (enclosingBoxes.size() == 1)
        { 
            // add Lidar point to bounding box
            enclosingBoxes[0]->lidarPoints.push_back(*it1);
        }

    } // eof loop over all Lidar points
}

/* 
* The show3DObjects() function below can handle different output image sizes, but the text output has been manually tuned to fit the 2000x2000 size. 
* However, you can make this function work for other sizes too.
* For instance, to use a 1000x1000 size, adjusting the text positions by dividing them by 2.
*/
void show3DObjects(std::vector<BoundingBox> &boundingBoxes, cv::Size worldSize, cv::Size imageSize, bool bWait)
{
    // create topview image
    cv::Mat topviewImg(imageSize, CV_8UC3, cv::Scalar(255, 255, 255));

    for(auto it1=boundingBoxes.begin(); it1!=boundingBoxes.end(); ++it1)
    {
        // create randomized color for current 3D object
        cv::RNG rng(it1->boxID);
        cv::Scalar currColor = cv::Scalar(rng.uniform(0,150), rng.uniform(0, 150), rng.uniform(0, 150));

        // plot Lidar points into top view image
        int top=1e8, left=1e8, bottom=0.0, right=0.0; 
        float xwmin=1e8, ywmin=1e8, ywmax=-1e8;
        for (auto it2 = it1->lidarPoints.begin(); it2 != it1->lidarPoints.end(); ++it2)
        {
            // world coordinates
            float xw = (*it2).x; // world position in m with x facing forward from sensor
            float yw = (*it2).y; // world position in m with y facing left from sensor
            xwmin = xwmin<xw ? xwmin : xw;
            ywmin = ywmin<yw ? ywmin : yw;
            ywmax = ywmax>yw ? ywmax : yw;

            // top-view coordinates
            int y = (-xw * imageSize.height / worldSize.height) + imageSize.height;
            int x = (-yw * imageSize.width / worldSize.width) + imageSize.width / 2;

            // find enclosing rectangle
            top = top<y ? top : y;
            left = left<x ? left : x;
            bottom = bottom>y ? bottom : y;
            right = right>x ? right : x;

            // draw individual point
            cv::circle(topviewImg, cv::Point(x, y), 4, currColor, -1);
        }

        // draw enclosing rectangle
        cv::rectangle(topviewImg, cv::Point(left, top), cv::Point(right, bottom),cv::Scalar(0,0,0), 2);

        // augment object with some key data
        char str1[200], str2[200];
        sprintf(str1, "id=%d, #pts=%d", it1->boxID, (int)it1->lidarPoints.size());
        putText(topviewImg, str1, cv::Point2f(left-250, bottom+50), cv::FONT_ITALIC, 2, currColor);
        sprintf(str2, "xmin=%2.2f m, yw=%2.2f m", xwmin, ywmax-ywmin);
        putText(topviewImg, str2, cv::Point2f(left-250, bottom+125), cv::FONT_ITALIC, 2, currColor);  
    }

    // plot distance markers
    float lineSpacing = 2.0; // gap between distance markers
    int nMarkers = floor(worldSize.height / lineSpacing);
    for (size_t i = 0; i < nMarkers; ++i)
    {
        int y = (-(i * lineSpacing) * imageSize.height / worldSize.height) + imageSize.height;
        cv::line(topviewImg, cv::Point(0, y), cv::Point(imageSize.width, y), cv::Scalar(255, 0, 0));
    }

    // display image
    string windowName = "3D Objects";
    cv::namedWindow(windowName, 1);
    cv::imshow(windowName, topviewImg);

    if(bWait)
    {
        cv::waitKey(0); // wait for key to be pressed
    }
}


// associate a given bounding box with the keypoints it contains
void clusterKptMatchesWithROI(BoundingBox &boundingBox, std::vector<cv::KeyPoint> &kptsPrev, std::vector<cv::KeyPoint> &kptsCurr, std::vector<cv::DMatch> &kptMatches)
{
  // Given the input bounding box, we need to make sure that the matches between the previous and current frames
  // are within the bounding box

  // cv::DMatches
  // trainIdx --> current frame
  // queryIdx --> previous frame
  std::vector<cv::KeyPoint> temp_keypoints_prev;
  std::vector<cv::KeyPoint> temp_keypoints_curr;
  std::vector<cv::DMatch> temp_matches;

  // First stage - just check to see if it's within ROI
  // Are there no matches present?
  if(kptMatches.size() == 0)
  { 
    return;
  }
  for (int i = 0; i<kptMatches.size();i++) 
  {
    const int curr_frame_index = kptMatches[i].trainIdx;
    const int prev_frame_index = kptMatches[i].queryIdx;
    const cv::Point2f curr_pt = kptsCurr[curr_frame_index].pt;
    const cv::Point2f prev_pt = kptsPrev[prev_frame_index].pt;
    if (boundingBox.roi.contains(curr_pt) && boundingBox.roi.contains(prev_pt)) 
    {
      temp_keypoints_prev.push_back(kptsCurr[curr_frame_index]);
      temp_keypoints_curr.push_back(kptsPrev[curr_frame_index]);
      temp_matches.push_back(kptMatches[i]);
    }
    
  }

  // Find distances between the keypoint matches to get the mean distance
  std::vector<float> distances;
  double ref_scale;
  if(temp_keypoints_prev.size() == 0)
  {
    return;
  }

  for (size_t i = 0; i < temp_keypoints_prev.size(); i++) 
  {
    float k = cv::norm(temp_keypoints_curr[i].pt - temp_keypoints_prev[i].pt);
    distances.push_back(k);
    ref_scale += k;
  }
  ref_scale = 1.5*(ref_scale / temp_keypoints_prev.size());
  // Finally, go through each match and make sure that they aren't outliers
  for (size_t i = 0; i < temp_keypoints_prev.size(); i++)
  {
		if (!boundingBox.roi.contains(temp_keypoints_curr[i].pt)) { continue; }

    if (cv::norm(temp_keypoints_prev[i].pt - temp_keypoints_curr[i].pt) < ref_scale) 
    {
      boundingBox.keypoints.push_back(temp_keypoints_curr[i]);
      boundingBox.kptMatches.push_back(temp_matches[i]);
    }
	}
  return;
}


// Compute time-to-collision (TTC) based on keypoint correspondences in successive images
void computeTTCCamera(std::vector<cv::KeyPoint> &kptsPrev, std::vector<cv::KeyPoint> &kptsCurr, 
                      std::vector<cv::DMatch> kptMatches, double frameRate, double &TTC, cv::Mat *visImg)
{
        // compute distance ratios between all matched keypoints
    vector<double> distRatios; // stores the distance ratios for all keypoints between curr. and prev. frame
    for (auto it1 = kptMatches.begin(); it1 != kptMatches.end() - 1; ++it1)
    { // outer kpt. loop

        // get current keypoint and its matched partner in the prev. frame
        cv::KeyPoint kpOuterCurr = kptsCurr.at(it1->trainIdx);
        cv::KeyPoint kpOuterPrev = kptsPrev.at(it1->queryIdx);

        for (auto it2 = kptMatches.begin() + 1; it2 != kptMatches.end(); ++it2)
        { // inner kpt.-loop

            double minDist = 100.0; // min. required distance

            // get next keypoint and its matched partner in the prev. frame
            cv::KeyPoint kpInnerCurr = kptsCurr.at(it2->trainIdx);
            cv::KeyPoint kpInnerPrev = kptsPrev.at(it2->queryIdx);

            // compute distances and distance ratios
            double distCurr = cv::norm(kpOuterCurr.pt - kpInnerCurr.pt);
            double distPrev = cv::norm(kpOuterPrev.pt - kpInnerPrev.pt);

            if (distPrev > std::numeric_limits<double>::epsilon() && distCurr >= minDist)
            { // avoid division by zero

                double distRatio = distCurr / distPrev;
                distRatios.push_back(distRatio);
            }
        } // eof inner loop over all matched kpts
    }     // eof outer loop over all matched kpts
    // only continue if list of distance ratios is not empty
    if (distRatios.size() == 0)
    {
        TTC = NAN;
        return;
    }


    // STUDENT TASK (replacement for meanDistRatio)
    std::sort(distRatios.begin(), distRatios.end());
    long medIndex = floor(distRatios.size() / 2.0);
    double medDistRatio = distRatios.size() % 2 == 0 ? (distRatios[medIndex - 1] + distRatios[medIndex]) / 2.0 : distRatios[medIndex]; // compute median dist. ratio to remove outlier influence

    double dT = 1 / frameRate;
    TTC = -dT / (1 - medDistRatio);
    // EOF STUDENT TASK
}

void isOutliner(double)
{

}

void computeTTCLidar(std::vector<LidarPoint> &lidarPointsPrev,
                     std::vector<LidarPoint> &lidarPointsCurr, double frameRate, double &TTC)
{
    // assignments
    // time between two measurements in seconds
    // find closest distance to Lidar points

    // check if the points are outliners
    // finding the mean X point
    double meanXP = 0, meanXC = 0;
    for(auto it=lidarPointsPrev.begin(); it!=lidarPointsPrev.end(); ++it) 
    {
      meanXP += it->x;
    }
    meanXP = meanXP / lidarPointsPrev.size();

    for(auto it=lidarPointsCurr.begin(); it!=lidarPointsCurr.end(); ++it) 
    {
      meanXC += it->x;
    }
    meanXC = meanXC / lidarPointsCurr.size();

    // finding the stdX
    double stdP = 0, stdC = 0;
    for(auto it=lidarPointsPrev.begin(); it!=lidarPointsPrev.end(); ++it) 
    {
      stdP += pow(it->x - meanXP , 2);
    }
    stdP = sqrt(stdP / lidarPointsPrev.size());

    for(auto it=lidarPointsCurr.begin(); it!=lidarPointsCurr.end(); ++it) 
    {
      stdC += pow(it->x - meanXC , 2);
    }
    stdC = sqrt(stdC / lidarPointsCurr.size());
    
    // removing outlier - which are 3*std away from mean
    // compute TTC from both measurements
    double minXPrev = 1e9, minXCurr = 1e9;
    for(auto it=lidarPointsPrev.begin(); it!=lidarPointsPrev.end(); ++it) 
    {
      if(it->x <= (meanXP + (3 * stdP)) && it->x >= ( meanXP - (3 * stdP)))
      {
        minXPrev = minXPrev>it->x ? it->x : minXPrev;
      }
    }

    for(auto it=lidarPointsCurr.begin(); it!=lidarPointsCurr.end(); ++it) 
    {
      if(it->x <= (meanXC + (3 * stdC)) && it->x >= ( meanXC - (3 * stdC)))
      {
        minXCurr = minXCurr>it->x ? it->x : minXCurr;
      }
    }
    double dt = 1/frameRate;
    TTC = (minXCurr * dt) / abs(minXPrev-minXCurr);
}


void matchBoundingBoxes(std::vector<cv::DMatch> &matches, std::map<int, int> &bbBestMatches, DataFrame &prevFrame, DataFrame &currFrame)
{
  // trainIdx --> current frame
  // queryIdx --> previous frame
  // Goal - Figure out which bounding box ID from the previous frame matches the current frame
  //        given that most of the keypoints overlap between the two bounding boxes
  
  // map element consisting of {{bbox_prev, bbox_curr}, counts}
  // bbox_prev = bounding box id in the previous frame
  // bbox_curr = bounding box id in the cuurent frame
  // counts = no of common keypoint matches between the boxes
  std::map<pair<int, int>, int> map_count;
  // it->first, it->first.first, it->first.second
  // it->second

  // STEP 1 - Loop over each match in the matches vector
  // Find the corresponding box matches for above matched keypoints. 
  for(const auto& dm : matches)
  {
    int kpCurrId = dm.trainIdx;
    int kpPrevId = dm.queryIdx;
    cv::Point2f curr_pt = currFrame.keypoints[kpCurrId].pt;
    cv::Point2f prev_pt = prevFrame.keypoints[kpPrevId].pt;

    for (const BoundingBox& prev_bbox : prevFrame.boundingBoxes) 
    {
      for (const BoundingBox& curr_bbox : currFrame.boundingBoxes) 
      {
        // If both keypoints of the match are in the boundingBoxes
        if (prev_bbox.roi.contains(prev_pt) && curr_bbox.roi.contains(curr_pt)) 
        {
          // new box entry
          const int prev_bbox_id = prev_bbox.boxID;
          const int curr_bbox_id = curr_bbox.boxID;

          std::map<std::pair<int, int>, int>::iterator it;
          it = map_count.find(std::make_pair(prev_bbox_id, curr_bbox_id));
          if (it == map_count.end()) 
          {
            map_count.insert(std::pair<std::pair<int, int>, int>(std::make_pair(prev_bbox_id, curr_bbox_id), 1));
          } 
          // repetitive box pair
          else 
          {
            const int count = it->second;
            it->second = count + 1;
          }
        }
      }
    }
  }
  // STEP 2 - loop over all possible bounding box IDs from the previous frame
  // see how many combinations of matches to the current frame we have
  // then choose the largest one
  for(const BoundingBox& prev_bbox : prevFrame.boundingBoxes)
  {
    BoundingBox current_bbox;
    int max_counts = 1; 
    for (std::map<pair<int, int>, int>::iterator it=map_count.begin(); it!=map_count.end(); ++it )
    { 
      if(it->first.first == prev_bbox.boxID)
      {
        if(it->second >= max_counts )
        {
          max_counts = it->second;
          current_bbox.boxID = it->first.second;
        }
      }
    }

    bbBestMatches.insert(std::pair<int, int>(prev_bbox.boxID, current_bbox.boxID));
  }
}