#include "../include/opencv.hpp"
#include "../include/tools.hpp"
#include <opencv2/imgproc.hpp>
#include <opencv2/calib3d.hpp>

namespace immersight {
	void getPatternDescriptors(const cv::Mat& image, const cv::Ptr<cv::FeatureDetector>& detector, const  cv::Ptr<cv::DescriptorExtractor>& descriptor, std::vector<cv::KeyPoint>& outKeypoints, cv::Mat& outDescriptor) {
		cv::Mat tmpImg;
		image.copyTo(tmpImg);
		if (tmpImg.type() != CV_8U)
			tmpImg.convertTo(tmpImg, CV_8U);
		cv::Size imageSize = tmpImg.size();
		detector->detect(tmpImg, outKeypoints);
		descriptor->compute(tmpImg, outKeypoints, outDescriptor);
		outDescriptor.convertTo(outDescriptor, CV_32F);
	}
	void crossCheckMatching(const cv::Ptr<cv::DescriptorMatcher>& descriptorMatcher, const cv::Mat& descriptors1, const cv::Mat& descriptors2, std::vector<cv::DMatch>& filteredMatches12, int knn)	{
		filteredMatches12.clear();
		std::vector<std::vector<cv::DMatch> > matches12, matches21;
		descriptorMatcher->knnMatch(descriptors1, descriptors2, matches12, knn);
		descriptorMatcher->knnMatch(descriptors2, descriptors1, matches21, knn);
		for (size_t m = 0; m < matches12.size(); m++) {
			bool findCrossCheck = false;
			for (size_t fk = 0; fk < matches12[m].size(); fk++)	{
				cv::DMatch forward = matches12[m][fk];

				for (size_t bk = 0; bk < matches21[forward.trainIdx].size(); bk++) {
					cv::DMatch backward = matches21[forward.trainIdx][bk];
					if (backward.trainIdx == forward.queryIdx) {
						filteredMatches12.push_back(forward);
						findCrossCheck = true;
						break;
					}
				}
				if (findCrossCheck) break;
			}
		}
	}
	void keyPoints2MatchedLocation(const std::vector<cv::KeyPoint>& imageKeypoints, const std::vector<cv::KeyPoint>& patternKeypoints, const std::vector<cv::DMatch> matchces, cv::Mat& matchedImagelocation, cv::Mat& matchedPatternLocation)
	{
		matchedImagelocation.release();
		matchedPatternLocation.release();
		std::vector<cv::Vec2d> image, pattern;
		for (int i = 0; i < (int)matchces.size(); ++i)
		{
			cv::Point2f imgPt = imageKeypoints[matchces[i].queryIdx].pt;
			cv::Point2f patPt = patternKeypoints[matchces[i].trainIdx].pt;
			image.push_back(cv::Vec2d(imgPt.x, imgPt.y));
			pattern.push_back(cv::Vec2d(patPt.x, patPt.y));
		}
		cv::Mat(image).convertTo(matchedImagelocation, CV_64FC2);
		cv::Mat(pattern).convertTo(matchedPatternLocation, CV_64FC2);
	}
	void getFilteredLocation(cv::Mat& imageKeypoints, cv::Mat& patternKeypoints, const cv::Mat mask)
	{
		cv::Mat tmpKeypoint, tmpPattern;
		imageKeypoints.copyTo(tmpKeypoint);
		patternKeypoints.copyTo(tmpPattern);
		imageKeypoints.release();
		patternKeypoints.release();
		std::vector<cv::Vec2d> vecKeypoint, vecPattern;
		for (int i = 0; i < (int)mask.total(); ++i)
		{
			if (mask.at<uchar>(i) == 1)
			{
				vecKeypoint.push_back(tmpKeypoint.at<cv::Vec2d>(i));
				vecPattern.push_back(tmpPattern.at<cv::Vec2d>(i));
			}
		}
		cv::Mat(vecKeypoint).convertTo(imageKeypoints, CV_64FC2);
		cv::Mat(vecPattern).convertTo(patternKeypoints, CV_64FC2);
	}
	void drawCorrespondence(const std::string image_url, const cv::Mat& image1, const std::vector<cv::KeyPoint> keypoint1, const cv::Mat& image2, const std::vector<cv::KeyPoint> keypoint2, const std::vector<cv::DMatch> matchces, const cv::Mat& mask1, const cv::Mat& mask2, const int step)
	{
		cv::Mat img_corr, img_key, patt_key;
		if (step == 1) {
			drawMatches(image1, keypoint1, image2, keypoint2, matchces, img_corr);
		}
		else if (step == 2) {
			std::vector<cv::DMatch> matchesFilter;
			for (int i = 0; i < (int)mask1.total(); ++i) {
				if (!mask1.empty() && mask1.at<uchar>(i) == 1) {
					matchesFilter.push_back(matchces[i]);
				}
			}
			drawMatches(image1, keypoint1, image2, keypoint2, matchesFilter, img_corr);
		}
		else if (step == 3) {
			std::vector<cv::DMatch> matchesFilter;
			int j = 0;
			for (int i = 0; i < (int)mask1.total(); ++i) {
				if (mask1.at<uchar>(i) == 1) {
					if (!mask2.empty() && mask2.at<uchar>(j) == 1) {
						matchesFilter.push_back(matchces[i]);
					}
					j++;
				}
			}
			drawMatches(image1, keypoint1, image2, keypoint2, matchesFilter, img_corr);
			drawKeypoints(image1, keypoint1, patt_key);
			drawKeypoints(image2, keypoint2, img_key);
			imwrite(immersight::insert_back(image_url, 4, "-pattern-features"), patt_key);
			imwrite(immersight::insert_back(image_url, 4, "-image-features"), img_key);
		}
		imwrite(immersight::insert_back(image_url, 4, "-matches-" + std::to_string(step)), img_corr);
	}

	void drawMatchingForImage(const std::string image_url, cv::Mat& inputImage, const cv::Mat& patternImage, const cv::Ptr<cv::FeatureDetector>& detector, const cv::Ptr<cv::DescriptorMatcher> matcher, const  cv::Ptr<cv::DescriptorExtractor>& descriptor, const std::vector<cv::KeyPoint>& patternKeypoints, const cv::Mat& patternDescriptor, const cv::Size patternImageSize, const cv::Size2f patternSize, int depth)
	{
		std::vector<cv::Mat> r(2);
		cv::Mat image, descriptorImage1, descriptorImage2, descriptorImage, imageEquHist;
		std::vector<cv::KeyPoint> keypointsImage1, keypointsImage2, keypointsImage;
		inputImage.copyTo(image);
		if (image.type() != CV_8U)
		{
			image.convertTo(image, CV_8U);
			cv::cvtColor(image, image, CV_BGR2GRAY);
		}

		cv::equalizeHist(image, imageEquHist);

		detector->detect(image, keypointsImage1);
		descriptor->compute(image, keypointsImage1, descriptorImage1);
		detector->detect(imageEquHist, keypointsImage2);
		descriptor->compute(imageEquHist, keypointsImage2, descriptorImage2);
		descriptorImage1.convertTo(descriptorImage1, CV_32F);
		descriptorImage2.convertTo(descriptorImage2, CV_32F);

		// match with pattern
		std::vector<cv::DMatch> matchesImgtoPat, matchesImgtoPat1, matchesImgtoPat2;

		cv::Mat keypointsImageLocation, keypointsPatternLocation;

		crossCheckMatching(matcher, descriptorImage1, patternDescriptor, matchesImgtoPat1, 1);
		crossCheckMatching(matcher, descriptorImage2, patternDescriptor, matchesImgtoPat2, 1);
		if ((int)matchesImgtoPat1.size() > (int)matchesImgtoPat2.size())
		{
			matchesImgtoPat = matchesImgtoPat1;
			keypointsImage = keypointsImage1;
		}
		else
		{
			matchesImgtoPat = matchesImgtoPat2;
			keypointsImage = keypointsImage2;
		}

		keyPoints2MatchedLocation(keypointsImage, patternKeypoints, matchesImgtoPat, keypointsImageLocation, keypointsPatternLocation);

		cv::Mat img_corr;

		// innerMask is CV_8U type
		cv::Mat innerMask1, innerMask2;

		// draw raw correspondence
		drawCorrespondence(image_url, inputImage, keypointsImage, patternImage, patternKeypoints, matchesImgtoPat, innerMask1, innerMask2, 1);



		// outlier remove
		cv::findFundamentalMat(keypointsImageLocation, keypointsPatternLocation, cv::FM_RANSAC, 1, 0.995, innerMask1);
		getFilteredLocation(keypointsImageLocation, keypointsPatternLocation, innerMask1);


		drawCorrespondence(image_url, inputImage, keypointsImage, patternImage, patternKeypoints, matchesImgtoPat, innerMask1, innerMask2, 2);

		findHomography(keypointsImageLocation, keypointsPatternLocation, cv::RANSAC, 30 * inputImage.cols / 1000, innerMask2);
		getFilteredLocation(keypointsImageLocation, keypointsPatternLocation, innerMask2);


		drawCorrespondence(image_url, inputImage, keypointsImage, patternImage, patternKeypoints, matchesImgtoPat, innerMask1, innerMask2, 3);
	}
} // immersight