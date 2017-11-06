#pragma once

#include <opencv2/core.hpp>
#include <opencv2/xfeatures2d.hpp>
#include <opencv2/ccalib/randpattern.hpp>

#include <vector>

namespace immersight {
	void getPatternDescriptors(const cv::Mat& image, const cv::Ptr<cv::FeatureDetector>& detector, const  cv::Ptr<cv::DescriptorExtractor>& descriptor, std::vector<cv::KeyPoint>& outKeypoints, cv::Mat& outDescriptor);
	void crossCheckMatching(const cv::Ptr<cv::DescriptorMatcher>& descriptorMatcher, const cv::Mat& descriptors1, const cv::Mat& descriptors2, std::vector<cv::DMatch>& filteredMatches12, int knn);
	void keyPoints2MatchedLocation(const std::vector<cv::KeyPoint>& imageKeypoints, const std::vector<cv::KeyPoint>& patternKeypoints, const std::vector<cv::DMatch> matchces, cv::Mat& matchedImagelocation, cv::Mat& matchedPatternLocation);
	void getFilteredLocation(cv::Mat& imageKeypoints, cv::Mat& patternKeypoints, const cv::Mat mask);
	void drawCorrespondence(const std::string image_url, const cv::Mat& image1, const std::vector<cv::KeyPoint> keypoint1, const cv::Mat& image2, const std::vector<cv::KeyPoint> keypoint2, const std::vector<cv::DMatch> matchces, const cv::Mat& mask1, const cv::Mat& mask2, const int step);
	void drawMatchingForImage(const std::string image_url, cv::Mat& inputImage, const cv::Mat& patternImage, const cv::Ptr<cv::FeatureDetector>& detector, const cv::Ptr<cv::DescriptorMatcher> matcher, const  cv::Ptr<cv::DescriptorExtractor>& descriptor, const std::vector<cv::KeyPoint>& patternKeypoints, const cv::Mat& patternDescriptor, const cv::Size patternImageSize, const cv::Size2f patternSize, int depth = CV_32F);
} // end immersight