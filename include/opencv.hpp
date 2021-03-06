#pragma once

#include <opencv2/core.hpp>
#include <opencv2/xfeatures2d.hpp>
#include <opencv2/ccalib/randpattern.hpp>
#include "defines.hpp"
#include <vector>
#include <opencv2/xfeatures2d/cuda.hpp>
#include <opencv2/cudafeatures2d.hpp>

namespace immersight {
    struct ProcessedImage
    {
        cv::Mat image;
        cv::Mat descriptor;
        std::vector<cv::KeyPoint> keypoints;
    };
	void getPatternDescriptors(const cv::Mat& image, const cv::Ptr<cv::FeatureDetector>& detector, const  cv::Ptr<cv::DescriptorExtractor>& descriptor, std::vector<cv::KeyPoint>& outKeypoints, cv::Mat& outDescriptor);
	void crossCheckMatching(const cv::Ptr<cv::DescriptorMatcher>& descriptorMatcher, const cv::Mat& descriptors1, const cv::Mat& descriptors2, std::vector<cv::DMatch>& filteredMatches12, int knn);
	void keyPoints2MatchedLocation(const std::vector<cv::KeyPoint>& imageKeypoints, const std::vector<cv::KeyPoint>& patternKeypoints, const std::vector<cv::DMatch> matchces, cv::Mat& matchedImagelocation, cv::Mat& matchedPatternLocation);
	void getFilteredLocation(cv::Mat& imageKeypoints, cv::Mat& patternKeypoints, const cv::Mat mask);
	void drawCorrespondence(const std::string imageUrl, const cv::Mat& image1, const std::vector<cv::KeyPoint> keypoint1, const cv::Mat& image2, const std::vector<cv::KeyPoint> keypoint2, const std::vector<cv::DMatch> matchces, const cv::Mat& mask1, const cv::Mat& mask2, const int step);
	void drawMatchingForImage(const std::string imageUrl, cv::Mat& inputImage, const cv::Mat& patternImage, const cv::Ptr<cv::FeatureDetector>& detector, const cv::Ptr<cv::DescriptorMatcher> matcher, const  cv::Ptr<cv::DescriptorExtractor>& descriptor, const std::vector<cv::KeyPoint>& patternKeypoints, const cv::Mat& patternDescriptor, const cv::Size patternImageSize, const cv::Size2f patternSize, int depth = CV_32F);
    std::vector<cv::DMatch> computeMatchingFeatures(const ProcessedImage &pImage, const ProcessedImage &pRef, cv::Ptr<cv::FeatureDetector> detector, cv::Ptr<cv::DescriptorExtractor> descriptor, cv::Ptr<cv::DescriptorMatcher> matcher);
    void processImage(const cv::Mat& image, ProcessedImage& pImage, const cv::Ptr<cv::FeatureDetector> detector, const cv::Ptr<cv::DescriptorExtractor> descriptor);
    void convertVideoToImageSequence(const std::string& video, const int frames = 1);
    void filterMatches(const int cols, std::vector<cv::KeyPoint>& kA, std::vector<cv::KeyPoint>& kB, std::vector<cv::DMatch>& matches);
    namespace cuda_gpu
    {
        struct GpuImage
        {
            cv::Mat color;
            cv::cuda::GpuMat image;
            cv::cuda::GpuMat descriptor_gpu;
            cv::cuda::GpuMat keypoints_gpu;
            std::vector<cv::KeyPoint> keypoints;
            std::vector<float> descriptor;
        };
        void computeMatches(const cv::Mat& imgA, const cv::Mat& imgB, std::vector<cv::DMatch>& matches, cv::cuda::SURF_CUDA& surf, const cv::Ptr<cv::cuda::DescriptorMatcher> matcher);
        void processImage_CUDA(const cv::cuda::GpuMat& image, GpuImage& pImage, cv::cuda::SURF_CUDA& detector);
        void detectAndComputeSURF_CUDA(const std::string& imagePath, std::vector<cv::Mat>& images, std::vector<cv::Mat>& keypoints, std::vector<cv::Mat>& descriptors, const bool video = false);
        void crossCheckMatching_CUDA(const cv::Ptr<cv::cuda::DescriptorMatcher>& descriptorMatcher, const cv::cuda::GpuMat& descriptors1, const cv::cuda::GpuMat& descriptors2, std::vector<cv::DMatch>& filteredMatches12, int knn);
        void computeMatchingFeatures_CUDA(const GpuImage& pImage, const GpuImage& pRef, const cv::Ptr<cv::cuda::DescriptorMatcher> matcher, std::vector<cv::DMatch>& matches);
        void processImagesFromVideo_CUDA(const std::string& video, std::vector<GpuImage>& images, const int frames = 1);
        void processImagesFromImageSequence_CUDA(const std::string& sequence, std::vector<GpuImage>& images);
        void processImages_CUDA(const std::vector<std::string>& list, std::vector<GpuImage>& images);
    }
} // end immersight