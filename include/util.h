#pragma once
#include <string>
#include <vector>
#include <glm/glm.hpp>
#include <opencv2/core.hpp>

namespace immersight {
	enum CubeFaces
	{
		Back = 4,
		Bottom = 3,
		Front = 5,
		Left = 1,
		Right = 0,
		Top = 2
	};
	struct camera {
		glm::mat4 view;
		glm::mat4 perspective;
		double rms;
	};

	void writeFramebuffer(const std::string file, const int imgWidth, const int imgHeight, const std::vector<glm::vec3> &framebuffer);
	glm::vec3 get3DpositionFromSkybox(const float x, const float y, const float cube_size, const CubeFaces faceID);
	cv::Vec3b readColorFromCamerasAndInterpolate(const std::vector<cv::Mat> &textures, std::vector<cv::Mat> &debug_textures, const std::vector<camera> &cameras, const glm::vec3 position, const float imgWidth, const float imgHeight, const float nNear, const float nFar);
	bool isInFrustum(const glm::vec3 position);
	glm::vec3 viewportTransformation(const glm::vec3 p, const float imgWidth, const float imgHeight, const float nNear, const float nFar);
	glm::vec3 centroid(std::vector<cv::Mat> &mats);
	glm::mat4 CVasGLM(const cv::Mat& mat);
	glm::vec2 circleCoord(float angle);
	cv::Mat rotationX(float angle);
	cv::Mat rotationY(float angle);
	cv::Mat rotationZ(float angle);
	void insertMat(const int row, const int col, const cv::Mat &src, cv::Mat &dst);
	template <typename T>
	T clamp(T min, T max, T val) {
		if (val < min) {
			return min;
		}
		else if (val > max) {
			return max;
		}
		else {
			return val;
		}
	}
}