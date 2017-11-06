#include "../include/util.h"
#include <fstream>
#include <iostream>
#include <glm/ext.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>


namespace immersight {
	void writeFramebuffer(const std::string file, const int imgWidth, const int imgHeight, const std::vector<glm::vec3> &framebuffer) {
		std::ofstream ofs(file, std::ios::out | std::ios::binary);
		ofs << "P6\n" << imgWidth << " " << imgHeight << "\n255\n";
		for (int i = 0; i < (imgHeight * imgWidth); ++i) {
			float tmp = 255.0f * clamp(0.0f, 1.0f, framebuffer[i].r);
			char r = static_cast<char>(255.0f * clamp(0.0f, 1.0f, framebuffer[i].r));
			char g = static_cast<char>(255.0f * clamp(0.0f, 1.0f, framebuffer[i].g));
			char b = static_cast<char>(255.0f * clamp(0.0f, 1.0f, framebuffer[i].b));
			ofs << r << g << b;
		}
		ofs.close();
	}
	glm::vec3 centroid(std::vector<cv::Mat> &mats) {
		glm::vec3 centroid(0.0f);
		for (size_t i = 0; i < mats.size(); i++) {
			centroid.x += mats[i].at<float>(0, 3);
			centroid.y += mats[i].at<float>(1, 3);
			centroid.z += mats[i].at<float>(2, 3);
		}
		centroid /= mats.size();
		return centroid;
	}
	glm::vec3 get3DpositionFromSkybox(const float x, const float y, const float cube_size, const CubeFaces faceID) {
		glm::vec3 result(0);
		const float c = cube_size / 2.0f;
		switch (faceID)
		{
		case Right:
			result = glm::vec3( c, y - c, x - c);
			break;
		case Left:
			result = glm::vec3(-c, y - c, c - x);
			break;
		case Top:
			result = glm::vec3(x - c, c, y - c);
			break;
		case Bottom:
			result = glm::vec3(x - c, -c, c - y);
			break;
		case Back:
			result = glm::vec3(c - x, y - c, c);
			break;
		case Front:
			result = glm::vec3(x - c, y - c, -c);
			break;
		default:
			break;
		}
		return result;
	}
	glm::vec2 circleCoord(float angle) {
		return glm::vec2(glm::cos(angle),glm::sin(angle));
	}
	void insertMat(const int row, const int col, const cv::Mat &src, cv::Mat &dst)
	{
		for (size_t i = 0; i < src.rows; i++)
		{
			for (size_t j = 0; j < src.cols; j++)
			{
				switch (src.type())
				{
				case CV_32F:
					dst.at<float>(row + i, col + j) = src.at<float>(i, j);
					break;
				case CV_64F:
					dst.at<double>(row + i, col + j) = src.at<double>(i, j);
					break;
				case CV_8UC3:
					dst.at<cv::Vec3b>(row + i, col + j) = src.at<cv::Vec3b>(i,j);
					break;
				case CV_8UC4:
					dst.at<cv::Vec4b>(row + i, col + j) = src.at<cv::Vec4b>(i, j);
					break;
				}
			}
		}
	}
	cv::Vec3b readColorFromCamerasAndInterpolate(const std::vector<cv::Mat> &textures, std::vector<cv::Mat> &debug_textures, const std::vector<camera> &cameras, const glm::vec3 position, const float imgWidth, const float imgHeight, const float nNear, const float nFar) {
		glm::vec3 p;
		glm::vec4 ndc, tmp;
		//std::vector<cv::Vec3b> pixelColorValues;
		cv::Vec3b color;
		cv::Point neighbour, atPoint;
		double err = 2.0f;
		
		for (size_t camera = 0; camera < cameras.size(); ++camera) {
			// projection * view * pos;
			ndc = cameras[camera].perspective * cameras[camera].view * glm::vec4(position, 1.0f);
			ndc /= ndc.w;
			p = glm::vec3(ndc);
			if (isInFrustum(p)) {
				if (cameras[camera].rms < err) {
					err = cameras[camera].rms;
					p = viewportTransformation(p, imgWidth, imgHeight, nNear, nFar);
					atPoint = cv::Point(p.x, p.y);
					if (atPoint.x < textures[camera].cols && atPoint.y < textures[camera].rows && atPoint.x >= 0 && atPoint.y >= 0) {
						//pixelColorValues.push_back(textures[camera].at<cv::Vec3b>(atPoint));
						color = textures[camera].at<cv::Vec3b>(atPoint);
						debug_textures[camera].at<cv::Vec3b>(atPoint) = cv::Vec3b(0, 0, 255);
						for (int i = 0; i < 3; i++) {
							for (int j = 0; j < 3; j++) {
								neighbour = cv::Point(i + p.x - 1, j + p.y - 1);
								if (neighbour.x < debug_textures[camera].cols && neighbour.y < debug_textures[camera].rows && neighbour.x >= 0 && neighbour.y >= 0) {
									debug_textures[camera].at<cv::Vec3b>(neighbour) = cv::Vec3b(0, 0, 255);
								}
							}
						}
					}
				}
			}
		}
		/*if (pixelColorValues.size() > 0) {
			float t = 1.0f / pixelColorValues.size();
			for (size_t i = 0; i < pixelColorValues.size(); ++i) {
				color.val[0] += t * pixelColorValues[i].val[0];
				color.val[1] += t * pixelColorValues[i].val[1];
				color.val[2] += t * pixelColorValues[i].val[2];
			}
		}
		else {
			//std::cout << "no camera found for point: (" << position.x << "," << position.y << "," << position.z << ")" << std::endl;
		}*/
		
		return color;
	}
	bool isInFrustum(const glm::vec3 position) {
		/*
		http://www.lighthouse3d.com/tutorials/view-frustum-culling/clip-space-approach-extracting-the-planes/
		*/
		return 
			(position.x >= -1 && position.x <= 1) &&
			(position.y >= -1 && position.y <= 1) &&
			(position.z >= -1 && position.z <= 1);
	}
	glm::vec3 viewportTransformation(const glm::vec3 p, const float imgWidth, const float imgHeight, const float nNear, const float nFar) {
		auto np = 0.5f * p + glm::vec3(0.5f);
		glm::vec3 offset(0.0f, 0.0f, nNear);
		glm::vec3 size(imgWidth, imgHeight, nFar - nNear);
		return offset + np * size;
	}
	glm::mat4 CVasGLM(const cv::Mat& mat) {
		cv::Mat transpose = mat.t();
		const float* data = &mat.at<float>(0, 0); // access the data
		return std::move(glm::make_mat4(data));
	}
	cv::Mat rotationX(float angle) {
		/*
		https://en.wikipedia.org/wiki/Rotation_matrix
		*/
		float cosT = std::cos(angle);
		float sinT = std::sin(angle);
		float data[16] = { 
			1.0f, 0.0f, 0.0f, 0.0f,
			0.0f, cosT,-sinT, 0.0f,
			0.0f, sinT, cosT, 0.0f,
			0.0f, 0.0f, 0.0f, 1.0f
		};
		return cv::Mat(4, 4, CV_32F, &data).clone();
	}
	cv::Mat rotationY(float angle) {
		/*
		https://en.wikipedia.org/wiki/Rotation_matrix
		*/
		float cosT = std::cos(angle);
		float sinT = std::sin(angle);
		float data[16] = {
			cosT, 0.0f, sinT, 0.0f,
			0.0f, 1.0f, 0.0f, 0.0f,
		   -sinT, 0.0f, cosT, 0.0f,
			0.0f, 0.0f, 0.0f, 1.0f
		};
		return cv::Mat(4, 4, CV_32F, &data).clone();
	}
	cv::Mat rotationZ(float angle) {
		/*
		https://en.wikipedia.org/wiki/Rotation_matrix
		*/
		float cosT = std::cos(angle);
		float sinT = std::sin(angle);
		float data[16] = {
			cosT,-sinT, 0.0f, 0.0f,
			sinT, cosT, 0.0f, 0.0f,
			0.0f, 0.0f, 0.0f, 0.0f,
			0.0f, 0.0f, 0.0f, 1.0f
		};
		return cv::Mat(4, 4, CV_32F, &data).clone();
	}
}