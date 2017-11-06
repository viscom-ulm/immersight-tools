// tools.cpp
#include "../include/tools.hpp"
#include "../include/defines.hpp"

#include <opencv2/imgproc.hpp>
#include <opencv2/calib3d.hpp>

#include <iostream>
#include <string>
#include <sstream>
#include <algorithm>
#include <iterator>

namespace immersight {
	/*
	write multicalibration data to xml
	*/
	void writeMulticalibrationParameter(std::string filename, calibIO& data) {
		cv::FileStorage fs(filename, cv::FileStorage::WRITE);

		fs << "nCameras" << data.nCamera;

		for (int camIdx = 0; camIdx < data.nCamera; ++camIdx)
		{
			char num[10];
			sprintf(num, "%d", camIdx);
			std::string cameraMatrix = "camera_matrix_" + std::string(num);
			std::string cameraPose = "camera_pose_" + std::string(num);
			std::string cameraDistortion = "camera_distortion_" + std::string(num);
			std::string cameraId = "camera_id_" + std::string(num);
			std::string cameraXi = "xi_" + std::string(num);
			std::string image = "image_" + std::string(num);

			fs << cameraMatrix << data.cameraMatrix[camIdx];
			fs << cameraDistortion << data.distortCoeff[camIdx];
			fs << cameraPose << data.cameraPoses[camIdx];
			fs << cameraId << data.cameraId[camIdx];
			for (int i = 0; i < data.availableTimestamps[camIdx].size(); i++) {
				fs << image +"_"+std::to_string(i) << data.availableTimestamps[camIdx][i];
			}
		}

		fs << "meanReprojectError" << data.rms;
		for (std::map<int, cv::Mat>::iterator it = data.patternPoses.begin(); it != data.patternPoses.end(); it++)
		{
			char timestamp[100];
			sprintf(timestamp, "%d", it->first);
			std::string photoTimestamp = "pose_timestamp_" + std::string(timestamp);

			fs << photoTimestamp << it->second;
		}
		fs << "nPattern" << (int)data.patternPoses.size();
	}
	/*
	read multicalibration data to xml
	*/
	void readMulticalibrationParameter(std::string filename, calibIO& data) {
		cv::FileStorage fs(filename, cv::FileStorage::READ);
		int n;
		fs["nCameras"] >> n;
		data.nCamera = n;
		data = calibIO(n);

		for (int camIdx = 0; camIdx < data.nCamera; ++camIdx)
		{
			char num[10];
			sprintf(num, "%d", camIdx);
			std::string cameraMatrix = "camera_matrix_" + std::string(num);
			std::string cameraPose = "camera_pose_" + std::string(num);
			std::string cameraDistortion = "camera_distortion_" + std::string(num);
			std::string cameraId = "camera_id_" + std::string(num);
			std::string cameraXi = "xi_" + std::string(num);
			std::string image = "image_" + std::string(num);
			std::string prePose = "prePose_" + std::string(num);

			fs[cameraMatrix] >> data.cameraMatrix[camIdx];
			fs[cameraDistortion] >> data.distortCoeff[camIdx];
			fs[cameraPose] >> data.cameraPoses[camIdx];
			fs[cameraId] >> data.cameraId[camIdx];
			fs[prePose] >> data.prePose[camIdx];
			int i = 0;
			cv::FileNode s;
			do {
				s = fs[image +"_"+std::to_string(i++)];
				if (!s.empty()) {
					data.availableTimestamps[camIdx].push_back(s);
				}
			} while (!s.empty());
		}

		fs["meanReprojectError"] >> data.rms;
		cv::Mat patternPose;
		int i = 0;
		int nPattern;
		fs["nPattern"] >> nPattern;
		while (data.patternPoses.size() < nPattern) {
			patternPose = cv::Mat();
			fs["pose_timestamp_" + std::to_string(i)] >> patternPose;

			if (patternPose.rows <= 0 || patternPose.cols <= 0) {
				i++;
				continue;
			}
			data.patternPoses[i] = patternPose;
			i++;
		}
	}
	/*
	angle degree to radian in float
	*/
	float deg2rad(float deg) {
		return float(deg * M_PI) / 180.f;
	}
	/*
	angle degree to radian in double
	*/
	double deg2rad(double deg) {
		return double(deg * M_PI) / 180.0;
	}
	/*
	random number generator returns a number between 0 and max
	*/
	double nrand(double max) { return ((double)rand() / (double)RAND_MAX)*max; }
	int nrand(int max) { return (int)(((double)rand() / (double)RAND_MAX))*max; }
	/*
	returns a pre-configured SURF descriptor
	*/
	cv::Ptr<cv::xfeatures2d::SURF> getSURF() {
		int metrixThreshold = 200; //minHessian
		int numberOfOctaves = 8;
		int numScaleLevels = 4; // nOctavesLayer
		return cv::xfeatures2d::SURF::create(metrixThreshold, numberOfOctaves, numScaleLevels);
	}
	/*
	returns a Flann based descriptor matcher
	*/
	cv::Ptr<cv::DescriptorMatcher> getFLANN() {
		return cv::DescriptorMatcher::create(cv::DescriptorMatcher::FLANNBASED);
	}
	/*
	returns a pre-configured PatternFinder
	*/
	cv::randpattern::RandomPatternCornerFinder getRandomPatternFinder(float patternWidth, float patternHeight, int minMatches, bool verbose, bool showExtraction) {
		return cv::randpattern::RandomPatternCornerFinder(patternWidth,
			patternHeight,
			minMatches,
			CV_32F,
			verbose,
			showExtraction,
			getSURF(),
			getSURF(),
			cv::DescriptorMatcher::create(cv::DescriptorMatcher::FLANNBASED));
	}
	std::string cameraIndex2string(int index) {
		if (index < 9) {
			return "0" + std::to_string(index+1);
		}
		else {
			return std::to_string(index+1);
		}
	}
	/*
	reads file and returns string list of files
	*/
	VecStr readStringList(std::string file)
	{
		VecStr l;
		l.resize(0);
		cv::FileStorage fs(file, cv::FileStorage::READ);
		if (fs.isOpened()) {
			cv::FileNode n = fs.getFirstTopLevelNode();

			cv::FileNodeIterator it = n.begin(), it_end = n.end();
			for (; it != it_end; ++it) {
				l.push_back((std::string)*it);
			}
		}
		return l;
	}
	int getTimestampIndex(int timestamp, VecInt& timestamps) {
		int index = 0;
		std::for_each(timestamps.begin(), timestamps.end(), [&](int t) {if (timestamp == t) { return; }index++; });
		return index;
	}
	void parseCamAndTimestampSingle(const std::string& string, int *cam, int *timestamp) {
		std::string filename = string.substr(0, string.find('.'));
		size_t spritPosition1 = filename.rfind('/');
		size_t spritPosition2 = filename.rfind('\\');
		if (spritPosition1 != std::string::npos)
		{
			filename = filename.substr(spritPosition1 + 1, filename.size() - 1);
		}
		else if (spritPosition2 != std::string::npos)
		{
			filename = filename.substr(spritPosition2 + 1, filename.size() - 1);
		}
		sscanf(filename.c_str(), "%d-%d", cam, timestamp);
	}
	/*
	for the given file list and number of camera, two lists containing cam ids and timestamps are returned
	*/
	void parseCamAndTimestamp(const int nCamera, const VecStr& file_list, VecStr2D& filesEachCameraFull, VecInt2D& timestampFull) {
		filesEachCameraFull.resize(nCamera);
		timestampFull.resize(nCamera);
		for (int i = 0; i < (int)file_list.size(); ++i)
		{
			int cameraVertex, timestamp;
			parseCamAndTimestampSingle(file_list[i],&cameraVertex,&timestamp);
			filesEachCameraFull[cameraVertex].push_back(file_list[i]);
			timestampFull[cameraVertex].push_back(timestamp);
		}
	}
	/*
	takes a list of images and matches with pattern
	*/
	void match(VecStr2D& image_list, VecStr& skipped_files, VecStr2D& kept_files, int nCamera, int minMatches, float scale, cv::randpattern::RandomPatternCornerFinder& matcher, VecMat2D& outImagePoints, VecMat2D& outObjPoints, bool verbose) {
		INFO("Begin to match images from image list with pattern");
		outImagePoints.resize(nCamera);
		outObjPoints.resize(nCamera);
		kept_files.resize(nCamera);
		// calibrate each camera individually
		for (int camera = 0; camera < nCamera; camera++) {
			if (verbose) {
				INFO("Start with images from camera " + std::to_string(camera));
			}
			cv::Mat image, cameraMatrix, distortCoeffs;
			// find image and object points
			for (int imgIdx = 0; imgIdx < (int)image_list[camera].size(); ++imgIdx)
			{
				if (verbose) {
					INFO("Reading image " + image_list[camera][imgIdx]);
				}
				image = cv::imread(image_list[camera][imgIdx], cv::IMREAD_GRAYSCALE);
				if (image.empty()) {
					if (verbose) {
						WARN("file " + image_list[camera][imgIdx] + " is empty");
					}
					skipped_files.push_back(image_list[camera][imgIdx]);
					continue;
				}
				cv::resize(image, image, cv::Size(), scale, scale);
				VecMat imgObj = matcher.computeObjectImagePointsForSingle(image);
				if ((int)imgObj[0].total() > minMatches)
				{
					outImagePoints[camera].push_back(imgObj[0]);
					outObjPoints[camera].push_back(imgObj[1]);
					kept_files[camera].push_back(image_list[camera][imgIdx]);
				}
				else {
					if (verbose) {
						WARN("Skipping file " + image_list[camera][imgIdx] + " due to few matches!");
					}
					skipped_files.push_back(image_list[camera][imgIdx]);
				}
			}
			if (verbose) {
				INFO("Images read for camera " + std::to_string(camera));
			}
		}
	}

	/*
	*/
	void writeImageAndObjectPoints(const std::string outfile, const VecStr2D& kept_files, const VecMat2D& imagepoints, const VecMat2D& objectpoints) {
		cv::FileStorage fs(outfile, cv::FileStorage::WRITE);
		if (fs.isOpened()) {
			for (int i = 0; i < (int)kept_files.size(); i++) {
				for (int j = 0; j < (int)kept_files[i].size(); j++) {
					fs << "url_" + std::to_string(i) + "_" + std::to_string(j) << kept_files[i][j];
					fs << "imgpoints_" + std::to_string(i) + "_" + std::to_string(j) << imagepoints[i][j];
					fs << "objpoints_" + std::to_string(i) + "_" + std::to_string(j) << objectpoints[i][j];
				}
			}
		}
		fs.release();
	}
	/*
	*/
	void readImageAndObjectPoints(const std::string infile, VecStr2D& kept_files, ImgPoints2D& imagepoints, ObjPoints2D& objectpoints) {
		cv::FileStorage fs(infile, cv::FileStorage::READ);
		if (fs.isOpened()) {
			int i, j;
			i = 0;
			j = 0;
			while (true) {
				j = 0;
				while (true) {
					if (kept_files.size() <= i) {
						kept_files.resize(i + 1);
						imagepoints.resize(i + 1);
						objectpoints.resize(i + 1);
					}
					if (kept_files[i].size() <= j) {
						kept_files[i].resize(j + 1);
						imagepoints[i].resize(j + 1);
						objectpoints[i].resize(j + 1);
					}
					fs["url_" + std::to_string(i) + "_" + std::to_string(j)] >> kept_files[i][j];
					fs["imgpoints_" + std::to_string(i) + "_" + std::to_string(j)] >> imagepoints[i][j];
					fs["objpoints_" + std::to_string(i) + "_" + std::to_string(j)] >> objectpoints[i][j];
					if (kept_files[i][j].empty()) {
						kept_files.pop_back();
						imagepoints.pop_back();
						objectpoints.pop_back();
						break;
					}
					j++;
				}
				if (j == 0) {
					break;
				}
				i++;
			}
		}
		fs.release();
	}
	/*
	*/
	VecMat makeMatrix(cv::Mat keypointsImageLocation, cv::Mat keypointsPatternLocation) {
		VecMat r(2);
		int imagePointsType = CV_MAKETYPE(CV_32F, 2);
		int objectPointsType = CV_MAKETYPE(CV_32F, 3);
		keypointsImageLocation.convertTo(r[0], imagePointsType);
		keypointsPatternLocation.convertTo(r[1], objectPointsType);
		return r;
	}
	/*
	load imagepoints and objectpoints from file
	*/
	void loadImageAndObjectPoints(std::string input, VecStr2D& urls, VecInt2D& timestampsAvailable, VecMat2D& imgPoints, VecMat2D& objPoints, int& nCamera, cv::Size& imageSize) {
		INFO("Loading image/object Points from file " + input);
		cv::Mat i, o;
		VecMat tmp;
		std::string url;
		cv::FileStorage fs(input, cv::FileStorage::READ);
		int camera = 0;
		int seq = 0;
		do {
			do {
				fs["url_" + std::to_string(camera) + "_" + std::to_string(seq)] >> url;
				fs["imgpoints_" + std::to_string(camera) + "_" + std::to_string(seq)] >> i;
				fs["objpoints_" + std::to_string(camera) + "_" + std::to_string(seq)] >> o;
				if (timestampsAvailable.size() == camera) { timestampsAvailable.resize(camera + 1); }
				if (!url.empty()) {
					std::string filename = url;
					int cameraVertex, timestamp;
					size_t spritPosition1 = filename.rfind('/');
					size_t spritPosition2 = filename.rfind('\\');
					if (spritPosition1 != std::string::npos)
					{
						filename = filename.substr(spritPosition1 + 1, filename.size() - 1);
					}
					else if (spritPosition2 != std::string::npos)
					{
						filename = filename.substr(spritPosition2 + 1, filename.size() - 1);
					}
					sscanf(filename.c_str(), "%d-%d", &cameraVertex, &timestamp);
					timestampsAvailable[cameraVertex].push_back(timestamp);
				}
				if (urls.size() == camera) { urls.resize(camera + 1); }
				if (imgPoints.size() == camera) { imgPoints.resize(camera + 1); }
				if (objPoints.size() == camera) { objPoints.resize(camera + 1); }
				if (i.rows <= 0 || i.cols <= 0) {
					break;
				}
				urls[camera].push_back(url);
				tmp = makeMatrix(i, o);
				imgPoints[camera].push_back(tmp[0]);
				objPoints[camera].push_back(tmp[1]);
				seq++;
			} while (true);
			if (seq == 0) {
				break;
			}
			seq = 0;
			camera++;
		} while (true);
		if (imgPoints.size() != objPoints.size()) {
			ERR("imgPoints.size() != objPoints.size()");
			return;
		}
		imgPoints.pop_back();
		objPoints.pop_back();
		timestampsAvailable.pop_back();
		urls.pop_back();
		nCamera = (int)imgPoints.size();
		imageSize = cv::imread(urls[0][0]).size();
		if (imageSize.width == 0 || imageSize.height == 0) {
			WARN("No image size found!");
		}
		INFO("Loading done.");
	}

	void pretty_print(const std::vector<int>& v) {
		static int count = 0;
		std::cout << "combination no " << (++count) << ": [ ";
		for (int i = 0; i < v.size(); ++i) { std::cout << v[i] << " "; }
		std::cout << "] " << std::endl;
	}

	void permutate(const std::vector<int>& list, std::vector<int>& tmp, int offset, int k, VecInt2D& result, bool verbose) {
		if (k == 0) {
			if (verbose) { pretty_print(tmp); }
			result.push_back(tmp);
			return;
		}
		for (int i = offset; i <= list.size() - k; ++i) {
			tmp.push_back(list[i]);
			permutate(list, tmp, i + 1, k - 1, result);
			tmp.pop_back();
		}
	}
	void permutate_deep(const VecInt& list, VecInt2D& result) {
		std::vector<int> comb;
		for (int k = 1; k <= list.size(); k++) {
			permutate(list, comb, 0, k, result);
		}
	}
	void writeSingleCameraParameters(const std::string& filename, int camera, float rms, cv::Mat& cameraMatrix, cv::Mat& distortCoeffs, cv::Mat& pose, VecMat patternPoses, VecStr& urls)
	{
		cv::FileStorage fs(filename, cv::FileStorage::WRITE);

		fs << "cameras" << camera;


		char num[10];
		sprintf(num, "%d", camera);
		std::string cm = "camera_matrix_" + std::string(num);
		std::string cameraPose = "camera_pose_" + std::string(num);
		std::string cameraDistortion = "camera_distortion_" + std::string(num);
		std::string cameraId = "camera_id_" + std::string(num);
		std::string cameraXi = "xi_" + std::string(num);

		fs << cm << cameraMatrix;
		fs << cameraDistortion << distortCoeffs;
		fs << cameraPose << pose;


		fs << "meanReprojectError" << rms;

		for (int i = 0; i < (int)patternPoses.size(); i++)
		{
			char timestamp[100];
			sprintf(timestamp, "%d", i);
			std::string photoTimestamp = "pose_timestamp_" + std::string(timestamp);
			std::string url = "url_" + std::string(timestamp);
			fs << photoTimestamp << patternPoses[i];
			fs << url << urls[i];
		}
	}

	cv::Mat concat(const cv::Mat& first, const cv::Mat& second) {
		if (first.rows != second.rows)
			return cv::Mat();
		cv::Mat result(first.rows,first.cols+second.cols, CV_64F);
		for (size_t row = 0; row < first.rows; row++) {
			for (size_t col = 0; col < first.cols; col++) {
				result.at<double>(row,col) = first.at<double>(row,col);
			}
		}
		for (size_t row = 0; row < second.rows; row++) {
			for (size_t col = 0; col < second.cols; col++) {
				result.at<double>(row, col+first.cols) = second.at<double>(row, col);
			}
		}
		return result;
	}

	void mergeRvecTvec(const cv::Mat& rvec, const cv::Mat& tvec, cv::Mat& transform) {
		cv::Mat rv, tv;
		if (rvec.type() != CV_32F)
		{
			rvec.convertTo(rv, CV_32F);
		}
		if (tvec.type() != CV_32F)
		{
			tvec.convertTo(tv, CV_32F);
		}

		transform = cv::Mat::eye(4, 4, CV_32F);
		cv::Mat R, T;
		cv::Rodrigues(rv, R);
		T = (tv).reshape(1, 3);
		R.copyTo(transform.rowRange(0, 3).colRange(0, 3));
		T.copyTo(transform.rowRange(0, 3).col(3));
	}

	cv::Mat XYZ2UV(const cv::Mat& intrinsics, const cv::Mat& extrinsics, const cv::Mat xyz) {
		if (intrinsics.rows == intrinsics.cols && intrinsics.rows == 3 && extrinsics.rows == 3 && extrinsics.cols == 4 && xyz.rows == 4 && xyz.cols == 1) {
			return intrinsics * extrinsics * xyz;
		} 
		std::cout << "wrong dimesions" << std::endl;
		return cv::Mat();
	}

	cv::Mat UV2XYZ(const cv::Mat& intrinsics, const cv::Mat& extrinsics, const cv::Mat uv) {
		return uv * extrinsics.inv() * intrinsics.inv();
	}

	cv::Vec3f cameraToWorld(const cv::Vec2f& uv, const cv::Mat& D, const cv::Mat& P) {
		cv::Vec3f xyzw(0., 0., 0.);
		if (D.rows != D.cols || D.rows != 3 || D.cols != 3) return xyzw;
		if (P.rows != 3 || P.cols != 4) return xyzw;
		cv::Mat matrix = cv::Mat::zeros(3, 1, CV_32F);
		cv::Mat result;
		result = D.inv() * P.inv() * matrix;
		return cv::Vec3f(
			result.at<float>(0, 0),
			result.at<float>(1, 0),
			result.at<float>(2, 0)
			);
	}
	cv::Vec2f worldToImage(const cv::Vec3f& xyz, const cv::Mat& D, const cv::Mat& P) {
		cv::Vec2f uv(0, 0);
		if (D.rows != D.cols || D.rows != 3 || D.cols != 3) return uv;
		if (P.rows != 3 || P.cols != 4) return uv;
		cv::Mat matrix = cv::Mat::zeros(4, 1, CV_32F);
		matrix.at<float>(0, 0) = xyz.val[0];
		matrix.at<float>(1, 0) = xyz.val[1];
		matrix.at<float>(2, 0) = xyz.val[2];
		matrix.at<float>(3, 0) = 1.f;
		cv::Mat result = D * P * matrix;
		return cv::Vec2f(result.at<float>(0, 0), result.at<float>(1, 0));
	}
	/*
	write calibration parameters
	*/
	void writeCalibrationParameters(const std::string filename, const int nCamera, const VecMat& cm, const VecMat& dc, const VecMat2D& rvecs, const VecMat2D& tvecs) {
		cv::FileStorage fs(filename, cv::FileStorage::WRITE);
		if (fs.isOpened()) {
			fs << "nCameras" << nCamera;

			for (int camIdx = 0; camIdx < nCamera; ++camIdx)
			{
				char num[10];
				sprintf(num, "%d", camIdx);
				std::string cameraMatrix = "camera_matrix_" + std::string(num);
				std::string cameraPose = "camera_pose_" + std::string(num);
				std::string cameraDistortion = "camera_distortion_" + std::string(num);
				std::string cameraId = "camera_id_" + std::string(num);
				std::string cameraXi = "xi_" + std::string(num);

				fs << cameraMatrix << cm[camIdx];
				fs << cameraDistortion << dc[camIdx];
				cv::Mat transform;
				mergeRvecTvec(rvecs[camIdx][0], tvecs[camIdx][0], transform);
				fs << cameraPose << transform;
				fs << cameraId << camIdx + 1;
			}
		}
	}
	/*
	write calibration parameters
	*/
	void writeParameters(const std::string filename, const int nCamera, const VecMat& cm, const VecMat& dc, const VecMat& transforms) {
		cv::FileStorage fs(filename, cv::FileStorage::WRITE);
		if (fs.isOpened()) {
			fs << "nCameras" << nCamera;

			for (int camIdx = 0; camIdx < nCamera; ++camIdx)
			{
				char num[10];
				sprintf(num, "%d", camIdx);
				std::string cameraMatrix = "camera_matrix_" + std::string(num);
				std::string cameraPose = "camera_pose_" + std::string(num);
				std::string cameraDistortion = "camera_distortion_" + std::string(num);
				std::string cameraId = "camera_id_" + std::string(num);
				std::string cameraXi = "xi_" + std::string(num);

				fs << cameraMatrix << cm[camIdx];
				fs << cameraDistortion << dc[camIdx];
				fs << cameraPose << transforms[camIdx];
				fs << cameraId << camIdx + 1;
			}
		}
		fs.release();
	}
	/*
	read calibration parameters
	*/
	void readParameters(const std::string filename, int* nCamera, VecMat& cm, VecMat& dc, VecMat& transforms) {
		cv::FileStorage fs(filename, cv::FileStorage::READ);
		if (fs.isOpened()) {
			fs["nCameras"] >> *nCamera;
			cm.resize(*nCamera);
			dc.resize(*nCamera);
			transforms.resize(*nCamera);
			for (int camIdx = 0; camIdx < *nCamera; ++camIdx)
			{
				char num[10];
				sprintf(num, "%d", camIdx);
				std::string cameraMatrix = "camera_matrix_" + std::string(num);
				std::string cameraPose = "camera_pose_" + std::string(num);
				std::string cameraDistortion = "camera_distortion_" + std::string(num);
				std::string cameraId = "camera_id_" + std::string(num);
				std::string cameraXi = "xi_" + std::string(num);

				fs[cameraMatrix] >> cm[camIdx];
				fs[cameraDistortion] >> dc[camIdx];
				fs[cameraPose] >> transforms[camIdx];
			}
		}
		fs.release();
	}
	/*
	removes item from vector
	*/
	void remove(std::vector<std::string>& vector, std::string item) {
		std::vector<std::string> new_vec;
		std::for_each(vector.begin(), vector.end(), [&](std::string s) {
			if (s != item) {
				new_vec.push_back(s);
			}
		});
		vector = new_vec;
	}
	/*
	OpenCV Matrix to std::string
	*/
	std::string mat2string(const cv::Mat& mat) {
		cv::Mat fMat;
		mat.copyTo(fMat);
		fMat.convertTo(fMat, CV_32F);
		std::string mat2string = "[";
		for (int row = 0; row < fMat.rows; row++) {
			for (int col = 0; col < fMat.cols; col++) {
				mat2string += std::to_string(fMat.at<float>(row, col));
				if (col + 1 != fMat.cols) { mat2string += ","; }
			}
			if (row + 1 != fMat.rows) { mat2string += "\n"; }
		}
		mat2string += "]";
		return mat2string;
	}

	std::ostream &operator<< (std::ostream &out, const glm::mat4 &mat) {
		out << "\n"
			<< mat[0][0] << "," << mat[1][0] << "," << mat[2][0] << "," << mat[3][0] << ",\n"
			<< mat[0][1] << "," << mat[1][1] << "," << mat[2][1] << "," << mat[3][1] << ",\n"
			<< mat[0][2] << "," << mat[1][2] << "," << mat[2][2] << "," << mat[3][2] << ",\n"
			<< mat[0][3] << "," << mat[1][3] << "," << mat[2][3] << "," << mat[3][3] << "\n";

		return out;
	}
	cv::Mat convertCVMat(std::string mat, int rows, int cols, int type) {
		cv::Mat result(rows,cols,type);
		std::istringstream iss(mat);
		std::vector<std::string> tokens{ std::istream_iterator<std::string>{iss},std::istream_iterator<std::string>{} };
		int index = 0;
		switch (type)
		{
		case CV_32F:
			index = 0;
			for (int row = 0; row < rows; row++) {
				for (int col = 0; col < cols; col++) {
					result.at<float>(row, col) = (float)std::atof(tokens[index++].c_str());
				}
			}
			break;
		case CV_64F:
			index = 0;
			for (int row = 0; row < rows; row++) {
				for (int col = 0; col < cols; col++) {
					result.at<double>(row, col) = std::atof(tokens[index++].c_str());
				}
			}
			break;
		default:
			return cv::Mat::eye(rows, cols, type);
		}
		return result;
	}

	cv::Mat calculatePatternPoseChessboardCV(const cv::Mat& img, const cv::Mat& CM, const cv::Mat& D, const cv::Size calibration_pattern_size, const cv::Size2f squareSize) {
		std::vector<cv::Point2f> corners;
		std::vector<cv::Point3f> obj_pts;
		if (cv::findChessboardCorners(img, calibration_pattern_size, corners)) {
			cv::Mat rvec{ cv::Size(3, 1), CV_64F };
			cv::Mat tvec{ cv::Size(3, 1), CV_64F };
			calcChessboardCorners(calibration_pattern_size, obj_pts, squareSize);
			if (cv::solvePnP(cv::Mat(obj_pts), cv::Mat(corners), CM, D, rvec, tvec)) {
				cv::Mat view_matrix_cv;
				//ViewMatrixFromRT<double>(rvec, tvec, view_matrix_cv);
				return view_matrix_cv;
			}
		}
		return cv::Mat::eye(4,4,CV_64F);
	}
	cv::Mat calculatePatternPoseRandomCV(const cv::Mat& img, const cv::Mat& pattern, const cv::Mat& CM, const cv::Mat& D, cv::Size2f patternSize, int minMatches, bool verbose, bool showExtraction) {
		cv::randpattern::RandomPatternCornerFinder finder = getRandomPatternFinder(patternSize.width, patternSize.height, minMatches, verbose, showExtraction);
		finder.loadPattern(pattern);
		cv::Mat rvec{ cv::Size(3, 1), CV_64F };
		cv::Mat tvec{ cv::Size(3, 1), CV_64F };
		VecMat points = finder.computeObjectImagePointsForSingle(img);
		if ((int)points[0].total() > minMatches) {
			if (cv::solvePnP(points[1], points[0], CM, D, rvec, tvec)) {
				cv::Mat view_matrix_cv;
				//ViewMatrixFromRT<double>(rvec, tvec, view_matrix_cv);
				return view_matrix_cv;
			}
		}
		return cv::Mat::eye(4,4,CV_64F);
	}
	glm::mat4 calculatePatternPoseChessboardGL(const cv::Mat& img, const cv::Mat& CM, const cv::Mat& D, const cv::Size calibration_pattern_size, const cv::Size2f squareSize) {
		return glm::mat4(1.0);/*ViewMatrixCVtoGL<double>(calculatePatternPoseChessboardCV(img,CM,D,calibration_pattern_size,squareSize));*/
	}
	glm::mat4 calculatePatternPoseRandomGL(const cv::Mat& img, const cv::Mat& pattern, const cv::Mat& CM, const cv::Mat& D, cv::Size2f patternSize, int minMatches, bool verbose, bool showExtraction) {
		return glm::mat4(1.0);/*ViewMatrixCVtoGL<double>(calculatePatternPoseRandomCV(img,pattern,CM,D,patternSize,minMatches,verbose,showExtraction));*/
	}
	void calcChessboardCorners(cv::Size boardSize, std::vector<cv::Point3f>& corners, cv::Size2f squareSize)
	{
		corners.resize(0);
		for (int i = 0; i < boardSize.height; i++)
			for (int j = 0; j < boardSize.width; j++)
				corners.push_back(cv::Point3f(float(j*squareSize.width),
					float(i*squareSize.height), 0));
	}
	cv::Mat getJulianKreiserCameraMatrix() {
		 return convertCVMat("1.2469551862064998e+003 0. 6.2854672757522837e+002 0. 1.2438407146922664e+003 5.2607698537800252e+002 0. 0. 1.", 3, 3, CV_64F);
	}
	cv::Mat getJulianKreiserDistortion() {
		return convertCVMat("-2.1191148811389071e-001 1.3695937236569153e-001 0. 0. -2.4612707433325057e-002", 5, 1, CV_64F);
	}
	cv::Mat getPanonoCamera07_CameraMatrix() {
		return convertCVMat("1.8562889350774328e+03 0. 1.0567999745101686e+03 0. 1.8573809798067821e+03 7.9577496350848378e+02 0. 0. 1.",3,3,CV_64F);
	}
	cv::Mat getPanonoCamera07_Distortion() {
		return convertCVMat("-1.5207055118693332e-02 -1.8297630436157002e-02 2.2255746540515370e-03 -9.5459018546836490e-04 -1.3185397071723429e-02",5,1,CV_64F);
	}
	bool readPoses(std::string file, VecMat& poses, VecStr& paths) {
		cv::FileStorage fs(file, cv::FileStorage::READ);
		if (fs.isOpened()) {
			int index = 0;
			poses.resize(0);
			paths.resize(0);
			cv::FileNode mat, path;
			while (true) {
				mat  = fs["pose_" + std::to_string(index)];
				path = fs["path_" + std::to_string(index)];
				if (mat.empty() || path.empty()) {
					break;
				}
				poses.push_back(mat.mat());
				paths.push_back(path.string());
				index++;
				
			}
			return true;
		}
		return false;
	}
	bool writeExtrinsics(const std::string output_xml_file, const VecMat2D& poses, const VecStr2D& urls) {
		if (poses.size() != urls.size()) {
			return false;
		}
		cv::FileStorage writer(output_xml_file, cv::FileStorage::WRITE);
		if (!writer.isOpened()) {
			return false;
		}
		for (int i = 0; i < poses.size(); i++) {
			if (poses[i].size() != urls[i].size()) {
				writer.release();
				return false;
			}
			for (int j = 0; j < poses[i].size(); j++) {
				writer << "pose_" + std::to_string(i) + "_" + std::to_string(j);
				writer << "path_" + std::to_string(i) + "_" + std::to_string(j);
			}
		}
		writer.release();
		return true;
	}
	bool readExtrinsics(std::string file, VecMat2D& poses, VecStr2D& paths) {
		cv::FileStorage fs(file, cv::FileStorage::READ);
		if (fs.isOpened()) {
			int index = 0;
			int stamp = 0;
			poses.resize(0);
			paths.resize(0);
			cv::FileNode mat, path;
			while (true) {
				stamp = 0;
				while (true)
				{
					mat = fs["pose_" + std::to_string(index) + "_" + std::to_string(stamp)];
					path = fs["path_" + std::to_string(index) + "_" + std::to_string(stamp)];
					if (mat.empty() || path.empty()) {
						break;
					}
					if (poses.size() == index) {
						poses.resize(index+1);
						paths.resize(index+1);
					}
					poses[index].push_back(mat.mat());
					paths[index].push_back(path.string());
					stamp++;
				}
				if (stamp == 0) {
					break;
				}
				index++;
			}
			return true;
		}
		return false;
	}

	bool readIntrinsics(const std::string intrinsics_xml_path, VecMat& CMs, VecMat& Ds) {
		cv::FileStorage fs(intrinsics_xml_path, cv::FileStorage::READ);
		if (fs.isOpened()) {
			CMs.resize(0);
			Ds.resize(0);
			cv::FileNode node_cm, node_d;
			double r;
			int i = 0;
			while (true)
			{
				node_cm = fs["camera_matrix_"+std::to_string(i)];
				node_d = fs["camera_distortion_" + std::to_string(i)];
				if (node_cm.empty()|| node_d.empty()) {
					break;
				}
				CMs.push_back(node_cm.mat());
				Ds.push_back(node_d.mat());
				i++;
			}
			return true;
		}
		return false;
	}

	bool writeIntrinsics(const std::string intrinsics_xml_path, const VecMat& CMs, const VecMat& Ds) {
		if (CMs.size() != Ds.size()) {
			return false;
		}
		cv::FileStorage writer(intrinsics_xml_path, cv::FileStorage::WRITE);
		if (writer.isOpened()) {
			for (int i = 0; i < CMs.size(); i++) {
				writer << "camera_matrix_" + std::to_string(i) << CMs[i];
				writer << "camera_distortion_" + std::to_string(i) << Ds[i];
			}
			writer.release();
			return true;
		}
		return false;
	}

	double calibrateCamera(VecStr stringlist, std::string pattern_file_path, cv::Mat& CM, cv::Mat& D, cv::Size2f patternSize, int minMatches) {
		cv::Mat pattern = cv::imread(pattern_file_path);
		cv::Mat image;
		cv::Size imageSize;
		immersight::ObjPoints objPoints;
		immersight::ImgPoints imgPoints;
		immersight::VecMat points;
		cv::randpattern::RandomPatternCornerFinder finder = immersight::getRandomPatternFinder(patternSize.width, patternSize.height, minMatches);
		finder.loadPattern(pattern);
		std::for_each(stringlist.begin(), stringlist.end(), [&](std::string path) {
			INFO(path);
			image = cv::imread(path, cv::IMREAD_GRAYSCALE);
			if (imageSize.width == 0) {
				imageSize = image.size();
			}
			if (image.size().width != imageSize.width) {
				return -1;
			}
			points = finder.computeObjectImagePointsForSingle(image);
			if ((int)points[0].total() > minMatches) {
				objPoints.push_back(points[1]);
				imgPoints.push_back(points[0]);
			}
		});
		cv::Mat R, T;
		double rms = cv::calibrateCamera(objPoints, imgPoints, imageSize, CM, D, R, T);
		return rms;
	}
	double calibrateCamera(std::string images_xml_path, std::string pattern_file_path, cv::Mat& CM, cv::Mat& D, cv::Size2f patternSize, int minMatches) {
		VecStr stringlist = immersight::readStringList(images_xml_path);
		return calibrateCamera(stringlist, pattern_file_path, CM, D,patternSize,minMatches);
	}

	void undistort(VecStr img_list, const cv::Mat& CM, const cv::Mat& D) {
		cv::Mat undist;
		std::for_each(img_list.begin(), img_list.end(), [&](std::string path) {
			cv::undistort(cv::imread(path), undist, CM, D);
			path.erase(path.end() - 3, path.end());
			cv::imwrite(path+"png", undist);
		});
	}
	bool calculatePoses(const std::string full_file_list_xml_path, const std::string intrinsics_xml_path, const std::string pattern_img_path, const std::string output_xml_path, bool verbose) {
		cv::Mat CM, D, R, T, image, pose;
		cv::FileStorage reader(intrinsics_xml_path, cv::FileStorage::READ);
		if (!reader.isOpened()) {
			return false;
		}
		immersight::VecStr list = immersight::readStringList(full_file_list_xml_path);
		immersight::VecStr2D list_all;
		immersight::VecInt2D timestamps;
		immersight::VecMat points;
		immersight::parseCamAndTimestamp(36, list, list_all, timestamps);
		cv::randpattern::RandomPatternCornerFinder finder = immersight::getRandomPatternFinder(109.f, 82.f, 20, false, false);
		finder.loadPattern(cv::imread(pattern_img_path, cv::IMREAD_GRAYSCALE));
		cv::FileStorage writer(output_xml_path, cv::FileStorage::WRITE);
		if (!writer.isOpened()) {
			reader.release();
			return false;
		}
		immersight::VecStr tmp;
		cv::FileNode node;
		for (int i = 0; i < list_all.size(); i++) {
			tmp = list_all[i];
			node = reader["camera_matrix_" + std::to_string(i)];
			CM = node.mat();
			if (node.empty()) {
				break;
			}
			node = reader["camera_distortion_" + std::to_string(i)];
			D = node.mat();
			if (node.empty()) {
				break;
			}
			for (int j = 0; j < tmp.size(); j++) {
				image = cv::imread(tmp[j], cv::IMREAD_GRAYSCALE);
				points = finder.computeObjectImagePointsForSingle(image);
				cv::solvePnP(points[1], points[0], CM, D, R, T, false, 0);
				viscom::ViewMatrixFromRT<double>(R, T, pose);
				writer << "pose_" + std::to_string(i) + "_" + std::to_string(j) << pose;
				writer << "path_" + std::to_string(i) + "_" + std::to_string(j) << tmp[j];
				if (verbose) { INFO("path_" + std::to_string(i) + "_" + std::to_string(j)); }
			}
		}
			
		writer.release();
		reader.release();
		return true;
	}
	bool readImgObj(cv::FileStorage& reader, const std::string id, cv::Mat& imgPoints, cv::Mat& objPoints, std::string url) {
		cv::FileNode node;
		if (reader.isOpened()) {
			node = reader[id + "_imgpoints"]; if (!node.empty()) { imgPoints = node.mat(); } else { return false; }
			node = reader[id + "_objPoints"]; if (!node.empty()) { objPoints = node.mat(); } else { return false; }
			node = reader[id + "_url"];       if (!node.empty()) { url = node.string(); }    else { return false; }
			return true;
		}
		return false;
	}
	void writeImgObj(cv::FileStorage& writer, const std::string id, const cv::Mat& imgPoints, const cv::Mat& objPoints, const std::string url) {
		if (writer.isOpened()) {
			writer << id + "_imgpoints" << imgPoints;
			writer << id + "_objPoints" << objPoints;
			writer << id + "_url" << url;
		}
	}
	bool readMulticalibrationData(cv::FileStorage& reader, int id, cv::Mat& CM, cv::Mat& D, cv::Mat& cam_pose, double *error) {
		cv::FileNode node_cm, node_d, node_pose, node_error;
		if (!reader.isOpened())
			return false;
		node_cm = reader["camera_matrix_"+std::to_string(id)];
		node_d = reader["camera_distortion_" + std::to_string(id)];
		node_pose = reader["camera_pose_" + std::to_string(id)];
		node_error = reader["error_" + std::to_string(id)];
		if (node_cm.empty() || node_d.empty() || node_pose.empty())
			return false;
		if (!node_error.empty())
			*error = node_error;
		node_cm.mat().copyTo(CM);
		node_d.mat().copyTo(D);
		node_pose.mat().copyTo(cam_pose);
		return true;
	}
	void readMulticalib(std::string calib_xml_path, VecMat& CMs, VecMat& Ds, VecMat& camera_poses, VecMat& pattern_poses, VecInt& pattern_idx, VecDouble& rms) {
		cv::FileStorage reader(calib_xml_path, cv::FileStorage::READ);
		if (reader.isOpened()) {
			CMs.resize(36);
			Ds.resize(36);
			rms.resize(36);
			camera_poses.resize(36);
			pattern_poses.resize(0);
			for (int i = 0; i < 36; i++) {
				if (!readMulticalibrationData(reader, i, CMs[i], Ds[i], camera_poses[i], &rms[i])) {
					WARN("could not read calibration data fom file: "+ calib_xml_path);
				}
			}
			cv::Mat pose;
			int timestamp = 0;
			while (timestamp < 1000) {
				if (readPatternPose(reader, "pose_timestamp_" + std::to_string(timestamp), pose)) {
					pattern_poses.push_back(pose);
					pattern_idx.push_back(timestamp);
				}
				timestamp++;
			}
		}
	}
	bool readPatternPose(cv::FileStorage& reader, std::string id, cv::Mat& pose) {
		if (!reader.isOpened())
			return false;
		cv::FileNode node;
		node = reader[id];
		if (node.empty())
			return false;
		pose = node.mat();
		return true;
	}

	std::string insert_back(const std::string s, int index, const std::string postfix) {
		std::string result(s);
		result.insert(s.size() - index, postfix);
		return result;
	}
} // namespace immerishgt