// tools.h
#pragma once
#include <opencv2/core.hpp>
#include <opencv2/xfeatures2d.hpp>
#include <opencv2/ccalib/randpattern.hpp>

#include <vector>
#include <map>

#include <glm/glm.hpp>
#include <glm/ext.hpp>

namespace immersight{
	typedef std::vector<cv::Mat> VecMat;
	typedef std::vector<std::vector<cv::Mat>> VecMat2D;
	typedef std::vector<std::string> VecStr;
	typedef std::vector<std::vector<std::string>> VecStr2D;
	typedef std::vector<double> VecDouble;
	typedef std::vector<int> VecInt;
	typedef std::vector<std::vector<int>> VecInt2D;
	typedef std::vector<cv::Vec2f> VecVec2f;
	typedef std::vector<std::vector<cv::Vec2f>> VecVec2f2D;
	typedef std::vector<cv::Vec3f> VecVec3f;
	typedef std::vector<std::vector<cv::Vec3f>> VecVec3f2D;
	typedef std::vector<std::vector<std::vector<cv::Vec3f>>> VecVec3f3D;
	typedef std::vector<std::vector<std::vector<cv::Vec2f>>> VecVec2f3D;
	typedef std::vector<cv::Point2f> VecPoint2f;
	typedef std::vector<cv::Point3f> VecPoint3f;
	typedef std::vector<std::vector<cv::Point2f>> VecPoint2f2D;
	typedef std::vector<std::vector<cv::Point3f>> VecPoint3f2D;
	typedef std::vector<std::vector<std::vector<cv::Point3f>>> VecPoint3f3D;
	typedef std::vector<std::vector<std::vector<cv::Point2f>>> VecPoint2f3D;

	typedef VecMat ObjPoints;
	typedef VecMat ImgPoints;
	typedef VecMat Rvecs;
	typedef VecMat Tvecs;
	typedef VecMat2D ObjPoints2D;
	typedef VecMat2D ImgPoints2D;
	typedef VecMat2D Rvecs2D;
	typedef VecMat2D Tvecs2D;

	/*
	struct that holds all relevant data from calibration to visualize
	*/
	struct calibIO {
		calibIO() { nCamera = 0; rms = 0.0; };
		calibIO(int c) {
			nCamera = c;
			availableTimestamps.resize(nCamera);
			cameraMatrix.resize(nCamera);
			distortCoeff.resize(nCamera);
			cameraPoses.resize(nCamera);
			cameraId.resize(nCamera);
			prePose.resize(nCamera);
		};
		calibIO(int n, double err, VecInt2D timestamps, VecMat cm, VecMat dc, VecMat cp, std::map<int, cv::Mat> pp, VecInt cid, VecMat pre) {
			nCamera = n;
			rms = err;
			availableTimestamps = timestamps;
			cameraMatrix = cm;
			distortCoeff = dc;
			cameraPoses = cp;
			patternPoses = pp;
			cameraId = cid;
			prePose = pre;
		};
		int nCamera;
		double rms;
		VecInt2D availableTimestamps;
		VecMat cameraMatrix;
		VecMat distortCoeff;
		VecMat cameraPoses;
		std::map<int,cv::Mat> patternPoses;
		VecInt cameraId;
		VecMat prePose;
	};
	
	void writeMulticalibrationParameter(std::string filename, calibIO& data);
	void readMulticalibrationParameter(std::string filename, calibIO& data);
	float deg2rad(float deg);
	double deg2rad(double deg);
	double nrand(double max);
	int nrand(int max);
	cv::Ptr<cv::xfeatures2d::SURF> getSURF();
	cv::Ptr<cv::DescriptorMatcher> getFLANN();
	cv::randpattern::RandomPatternCornerFinder getRandomPatternFinder(float patternWidth = 109.f, float patternHeight = 82.f, int minMatches = 20, bool verbose = 0, bool showExtraction = 0);
	VecStr readStringList(std::string file);
	int getTimestampIndex(int timestamp, VecInt& timestamps);
	void parseCamAndTimestamp(const int nCamera, const VecStr& file_list, VecStr2D& filesEachCameraFull, VecInt2D& timestampFull);
	void parseCamAndTimestampSingle(const std::string& string, int *cam, int *timestamp);
	void match(VecStr2D& image_list, VecStr& skipped_files, VecStr2D& kept_files, int nCamera, int minMatches, float scale, cv::randpattern::RandomPatternCornerFinder& matcher, VecMat2D& outImagePoints, VecMat2D& outObjPoints, bool verbose = false);
	void writeImageAndObjectPoints(const std::string outfile, const VecStr2D& kept_files, const VecMat2D& imagepoints, const VecMat2D& objectpoints);
	void readImageAndObjectPoints(const std::string infile, VecStr2D& kept_files, ImgPoints2D& imagepoints, ObjPoints2D& objectpoints);
	VecMat makeMatrix(cv::Mat keypointsImageLocation, cv::Mat keypointsPatternLocation);
	void loadImageAndObjectPoints(std::string input, VecStr2D& urls, VecInt2D& timestampsAvailable, VecMat2D& imgPoints, VecMat2D& objPoints, int& nCamera, cv::Size& imageSize);
	void pretty_print(const std::vector<int>& v);
	void permutate(const std::vector<int>& list, std::vector<int>& tmp, int offset, int k, VecInt2D& result, bool verbose = false);
	void permutate_deep(const VecInt& list, VecInt2D& result);
	void writeSingleCameraParameters(const std::string& filename, int camera, float rms, cv::Mat& cameraMatrix, cv::Mat& distortCoeffs, cv::Mat& pose, VecMat patternPoses, VecStr& urls);
	void mergeRvecTvec(const cv::Mat& rvec, const cv::Mat& tvec, cv::Mat& transform);
	cv::Vec3f cameraToWorld(const cv::Vec2f& uv, const cv::Mat& D, const cv::Mat& P);
	cv::Vec2f worldToImage(const cv::Vec3f& xyz, const cv::Mat& D, const cv::Mat& P);
	cv::Mat XYZ2UV(const cv::Mat& intrinsics, const cv::Mat& extrinsics, const cv::Mat xyz);
	cv::Mat UV2XYZ(const cv::Mat& intrinsics, const cv::Mat& extrinsics, const cv::Mat uv);
	cv::Mat concat(const cv::Mat& first, const cv::Mat& second);
	void writeCalibrationParameters(const std::string filename, const int nCamera, const VecMat& cm, const VecMat& dc, const VecMat2D& rvecs, const VecMat2D& tvecs);
	void writeParameters(const std::string filename, const int nCamera, const VecMat& cm, const VecMat& dc, const VecMat& transforms);
	void readParameters(const std::string filename, int* nCamera, VecMat& cm, VecMat& dc, VecMat& transforms);
	bool readPoses(std::string file, VecMat& poses, VecStr& paths);
	bool readIntrinsics(const std::string intrinsics_xml_path, VecMat& CMs, VecMat& Ds);
	bool writeIntrinsics(const std::string intrinsics_xml_path, const VecMat& CMs, const VecMat& Ds);
	void remove(std::vector<std::string>& vector, std::string item);
	std::string mat2string(const cv::Mat& mat);
	std::ostream &operator<< (std::ostream &out, const glm::mat4 &mat);
	cv::Mat convertCVMat(std::string mat, int rows, int cols, int type);
	glm::mat4 calculatePatternPoseChessboardGL(const cv::Mat& img, const cv::Mat& CM, const cv::Mat& D, const cv::Size calibration_pattern_size, const cv::Size2f squareSize);
	glm::mat4 calculatePatternPoseRandomGL(const cv::Mat& img, const cv::Mat& pattern, const cv::Mat& CM, const cv::Mat& D, cv::Size2f patternSize, int minMatches = 20, bool verbose = false, bool showExtraction = false);
	cv::Mat calculatePatternPoseChessboardCV(const cv::Mat& img, const cv::Mat& CM, const cv::Mat& D, const cv::Size calibration_pattern_size, const cv::Size2f squareSize);
	cv::Mat calculatePatternPoseRandomCV(const cv::Mat& img, const cv::Mat& pattern, const cv::Mat& CM, const cv::Mat& D, cv::Size2f patternSize, int minMatches = 20, bool verbose = false, bool showExtraction = false);
	void calcChessboardCorners(cv::Size boardSize, std::vector<cv::Point3f>& corners, const cv::Size2f squareSize);
	cv::Mat getJulianKreiserCameraMatrix();
	cv::Mat getJulianKreiserDistortion();
	cv::Mat getPanonoCamera07_CameraMatrix();
	cv::Mat getPanonoCamera07_Distortion();
	void undistort(VecStr image_paths, const cv::Mat& CM, const cv::Mat& D);
	double calibrateCamera(std::string images_xml_path, std::string pattern_file_path, cv::Mat& CM, cv::Mat& D, cv::Size2f patternSize = cv::Size2f(109.f, 82.f), int minMatches = 20);
	double calibrateCamera(VecStr paths, std::string pattern_file_path, cv::Mat& CM, cv::Mat& D, cv::Size2f patternSize = cv::Size2f(109.f, 82.f), int minMatches = 20);
	bool calculatePoses(const std::string full_file_list_xml_path, const std::string intrinsics_xml_path, const std::string pattern_img_path, const std::string output_xml_path, bool verbose = false);
	bool writeExtrinsics(const std::string output_xml_file, const VecMat2D& poses, const VecStr2D& urls);
	bool readExtrinsics(std::string extrinsics_xml_file, VecMat2D& poses, VecStr2D& paths);
	template <typename T>
	bool contains(const std::vector<T>& vector, T item) {
		return std::find(vector.begin(), vector.end(), item) != vector.end();
	}
	void writeImgObj(cv::FileStorage& writer, const std::string id, const cv::Mat& imgPoints, const cv::Mat& objPoints, const std::string url);
	bool readImgObj(cv::FileStorage& reader,  const std::string id, cv::Mat& imgPoints, cv::Mat& objPoints, std::string url);
	bool readMulticalibrationData(cv::FileStorage& reader, int id, cv::Mat& CM, cv::Mat& D, cv::Mat& cam_pose);
	void readMulticalib(std::string calib_xml_path, VecMat& CMs, VecMat& Ds, VecMat& camera_poses, VecMat& pattern_poses, VecInt& pattern_idx, VecDouble& rms);
	bool readPatternPose(cv::FileStorage& reader,std::string id, cv::Mat& pose);
	std::string cameraIndex2string(int index);
	std::string insert_back(const std::string s, int index, const std::string postfix);
	template <typename T>
	glm::mat4 opencvMat2openglMat(const cv::Mat& view_cv) {
		if (view_cv.rows == 0 || view_cv.cols == 0) {
			return glm::mat4(1);
		}
		cv::Mat view_gl;
		cv::transpose(view_cv, view_gl);

		const T* data = &view_gl.at<T>(0, 0); // access the data
		return std::move(glm::make_mat4(data));
	}
}  // end immersight

namespace viscom {

	/*Author Julian Kreiser*/
	template <typename T>
	void ViewMatrixFromRT(const cv::Mat& rvec, const cv::Mat& tvec, cv::Mat& view_matrix) {
		cv::Mat R;
		cv::Rodrigues(rvec, R);
		//cv::Mat pos = -R * tvec;
		//std::cout << "position:" << std::endl;
		//Print2DMatrix<float>(pos);
		view_matrix = cv::Mat::zeros(4, 4, R.type());
		for (int r = 0; r < R.rows; r++) {
			for (int c = 0; c < R.cols; c++) {
				view_matrix.at<T>(r, c) = R.at<T>(r, c);
			}
			view_matrix.at<T>(r, 3) = tvec.at<T>(r, 0);
		}
		view_matrix.at<T>(3, 3) = static_cast<T>(1.0);
	}
	/*Author Julian Kreiser*/
	template <typename T>
	void ViewMatrixFromRT(const cv::Vec3d& rvec, const cv::Vec3d& tvec, cv::Mat& view_matrix) {
		ViewMatrixFromRT<T>(cv::Mat(rvec), cv::Mat(tvec), view_matrix);
	}
	/*Author Julian Kreiser*/
	template <typename T>
	void ViewMatrixFromRT(const cv::Vec3f& rvec, const cv::Vec3f& tvec, cv::Mat& view_matrix) {
		ViewMatrixFromRT<T>(cv::Mat(rvec), cv::Mat(tvec), view_matrix);
	}
	/*Author Julian Kreiser*/
	template <typename T>
	void ViewMatrixCVtoGL(const cv::Mat& view_cv, cv::Mat& view_gl) {
		// http://answers.opencv.org/question/23089/opencv-opengl-proper-camera-pose-using-solvepnp/

		cv::Mat cv_to_gl = cv::Mat::eye(view_cv.rows, view_cv.cols, view_cv.type());
		cv_to_gl.at<T>(1, 1) = static_cast<T>(-1.0); // Invert the y axis
		cv_to_gl.at<T>(2, 2) = static_cast<T>(-1.0); // invert the z axis

		const cv::Mat view_matrix = cv_to_gl * view_cv; // transform axis
		view_gl = cv::Mat(view_cv.rows, view_cv.cols, view_cv.type());
		cv::transpose(view_matrix, view_gl); // row order to column order
	}
	/*Author Julian Kreiser*/
	template <typename T>
	glm::mat4 ViewMatrixCVtoGL(const cv::Mat& view_cv) {
		if (view_cv.rows == 0 || view_cv.cols == 0) {
			return glm::mat4(1);
		}
		cv::Mat view_gl;
		ViewMatrixCVtoGL<T>(view_cv, view_gl);

		const T* data = &view_gl.at<T>(0, 0); // access the data
		return std::move(glm::make_mat4(data));
	}
	/*Author Julian Kreiser*/
	template <typename T>
	glm::mat4 ProjectionMatrixFromCalibratedCamera(const cv::Mat& camera_matrix, float z_near, float z_far, float width, float height, float x0 = 0.0f, float y0 = 0.0f, bool y_up = false)
	{
		// https://strawlab.org/2011/11/05/augmented-reality-with-OpenGL


		const float depth = z_far - z_near;
		const float q = -(z_far + z_near) / depth;
		const float qn = -2.0f * (z_far * z_near) / depth;

		const float K00 = static_cast<float>(camera_matrix.at<T>(0, 0)); // alpha
		const float K01 = static_cast<float>(camera_matrix.at<T>(0, 1)); // s
		const float K02 = static_cast<float>(camera_matrix.at<T>(0, 2)); // x0
		const float K11 = static_cast<float>(camera_matrix.at<T>(1, 1)); // beta
		const float K12 = static_cast<float>(camera_matrix.at<T>(1, 2)); // y0

		glm::mat4 mat_proj;
		if (y_up) {
			mat_proj = glm::mat4(2.0f * K00 / width, -2.0f * K01 / width, (-2.0f * K02 + width + 2.0f * x0) / width, 0.0f,
				0.0f, -2.0f * K11 / height, (-2.0f * K12 + height + 2.0f * y0) / height, 0.0f,
				0.0f, 0.0f, q, qn,
				0.0f, 0.0f, -1.0f, 0.0f);
		}
		else
		{
			mat_proj = glm::mat4(
				2.0f * K00 / width, -2.0f * K01 / width, (-2.0f * K02 + width + 2.0f * x0) / width, 0.0f,

				0.0f, 2.0f * K11 / height, (2.0f * K12 - height + 2.0f * y0) / height, 0.0f,

				0.0f, 0.0f, q, qn,

				0.0f, 0.0f, -1.0f, 0.0f
				);
		}

		return std::move(glm::transpose(mat_proj));
	}

	/* Julian Kreiser */
	void DownloadOpenGLTexturetoOpenCVMat(cv::Mat& img, const cv::Size& s);
}  // end viscom