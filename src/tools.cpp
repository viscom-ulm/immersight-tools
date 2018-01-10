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
	cv::Ptr<cv::xfeatures2d::SURF> getSURF(int metrixThreshold, int numberOfOctaves, int numScaleLevels )
    {
		return cv::xfeatures2d::SURF::create(metrixThreshold, numberOfOctaves, numScaleLevels);
	}
    cv::Ptr<cv::ORB> getORB(int nfeatures, float scaleFactor, int nlevels, int edgeThreshold, int firstLevel, int wta_k, int scoreType, int patchSize, int fastThreshold)
	{
        return cv::ORB::create(nfeatures, scaleFactor, nlevels, edgeThreshold, firstLevel, wta_k, scoreType, patchSize, fastThreshold);
	}
	/*
	returns a Flann based descriptor matcher
	*/
	cv::Ptr<cv::DescriptorMatcher> getFLANN() {
		return cv::DescriptorMatcher::create(cv::DescriptorMatcher::FLANNBASED);
	}

    cv::Mat computeDescriptors(const cv::Mat& image, std::vector<cv::KeyPoint>& keyPoints, const cv::Ptr<cv::DescriptorExtractor> de)
    {
        cv::Mat descriptor;
        de->compute(image, keyPoints,descriptor);
        return descriptor;
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
	*/
	VecMat makeMatrix(cv::Mat keypointsImageLocation, cv::Mat keypointsPatternLocation) {
		VecMat r(2);
		int imagePointsType = CV_MAKETYPE(CV_32F, 2);
		int objectPointsType = CV_MAKETYPE(CV_32F, 3);
		keypointsImageLocation.convertTo(r[0], imagePointsType);
		keypointsPatternLocation.convertTo(r[1], objectPointsType);
		return r;
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

	void calcChessboardCorners(cv::Size boardSize, std::vector<cv::Point3f>& corners, cv::Size2f squareSize)
	{
		corners.resize(0);
		for (int i = 0; i < boardSize.height; i++)
			for (int j = 0; j < boardSize.width; j++)
				corners.push_back(cv::Point3f(float(j*squareSize.width),
					float(i*squareSize.height), 0));
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

    void parseFileNameAndExt(const std::string& path, std::string& filename, std::string& ext)
	{
        auto index = 0;
        auto length = 0;
	    for(auto i = path.size()-1; i >0; --i)
	    {
            if (path.at(i) == '/' || path.at(i) == '\\') { break; }
            index = i;
            length = path.size() - i;
	    }
        filename = path.substr(index, length);
        ext = filename.substr(filename.size() - 3, 3);
        filename = filename.substr(0, filename.size() - 4);
	}

    bool parseNumberOfDigits(const std::string& str, int& nod)
	{
        auto numberAsString = std::string();
	    auto numberStarted = false;
        for(const auto c : str)
        {
            if(std::isdigit(c))
            {
                numberStarted = true;
                numberAsString.push_back(c);
            } else
            {
                if (numberStarted) return false;
            }
        }
        nod = numberAsString.size();
        return nod >= 0;
	}

    void parseFileNameWithoutNumber(const std::string& path, std::string& pathwithoutfilename, std::string& stem, int& numberOfDigits, std::string& ext)
	{
        auto filename = std::string();
        parseFileNameAndExt(path, filename, ext);
        pathwithoutfilename = path.substr(0, path.size() - (filename.size() + 4));
        if(parseNumberOfDigits(filename, numberOfDigits))
        {
            for(auto c:filename)
            {
                if (!isdigit(c)) stem.push_back(c);
            }
        }
	}
} // namespace immerishgt