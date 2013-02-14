#pragma once

#include <opencv2/opencv.hpp>

class ColonyCounter
{
public:
	ColonyCounter(void);
	~ColonyCounter(void);

	void loadTraining(const char *path);
	void saveTraining(const char *path);

	// Trains the classifier given a set of sample images and label images which indicate
	// whether certain pixels are background, red colonies or blue colonies
	void trainClassifier(std::vector<std::string> trainPaths, std::vector<std::string> labelPaths);

	// Cleans up and normalizes an extracted petri film rectangle, keeping only the circle 
	// which fits within the rectangle.
	cv::Mat preprocessImage(cv::Mat petri, cv::Scalar& backgroundColor);
	cv::Mat preprocessImage(cv::Mat petri);

	// Classifies pixels within a preprocessed image to determine colony type or background
	cv::Mat classifyImage(cv::Mat img, bool debug = false, cv::Mat *debugImage = NULL);

	// Counts colonies in a classified image
	void countColonies(cv::Mat classified, int& red, int &blue, bool debug = false, cv::Mat *debugImage = NULL);

private:
	// True when svm has been trained
	bool trained;

	cv::SVM svm;
	static const int SVM_DIM = 3;

	// Classify a set of values that have been computed from a pixel
	int classifyValues(float* vals);
};