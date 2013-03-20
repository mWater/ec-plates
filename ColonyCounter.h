#pragma once

#include <opencv2/opencv.hpp>

/*
 * Main class for counting colonies. Uses a Support Vector Machine to
 * classify pixel colors. Can also use a 2-dimentional lookup table to
 * classify pixels. To use a lookup table, use loadTrainingQuantized.
 * Lookup table is used as the Support Vector Machine is quite slow
 *
 * Basic usage:
 *  loadTraining(...)
 *  preprocessImage(...)
 *  classifyImage(...)
 *  countColonies(...)
 *
 */
class ColonyCounter
{
public:
	ColonyCounter(void);
	~ColonyCounter(void);

	// Loads and saves training. See main.cpp for use.
	void loadTraining(const char *path);
	void loadTrainingQuantized(unsigned char *svmLookup, int *svmQuants);
	void saveTraining(const char *path);
	void saveTrainingQuantized(const char *path, int *svmQuants);

	// Trains the classifier given a set of sample images and label images which indicate
	// whether certain pixels are background, red colonies or blue colonies
	void trainClassifier(std::vector<std::string> trainPaths, std::vector<std::string> labelPaths, int *quants = NULL);

	// Cleans up and normalizes an extracted petri film rectangle, keeping only the circle 
	// which fits within the rectangle.
	cv::Mat preprocessImage(cv::Mat petri, cv::Scalar& backgroundColor);
	cv::Mat preprocessImage(cv::Mat petri);

	// Classifies pixels within a preprocessed image to determine colony type or background
	cv::Mat classifyImage(cv::Mat img, bool debug = false, cv::Mat *debugImage = NULL);
	cv::Mat classifyImageQuant(cv::Mat img, bool debug = false, cv::Mat *debugImage = NULL, int* quants = NULL);

	// Counts colonies in a classified image
	void countColonies(cv::Mat classified, int& red, int &blue, bool debug = false, cv::Mat *debugImage = NULL);

	// Test a quantization and prints debug info
	void testQuantization(cv::Mat img, int* quants);

private:
	// True when svm has been trained
	bool trained;

	// Dimension of SVM (number of inputs)
	cv::SVM svm;
	static const int SVM_DIM = 2;

	// Lookup table for SVM values using quantizations
	unsigned char *svmLookup;

	// Quantization values to use
	int *svmQuants;

	// Classify a set of values that have been computed from a pixel
	int classifyValues(float* vals);
};
