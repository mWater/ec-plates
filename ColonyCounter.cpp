#include "StdAfx.h"
#include "ColonyCounter.h"
#include "CircleFinder.h"

using namespace cv;

ColonyCounter::ColonyCounter(void)
{
	trained = false;
}

ColonyCounter::~ColonyCounter(void)
{
}


void ColonyCounter::loadTraining(const char *path) 
{
	svm.load(path);
	trained = true;
}

void ColonyCounter::saveTraining(const char *path) 
{
	svm.save(path);
}

int labelColorToIndex(Scalar c) 
{
	if (c==Scalar(0,255,0)) // Background
		return 0;
	if (c==Scalar(0,0,255)) // Red
		return 1;
	if (c==Scalar(255,255,0)) // Cyan = Blue
		return 2;
	return -1;
}

void convertColor(Vec3b &color, float *vals) 
{
	// Get lightness
	vals[0] = ((float)color(0) + (float)color(1) + (float)color(2))/600;

	// Get red vs blue
	vals[1] = (float)color(2)/(float)(color(0) + color(2));

	// Get green vs blue
	vals[2] = (float)color(1)/(float)(color(0) + color(1));
}

int ColonyCounter::classifyValues(float* vals) 
{
	assert(trained);

	Mat sampleMat = Mat(1, SVM_DIM, CV_32F, vals);
	float response = svm.predict(sampleMat);
	return response;
}


void ColonyCounter::trainClassifier(vector<string> trainPaths, vector<string> labelPaths) 
{
	// Count training data
	int trainCnt = 0;
	int trainByLabel[3] = { 0, 0, 0 };

	vector<Rect> petriRects;

	for (int k=0;k<trainPaths.size();k++)
	{
		// Extract rectangles
		Mat img = imread(trainPaths[k]);
		petriRects.push_back(findPetriRect(img));

		// Load label image
		Mat labelImg = imread(labelPaths[k]);

		// Count training data
		for (int x=0;x<labelImg.cols;x++)
		{
			for (int y=0;y<labelImg.rows;y++)
			{
				Scalar c = labelImg.at<Vec3b>(y, x);
				int label = labelColorToIndex(c);
				if (label >= 0)
				{
					trainByLabel[label]++;
					trainCnt++;
				}
			}
		}
	}

	// Create matricies for training
    Mat labelsMat(trainCnt, 1, CV_32FC1);
    Mat trainingDataMat(trainCnt, SVM_DIM, CV_32FC1);
	long n = 0;
	for (int k=0;k<trainPaths.size();k++)
	{
		// Load label image
		Mat labelImg = imread(labelPaths[k]);

		// Extract label for rectagle
		labelImg = labelImg(petriRects[k]);

		// Preprocess training image
		Mat trainImg = imread(trainPaths[k]);
		trainImg = preprocessImage(trainImg(petriRects[k]));

		for (int x=0;x<trainImg.cols;x++)
		{
			for (int y=0;y<trainImg.rows;y++)
			{
				Scalar c = labelImg.at<Vec3b>(y, x);
				int label = labelColorToIndex(c);
				if (label >= 0)
				{
					Vec3b color = trainImg.at<Vec3b>(y,x);
					float vals[SVM_DIM];
					convertColor(color, vals);

					for (int i=0;i<SVM_DIM;i++)
						trainingDataMat.at<float>(n, i) = vals[i];
					labelsMat.at<float>(n,0) = label;
					n++;
				}
			}
		}
	}

    CvSVMParams params;
    params.svm_type    = CvSVM::C_SVC;
	params.kernel_type = CvSVM::LINEAR;
    params.term_crit   = cvTermCriteria(CV_TERMCRIT_ITER, 100, 1e-6);
	params.C = 10;

	// Train the SVM
    bool res = svm.train(trainingDataMat, labelsMat, Mat(), Mat(), params);

	trained = res;
}

static Mat lowPass3C(Mat &img, Mat &mask3C, int blurSize) {
	// Create low-pass filter, only within mask
	Mat blurred;
	Mat blurredCount;
	boxFilter(img & mask3C, blurred, CV_32SC3, Size(blurSize, blurSize),
		Point(-1, -1), false, BORDER_CONSTANT);
	boxFilter(mask3C, blurredCount, CV_32SC3, Size(blurSize, blurSize),
		Point(-1, -1), false, BORDER_CONSTANT);
	blurred = blurred / (blurredCount / 255);
	Mat blurred8;
	blurred.convertTo(blurred8, CV_8UC3);
	return blurred8;
}

static Mat lowPass(Mat &img, Mat &mask, int blurSize) {
	Mat mask3C = Mat(img.size(), CV_8UC3, Scalar(0, 0, 0));
	mask3C.setTo(Scalar(255, 255, 255), mask);
	return lowPass3C(img, mask3C, blurSize);
}

static Mat findBackground(Mat& img, Mat& mask, int blurSize, Scalar& backgroundColor, int debug) {
	Mat lowpass = lowPass(img, mask, blurSize * 2 + 1);

	// Get outliers and remove from mask
	Mat diff1 = img-lowpass;
	Mat diff2 = lowpass-img;
	
	if (debug) {
		imshow("diff1", diff1);
		imshow("diff2", diff2);
	}

	Mat background1, background2;
	threshold(diff1, background1, 10, 255, CV_THRESH_BINARY_INV);
	threshold(diff2, background2, 10, 255, CV_THRESH_BINARY_INV);

	// Keep only pixels where all channels are close to lowpass
	Mat bgmask = mask.clone();
	vector<Mat> channels;
	split(background1&background2, channels);

	// Get background mask
	for (int i=0;i<channels.size();i++) {
		bgmask &= channels[i];
	}

	if (debug) {
		imshow("background1", background1);
		imshow("background2", background2);
	}

	// Calculate background
	Mat background = lowPass(img, bgmask, blurSize * 2 + 1);

	Mat bgmask3C = Mat(img.size(), CV_8UC3, Scalar(0, 0, 0));
	bgmask3C.setTo(Scalar(255, 255, 255), bgmask);

	// Get average background color
	Scalar backTotal = sum(background & bgmask3C);
	Scalar backCnt = sum(bgmask)/255;
	backgroundColor = backTotal/(backCnt[0]);

	return background;
}


Mat ColonyCounter::preprocessImage(Mat petri) 
{
	Scalar backgroundColor;
	return preprocessImage(petri, backgroundColor);
}

Mat ColonyCounter::preprocessImage(Mat petri, Scalar& backgroundColor) 
{
	// Create mask
	Mat mask(petri.size(), CV_8UC1, Scalar(0));
	circle(mask, Point(mask.cols/2, mask.rows/2), mask.rows/2, Scalar(255), CV_FILLED);

	Mat background = findBackground(petri, mask, mask.rows/5, backgroundColor, false);
	
	// High-pass image
	Mat highpass;
	highpass = (petri * 200) / background;

	// Mask outside of circle to background
	highpass.setTo(Scalar(200, 200, 200), 255 - mask);

	return highpass;
}

static vector<vector<Point> > countType(Mat classified, int type) 
{
	// Get type mask
	Mat mask = classified == type;

	// Remove tiny colonies
	Mat kernel = getStructuringElement(MORPH_CROSS, Size(3, 3));
	erode(mask, mask, kernel);
	dilate(mask, mask, kernel);
	//erode(mask, mask, kernel);

	// Dilate, erode to remove small
	kernel = getStructuringElement(MORPH_ELLIPSE, Size(5, 5));
	dilate(mask, mask, kernel);
	erode(mask, mask, kernel);
	dilate(mask, mask, kernel);
	erode(mask, mask, kernel);

	// Find contours
	vector<vector<Point> > contours;
	vector<Vec4i> hierarchy;
	findContours(mask, contours, hierarchy, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_NONE);

	// For each suitable one
	vector<vector<Point> > goodContours;
	for (int i=0;i<contours.size();i++) 
	{
//		if (contourArea(contours[i])>4)
		goodContours.push_back(contours[i]);
	}
	return goodContours;
}

Mat ColonyCounter::classifyImage(Mat img, bool debug, Mat *debugImage) 
{
	Mat classified(img.size(), CV_8U);

	Mat demo;
	if (debug)
		demo = img.clone();

	// Show predictions
	for (int x=0;x<img.cols;x++)
	{
		for (int y=0;y<img.rows;y++)
		{
			Vec3b color = img.at<Vec3b>(y,x);
			float vals[SVM_DIM];
			convertColor(color, vals);
			int cls = classifyValues(vals);
			classified.at<unsigned char>(y,x)=cls;
			if (debug) 
			{
				if (cls == 0)
					demo.at<Vec3b>(y,x)=Vec3b(255,255,255);
				if (cls == 1)
					demo.at<Vec3b>(y,x)=Vec3b(0,0,255);
				if (cls == 2)
					demo.at<Vec3b>(y,x)=Vec3b(255,0,0);
			}
		}
	}
	if (debug) 
		demo.copyTo(*debugImage);

	return classified;
}

void ColonyCounter::countColonies(Mat classified, int& red, int &blue, bool debug, Mat *debugImage) 
{
	vector<vector<Point> > redContours, blueContours;

	blueContours = countType(classified, 2);
	redContours = countType(classified, 1);

	if (debug) 
	{
		Mat contours(classified.size(), CV_8UC3, Scalar(255, 255, 255));
		drawContours(contours, redContours, -1, Scalar(128,128,255), -1);
		drawContours(contours, redContours, -1, Scalar(0,0,255));
		drawContours(contours, blueContours, -1, Scalar(255,125,128), -1);
		drawContours(contours, blueContours, -1, Scalar(255,0,0));
		contours.copyTo(*debugImage);
	}

	red = redContours.size();
	blue = blueContours.size();
}
