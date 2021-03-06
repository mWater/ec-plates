#include "stdafx.h"
#include "ColonyCounter.h"
#include "CircleFinder.h"

using namespace cv;

ColonyCounter::ColonyCounter(void)
{
	trained = false;
	svmLookup = NULL;
	svmQuants = NULL;
}

ColonyCounter::~ColonyCounter(void)
{
}


void ColonyCounter::loadTraining(const char *path) 
{
	svm.load(path);
	trained = true;
}

void ColonyCounter::loadTrainingQuantized(unsigned char *svmLookup, int *svmQuants)
{
	this->svmLookup = svmLookup;
	this->svmQuants = svmQuants;
	trained = true;
}


void ColonyCounter::saveTraining(const char *path) 
{
	svm.save(path);
}

/*
 * Writes out a header file that contains a large array of SVM results
 * to look up, where both inputs must be between zero and one.
 *
 * svmQuants is quantization to use. e.g. quantization of 10 will produce
 * lookup values for 0, 0.1, 0.2, ... 0.9
 */
void ColonyCounter::saveTrainingQuantized(const char *path, int *svmQuants)
{
	FILE *file;
	file = fopen(path, "w");
	fprintf(file, "// AUTOGENERATED FILE by ColonyCounter::saveTrainingQuantized\n");
	fprintf(file, "static int svmQuants[] = { ");
	for (int i=0;i<SVM_DIM;i++)
	{
		if (i>0)
			fprintf(file, ",");
		fprintf(file, "%d", svmQuants[i]);
	}
	fprintf(file, "};\n");

	fprintf(file, "static unsigned char svmLookup[] = { ");
	int index = 0;
	for (int q1=0;q1<svmQuants[1];q1++)
	{
		fprintf(file, "\n");
		for (int q0=0;q0<svmQuants[0];q0++)
		{
			float vals[2];
			vals[0] = q0 * 1.0 / svmQuants[0];
			vals[1] = q1 * 1.0 / svmQuants[1];
			int cls = classifyValues(vals);
			fprintf(file, " %d", cls);

			if (q0 != svmQuants[0] - 1 || q1 != svmQuants[1] - 1)
				fprintf(file, ",", cls);
			index++;
		}
	}

	fprintf(file, "};\n");

	fclose(file);
}

/*
 * Converts a color in a label file (used for training) to an
 * classification value.
 */
int labelColorToIndex(Vec3b c) 
{
	if (c==Vec3b(0,255,0)) // Background
		return 0;
	if (c==Vec3b(0,0,255)) // Red
		return 1;
	if (c==Vec3b(255,255,0)) // Cyan = Blue
		return 2;
	return -1;
}

/*
 * Converts a color to inputs to the support vector machine
 */
void convertColor(Vec3b &color, float *vals) 
{
	// Get lightness
	vals[0] = ((float)color(0) + (float)color(1) + (float)color(2))/600;
	if (vals[0] > 1)
		vals[0] = 1;

	// Get red vs blue
	vals[1] = (float)color(2)/(float)(color(0) + color(2));

	// Get green vs blue (removed to simplify SVM)
	//vals[2] = (float)color(1)/(float)(color(0) + color(1));
}

/*
 * Uses the support vector machine to classify a set of converted values
 */
int ColonyCounter::classifyValues(float* vals) 
{
	assert(trained);

	// Use quantization if present
	if (svmLookup) {
		// NOTE: Hard coded for SVM dim of 2
		assert(SVM_DIM == 2);
		int index = round(vals[1]*(svmQuants[1]-1));
		index *= svmQuants[0];
		index += round(vals[0]*(svmQuants[0]-1));
		return svmLookup[index];
	}

	Mat sampleMat = Mat(1, SVM_DIM, CV_32F, vals);
	float response = svm.predict(sampleMat);
	return response;
}

/*
 * Trains the classifier given a series of images and a matching
 * series of label images which are png files that have certain
 * parts of it marked with either red, cyan or green to indicate
 * total coliform, E. coli or background respectively.
 */
void ColonyCounter::trainClassifier(vector<string> trainPaths, vector<string> labelPaths, int *quants)
{
	// Count training data
	int trainCnt = 0;
	int trainByLabel[3] = { 0, 0, 0 };

	vector<Rect> petriRects;

	// Iterate through all images
	for (int k=0;k<trainPaths.size();k++)
	{
		// Extract rectangles
		Mat img = imread(trainPaths[k]);
		petriRects.push_back(findPetriRect(img));

		// Load label image
		Mat labelImg = imread(labelPaths[k]);
		labelImg = labelImg(petriRects[k]);

		// Count training data
		for (int x=0;x<labelImg.cols;x++)
		{
			for (int y=0;y<labelImg.rows;y++)
			{
				Vec3b c = labelImg.at<Vec3b>(y, x);
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
    Mat labelsMat(trainCnt, 1, CV_32SC1);
    Mat trainingDataMat(trainCnt, SVM_DIM, CV_32FC1);
	int n = 0;
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
				Vec3b c = labelImg.at<Vec3b>(y, x);
				int label = labelColorToIndex(c);
				if (label >= 0)
				{
					Vec3b color = trainImg.at<Vec3b>(y,x);
					float vals[SVM_DIM];
					convertColor(color, vals);

					// Quantize values if necessary
					if (quants) {
						for (int i=0;i<SVM_DIM;i++)
							vals[i] = roundf(vals[i] * quants[i])/quants[i];
					}

					for (int i=0;i<SVM_DIM;i++)
						trainingDataMat.at<float>(n, i) = vals[i];
					labelsMat.at<int>(n,0) = label;
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

/*
 * Perform a low pass filter within an arbitrary 3-channel mask. Returns the
 * low-passed image
 */
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

/*
 * Perform a low pass filter within an arbitrary 1-channel mask. Returns the
 * low-passed image
 */
static Mat lowPass(Mat &img, Mat &mask, int blurSize) {
	Mat mask3C = Mat(img.size(), CV_8UC3, Scalar(0, 0, 0));
	mask3C.setTo(Scalar(255, 255, 255), mask);
	return lowPass3C(img, mask3C, blurSize);
}

/*
 * Finds the background of an image by removing outliers and then blurring to fill
 * in gaps left by the removal of the outliers.
 */
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
	Scalar backCnt = sum(bgmask)/Scalar(255);
	backgroundColor = backTotal/(backCnt[0]);

	return background;
}


/*
 * Preprocesses a petri rectangle, normalizing all colors to 200=white
 * and removing anything outside of the circular mask.
 */
Mat ColonyCounter::preprocessImage(Mat petri) 
{
	Scalar backgroundColor;
	return preprocessImage(petri, backgroundColor);
}

/*
 * Preprocesses a petri rectangle, also returning the original background color
 */
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

/* Calculate the circularity of a contour */
static double calcCircularity(vector<Point> contour) {
	// Check circularity
	double perimeter = arcLength(contour, true);
	double area = contourArea(contour);
	double circularity = 4 * 3.14159265 * area / (perimeter * perimeter);
	return circularity;
}

/*
 * Counts colonies of a particular type on a classified image
 * Removes tiny colonies, joins colonies that are close together
 *  and then keeps appropriate candidate contours.
 */
static vector<vector<Point> > countType(Mat classified, int type) 
{
	// Get type mask
	Mat mask = classified == type;

	// Remove tiny colonies
	Mat kernel = getStructuringElement(MORPH_CROSS, Size(3, 3));
	erode(mask, mask, kernel);
	dilate(mask, mask, kernel);
	erode(mask, mask, kernel);

	// Dilate, erode to join colonies
	kernel = getStructuringElement(MORPH_ELLIPSE, Size(5, 5));
	dilate(mask, mask, kernel);
	erode(mask, mask, kernel);
	dilate(mask, mask, kernel);
	erode(mask, mask, kernel);

	// Find contours
	vector<vector<Point> > contours;
	vector<Vec4i> hierarchy;
	findContours(mask, contours, hierarchy, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_NONE);

	int minArea = 4;
	double minCircularity = 0.2;

	// For each suitable one
	vector<vector<Point> > goodContours;
	for (int i=0;i<contours.size();i++) 
	{
		// Make sure area is sufficiently large
		if (contourArea(contours[i])>=minArea)
		{
			// Check circularity
			double circularity = calcCircularity(contours[i]);
			if (circularity > minCircularity)
				goodContours.push_back(contours[i]);
		}
	}
	return goodContours;
}

/*
 * Classifies an image's pixels using the support vector machine
 */
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

/*
 * Classifies an image, rounding SVM inputs to the specified quantizations
 */
Mat ColonyCounter::classifyImageQuant(Mat img, bool debug, Mat *debugImage, int* quants)
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

			for (int i=0;i<SVM_DIM;i++)
				vals[i] = roundf(vals[i] * quants[i])/quants[i];

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

/*
 * Tests a level of quantization to make sure that values are still
 * appropriately quantized.
 */
void ColonyCounter::testQuantization(Mat img, int* quants)
{
	// Test classification
	int total=0, wrong=0, wrongrb=0, totalrb=0;

	for (int x=0;x<img.cols;x++)
	{
		for (int y=0;y<img.rows;y++)
		{
			Vec3b color = img.at<Vec3b>(y,x);
			float vals[SVM_DIM], valsq[SVM_DIM];

			convertColor(color, vals);
			for (int i=0;i<SVM_DIM;i++)
				valsq[i] = roundf(vals[i] * quants[i])/quants[i];

			int cls = classifyValues(vals);
			int clsq = classifyValues(valsq);

			if (cls != clsq)
				wrong++;
			if (cls != 0)
			{
				if (cls != clsq)
					wrongrb++;
				totalrb++;
			}
			total++;
		}
	}
	printf("wrong=%5d  of %10d\n", wrong, total);
	printf("wrong redblue=%5d  of %10d\n", wrongrb, totalrb);
}

/*
 * Counts colonies on and appropriately classified image, optionally
 * returning debugging information
 */
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
