#include <opencv2/core/core.hpp>

#include "Circle.h"
#include "CircleFinder.h"
#include "ColonyCounter.h"
#include "OpenCVActivityContext.h"
#include "svm_params.h"

#include <unistd.h>

using namespace cv;
using namespace std;

static Mat getScreenTransform(Size image, Size screen) {
	Point2f srcTri[3];
	Point2f dstTri[3];

	// Set 3 points to calculate the Affine Transform
	srcTri[0] = Point2f(0, 0);
	srcTri[1] = Point2f(image.width - 1, 0);
	srcTri[2] = Point2f(0, image.height - 1);

	// Calculate scale
	double scale = min(screen.width * 1.0 / image.width, screen.height * 1.0 / image.height);

	dstTri[0] = Point2f(0, 0);
	dstTri[1] = srcTri[1]*scale;
	dstTri[2] = srcTri[2]*scale;

	return getAffineTransform(srcTri, dstTri);
}

static Point transformPoint(Point pt, Mat matrix) {
	Mat pt2 = matrix * Mat(Vec3d(pt.x, pt.y, 1));
	return Point(pt2.at<double>(0), pt2.at<double>(1));
}

static double transformScalar(double val, Mat matrix) {
	return val * matrix.at<double>(0,0);
}

void analyseECPlate(OpenCVActivityContext& context) {
	// Load image
	Mat img = imread(context.getParam(0));

	Ptr<Mat> screen = context.getScreen();

	// Create affine transform for screen
	Mat screenMat = getScreenTransform(img.size(), screen->size());

	// Create BGR screen
	warpAffine(img, *screen, screenMat, screen->size());
	context.updateScreen();

	// Find petri img
	Rect petriRect = findPetriRect(img);
	Mat petri = img(petriRect);

	// Draw circle on screen
	Point center = transformPoint(Point((petriRect.tl()+petriRect.br())*0.5), screenMat);
	int radius = transformScalar(petriRect.height/2, screenMat);
	circle(*screen, center, radius, Scalar(0, 0, 255), 2);
	context.updateScreen();

	ColonyCounter colonyCounter;
	// Read from string, not disk : 
	//colonyCounter.loadTraining("svm_params.yml");
	colonyCounter.loadTrainingString(svm_params);

	// Preprocess image
	petri = colonyCounter.preprocessImage(petri);

	if (context.getParamCount() >= 3) {
		imwrite(context.getParam(2), petri);
	}

	// Classify image
	Mat classified = colonyCounter.classifyImage(petri);

	// Count colonies
	int red, blue;
	Mat debugImage;
	colonyCounter.countColonies(classified, red, blue, true, &debugImage);

	if (context.getParamCount() >= 2) {
		imwrite(context.getParam(1), debugImage);
	}

	// Show completed
	circle(*screen, center, radius, Scalar(0, 255, 0), 6);
	context.updateScreen();

	// Pause
	sleep(2);

	context.setReturnValue(format("{\"tc\": %d, \"ecoli\": %d, \"algorithm\": \"2013-02-14\"}", red, blue));
}
