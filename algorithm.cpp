#include <opencv2/core/core.hpp>

#include "Circle.h"
#include "CircleFinder.h"
#include "ColonyCounter.h"
#include "OpenCVActivityContext.h"
#include "svm_table.h"

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

//	// Create affine transform for screen
//	Mat screenMat = getScreenTransform(img.size(), screen->size());
//
//	context.log("Creating screen");
//
//	// Create BGR screen
//	warpAffine(img, *screen, screenMat, screen->size());
//	context.updateScreen();

void analyseECPlate(OpenCVActivityContext& context) {
	context.log("Reading image");

	// Load image
	Mat img = imread(context.getParam(0));
	if (img.empty()) {
		context.setReturnValue("{\"error\":\"Image file not found\"}");
		return;
	}

	context.updateScreen(img);

	context.log("Finding petri image");

	// Find petri img
	Rect petriRect = findPetriRect(img);

	if (petriRect.height == 0) {
		context.log("Circle not found");
		context.setReturnValue("{\"error\":\"EC Plate not detected\"}");
		return;
	}

	Mat petri = img(petriRect);
	context.updateScreen(petri);

	context.log("Loading training");

	ColonyCounter colonyCounter;
	colonyCounter.loadTrainingQuantized(svmLookup, svmQuants);

	context.log("Preprocessing image");

	// Preprocess image
	petri = colonyCounter.preprocessImage(petri);
	context.updateScreen(petri);

	if (context.getParamCount() >= 3) {
		imwrite(context.getParam(2), petri);
	}

	context.log("Classifying image");

	// Classify image
	Mat debugImage;
	Mat classified = colonyCounter.classifyImage(petri, true, &debugImage);
	context.updateScreen(debugImage);

	context.log("Counting colonies");

	// Count colonies
	int red, blue;
	colonyCounter.countColonies(classified, red, blue, true, &debugImage);
	context.updateScreen(debugImage);

	if (context.getParamCount() >= 2) {
		imwrite(context.getParam(1), debugImage);
	}

	context.log("Showing results");

	// Pause
	sleep(2);

	context.log("Done");

	context.setReturnValue(format("{\"tc\": %d, \"ecoli\": %d, \"algorithm\": \"2013-03-15\"}", red, blue));
}
