#include <opencv2/core/core.hpp>

#include "Circle.h"
#include "CircleFinder.h"
#include "ColonyCounter.h"
#include "OpenCVActivityContext.h"
#include "svm_table.h"

#include <unistd.h>

using namespace cv;
using namespace std;

/**
 * Analyzes an EC Compact Dry Plate.
 *
 * Algorithm steps are as follows:
 *
 * 1) Find the circle of the petri dish
 * 2) Preprocess the image
 * 3) Categorize pixels using a Support Vector Machine that has been trained
 * 4) Filter out small or unusually shaped colonies
 * 5) Return a final count
 */
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

	// Find petri disk rectangle
	Rect petriRect = findPetriRect(img);

	if (petriRect.height == 0) {
		context.log("Circle not found");
		context.setReturnValue("{\"error\":\"EC Plate not detected\"}");
		return;
	}

	// Update screen
	Mat petri = img(petriRect);
	context.updateScreen(petri);

	context.log("Loading training");

	// Create the colony counter
	ColonyCounter colonyCounter;
	colonyCounter.loadTrainingQuantized(svmLookup, svmQuants);

	context.log("Preprocessing image");

	// Preprocess image
	petri = colonyCounter.preprocessImage(petri);
	context.updateScreen(petri);

	// Optionally write out preprocessed image
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

	// Optionally write out colony image
	if (context.getParamCount() >= 2) {
		imwrite(context.getParam(1), debugImage);
	}

	context.log("Showing results");

	// Pause to give the user time to see the results
	sleep(2);

	context.log("Done");

	context.setReturnValue(format("{\"tc\": %d, \"ecoli\": %d, \"algorithm\": \"2013-03-19\"}", red, blue));
}
