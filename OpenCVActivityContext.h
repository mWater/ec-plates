#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
using namespace cv;
using namespace std;

#pragma once

class OpenCVActivityContext {
public:
	virtual ~OpenCVActivityContext() {}

	virtual string getParam(int n) = 0;
	virtual int getParamCount() = 0;

	virtual void setReturnValue(string val) = 0;

	// Gets BGR screen. Must call updateScreen to propagate change
	virtual Ptr<Mat> getScreen() = 0;
	virtual void updateScreen() = 0;

	virtual void log(string msg) = 0;

	virtual bool isAborted() = 0;
};

class ConsoleOpenCVActivityContext : public OpenCVActivityContext {
public:
	ConsoleOpenCVActivityContext(int argc, char* argv[]) :
		argc(argc), argv(argv) {
	}

	~ConsoleOpenCVActivityContext() {
	}

	string getParam(int n) {
		return argv[n];
	}

	int getParamCount() {
		return argc;
	}

	void setReturnValue(string val) {
		returnValue = val;
	}

	Ptr<Mat> getScreen() {
		return NULL;
	}

	void updateScreen() {
	}

	void log(string msg) {
		printf("%s\n", msg.c_str());
	}

	bool isAborted() {
		return false;
	}

	string returnValue;

private:
	int argc;
	char** argv;
};

class DesktopOpenCVActivityContext : public OpenCVActivityContext {
public:
	DesktopOpenCVActivityContext(int argc, char* argv[]) :
		argc(argc), argv(argv) {

		// Create Mat
		screen = new Mat(Size(800, 480), CV_8UC3);
		updateScreen();
	}

	~DesktopOpenCVActivityContext() {
	}

	string getParam(int n) {
		return argv[n];
	}

	int getParamCount() {
		return argc;
	}

	void setReturnValue(string val) {
		returnValue = val;
	}

	Ptr<Mat> getScreen() {
		return screen;
	}

	void updateScreen() {
		imshow("screen", *screen);
	}

	void log(string msg) {
		printf("%s\n", msg.c_str());
	}

	bool isAborted() {
		return false;
	}

	string returnValue;

private:
	int argc;
	char** argv;
	Ptr<Mat> screen;
};


