#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
using namespace cv;
using namespace std;
#include <stdio.h>

#pragma once

class OpenCVActivityContext {
public:
	virtual ~OpenCVActivityContext() {}

	virtual string getParam(int n) = 0;
	virtual int getParamCount() = 0;

	virtual void setReturnValue(string val) = 0;

	virtual void updateScreen(Mat& screen) = 0;

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

	void updateScreen(Mat& screen) {
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

	void updateScreen(Mat& screen) {
		imshow("screen", screen);
		waitKey(500);
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


