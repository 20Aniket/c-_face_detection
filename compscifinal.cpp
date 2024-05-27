#include <filesystem>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <thread>
#include <mutex>
#include <iomanip>
#include <chrono>
#include <ctime>
#include <dispatch/dispatch.h>

using namespace cv;
using namespace std;
namespace fs = std::__fs::filesystem; // Use standard filesystem
mutex dataMutex; // Mutex for thread-safe access to shared data

int maxFaces = 1; // Default maximum number of faces to detect

// Function to detect faces
void detectFaces(Mat &frame, CascadeClassifier &faceCascade, vector<Mat> &capturedFaces) {
    vector<Rect> faces;
    Mat gray;
    cvtColor(frame, gray, COLOR_BGR2GRAY);
    faceCascade.detectMultiScale(gray, faces);
    int faceCount = 0;
    for ( Rect &face : faces) {
        if (faceCount >= maxFaces) break; // Stop if we've reached the max number of faces
        rectangle(frame, face, Scalar(255, 0, 0), 2);
        Mat faceROI = gray(face);
        resize(faceROI, faceROI, Size(200, 200)); // Standardize the face size
        capturedFaces.push_back(faceROI);
        faceCount++;
    }
}

// Function to save captured faces to a folder called images
void saveCapturedFaces( vector<Mat> &capturedFaces,  string &directory) {
    for (size_t i = 0; i < capturedFaces.size(); ++i) {
        auto now = chrono::system_clock::now();
        auto in_time_t = chrono::system_clock::to_time_t(now);
        ostringstream ss;
        ss << put_time(localtime(&in_time_t), "%Y-%m-%d_%H-%M-%S");
        string timestamp = ss.str();
        string filename = directory + "/face_" + timestamp + "_" + to_string(i) + ".png";
        imwrite(filename, capturedFaces[i]);
    }
    cout << "Captured faces saved to " << directory << endl;
}

// Function to clear captured faces and delete associated image files
void clearCapturedFaces(vector<Mat> &capturedFaces,  string &directory) {
    capturedFaces.clear();
    for ( auto& entry : fs::directory_iterator(directory)) {
        if (entry.is_regular_file()) {
            fs::remove(entry.path());
        }
    }
    cout << "Captured faces cleared." << endl;
}

// Function to display the most recent captured faces
void showCapturedFaces( std::vector<cv::Mat>& faces) {
    for ( auto& face : faces) {
        dispatch_async(dispatch_get_main_queue(), ^{
            cv::imshow("Captured Face", face);
            cv::waitKey(0); // Wait for a key press to close the window
        });
    }
}

// Function to set the maximum number of faces to detect
void setMaxFaces(int max) {
    maxFaces = max;
    cout << "Maximum number of faces to detect set to " << maxFaces << endl;
}

//Function used to rename the most recent file in the image directory
void renameFile(string& newName,  string& directory) {
    // Get the list of files in the directory
    vector<string> files;
    for (auto& entry : fs::directory_iterator(directory)) {
        if (entry.is_regular_file()) {
            files.push_back(entry.path());
        }
    }

    // Find the most recent file
    auto compareLastWriteTime = [](string& a,  string& b) {
        return fs::last_write_time(a) > fs::last_write_time(b);
    };
    sort(files.begin(), files.end(), compareLastWriteTime);

    if (!files.empty()) {
        // Rename the most recent file with the provided name
        string mostRecentFile = files[0];
        fs::path filePath(mostRecentFile);
        fs::path newFilePath = filePath.parent_path() / (newName + filePath.extension().string());
        fs::rename(filePath, newFilePath);
        cout << "File renamed to: " << newFilePath << endl;
    } else {
        cout << "No files found in the directory." << endl;
    }
}

int main() {
    CascadeClassifier faceCascade;
    vector<Mat> capturedFaces;
    string dataDirectory = "/Users/aniketsethi/Documents/computer science/final/images";
    bool stopInputThread = false;
    VideoCapture cap(1); // Change to 0 for the default camera
    Mat frame; // Declare the frame variable
    if (!faceCascade.load("/Users/aniketsethi/Documents/computer science/final/haarcascade_frontalface_default.xml")) {
        cout << "Error loading face cascade\n";
        return -1;
    }
    if (!cap.isOpened()) {
        cout << "Error opening video capture\n";
        return -1;
    }
    thread inputThread([&]() {
    string command;
    while (!stopInputThread) {
        this_thread::sleep_for(chrono::seconds(1));
        cout << "Enter command: ";
        getline(cin, command);
        if (command == "capture") {
            {
                lock_guard<mutex> lock(dataMutex);
                saveCapturedFaces(capturedFaces, dataDirectory);
            }
        } else if (command == "clear") {
            {
                lock_guard<mutex> lock(dataMutex);
                clearCapturedFaces(capturedFaces, dataDirectory);
            }
        } else if (command == "show") {
            {
                lock_guard<mutex> lock(dataMutex);
                showCapturedFaces(capturedFaces);
            }
        } else if (command.find("setmax ") == 0) {
            int max = stoi(command.substr(7));
            {
                lock_guard<mutex> lock(dataMutex);
                setMaxFaces(max);
            }
        } else if (command == "exit") {
            stopInputThread = true;
            break;
        } else if (command == "help") {
            int lineLength = 30 * 2 + 4;
            cout << string(lineLength, '-') << endl;
            cout << left << setw(30) << "Command" << "| " << "Description" << endl;
            cout << string(lineLength, '-') << endl;
            cout << left << setw(30) << "capture" << "| " << "Manually capture detected faces" << endl;
            cout << left << setw(30) << "clear" << "| " << "Clear captured faces" << endl;
            cout << left << setw(30) << "show" << "| " << "Show captured faces" << endl;
            cout << left << setw(30) << "setmax <number>" << "| " << "Set max number of faces to detect" << endl;
            cout << left << setw(30) << "exit" << "| " << "Exit the program" << endl;
            cout << string(lineLength, '-') << endl;
        } else if (command.find("rename ") == 0) {
            string newName = command.substr(7); // Extract the new name from the command
            {
                lock_guard<mutex> lock(dataMutex);
                renameFile(newName, dataDirectory);
            }
        } else {
            cout << "Unknown command. Type 'help' for a list of available commands." << endl;
        }
        if (stopInputThread) {
            break;
        }
    }
});
    while (cap.read(frame)) {
        {
            lock_guard<mutex> lock(dataMutex);
            capturedFaces.clear(); // Clear previously detected faces in each frame
            detectFaces(frame, faceCascade, capturedFaces);
        }
        imshow("Face Detection", frame);
        if (waitKey(10) == 27) {
            stopInputThread = true;
            break;
        }
    }
    if (inputThread.joinable()) {
        inputThread.join();
    }
    return 0;
}
