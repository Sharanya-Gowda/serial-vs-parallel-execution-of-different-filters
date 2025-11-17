#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <opencv2/opencv.hpp>
#include <string.h>
#include <fstream>
#include <cmath>
#include <iostream>
#include <vector>

using namespace cv;
using namespace std;

// --- GLOBAL KERNELS ---
float sharpen_kernel[3][3] = {
    { 0, -1,  0},
    {-1,  5, -1},
    { 0, -1,  0}
};

float gaussian_kernel[3][3] = {
    {1.0/16.0, 2.0/16.0, 1.0/16.0},
    {2.0/16.0, 4.0/16.0, 2.0/16.0},
    {1.0/16.0, 2.0/16.0, 1.0/16.0}
};

float edge_kernel[3][3] = {
    { 0,  1,  0},
    { 1, -4,  1},
    { 0,  1,  0}
};

float Gx[3][3] = {{-1, 0, 1}, {-2, 0, 2}, {-1, 0, 1}};
float Gy[3][3] = {{-1, -2, -1}, { 0,  0,  0}, { 1,  2,  1}};

// =================================================================
// FILTER FUNCTIONS (Processing Logic)
// =================================================================

void apply_serial_filter(Mat& src, Mat& dst, float kernel[3][3]) {
    int width = src.cols; int height = src.rows; int nChannels = src.channels(); int step = src.step;
    unsigned char* src_data = src.data; unsigned char* dst_data = dst.data;

    for (int y = 1; y < height - 1; y++) {
        for (int x = 1; x < width - 1; x++) {
            for (int c = 0; c < nChannels; c++) {
                float sum = 0.0;
                for (int ky = -1; ky <= 1; ky++) {
                    for (int kx = -1; kx <= 1; kx++) {
                        sum += src_data[(y + ky) * step + (x + kx) * nChannels + c] * kernel[ky + 1][kx + 1];
                    }
                }
                if (sum < 0) sum = 0; if (sum > 255) sum = 255;
                dst_data[y * step + x * nChannels + c] = (unsigned char)sum;
            }
        }
    }
}

void apply_parallel_filter(Mat& src, Mat& dst, float kernel[3][3]) {
    int width = src.cols; int height = src.rows; int nChannels = src.channels(); int step = src.step;
    unsigned char* src_data = src.data; unsigned char* dst_data = dst.data;

    #pragma omp parallel for
    for (int y = 1; y < height - 1; y++) {
        for (int x = 1; x < width - 1; x++) {
            for (int c = 0; c < nChannels; c++) {
                float sum = 0.0;
                for (int ky = -1; ky <= 1; ky++) {
                    for (int kx = -1; kx <= 1; kx++) {
                        sum += src_data[(y + ky) * step + (x + kx) * nChannels + c] * kernel[ky + 1][kx + 1];
                    }
                }
                if (sum < 0) sum = 0; if (sum > 255) sum = 255;
                dst_data[y * step + x * nChannels + c] = (unsigned char)sum;
            }
        }
    }
}

void apply_sobel_serial(Mat& src, Mat& dst) {
    int width = src.cols; int height = src.rows; int nChannels = src.channels(); int step = src.step;
    unsigned char* src_data = src.data; unsigned char* dst_data = dst.data;

    for (int y = 1; y < height - 1; y++) {
        for (int x = 1; x < width - 1; x++) {
            for (int c = 0; c < nChannels; c++) {
                float sumX = 0.0, sumY = 0.0;
                for (int ky = -1; ky <= 1; ky++) {
                    for (int kx = -1; kx <= 1; kx++) {
                        unsigned char val = src_data[(y + ky) * step + (x + kx) * nChannels + c];
                        sumX += val * Gx[ky + 1][kx + 1];
                        sumY += val * Gy[ky + 1][kx + 1];
                    }
                }
                float mag = sqrt(sumX*sumX + sumY*sumY);
                if (mag > 255) mag = 255;
                dst_data[y * step + x * nChannels + c] = (unsigned char)mag;
            }
        }
    }
}

void apply_sobel_parallel(Mat& src, Mat& dst) {
    int width = src.cols; int height = src.rows; int nChannels = src.channels(); int step = src.step;
    unsigned char* src_data = src.data; unsigned char* dst_data = dst.data;

    #pragma omp parallel for
    for (int y = 1; y < height - 1; y++) {
        for (int x = 1; x < width - 1; x++) {
            for (int c = 0; c < nChannels; c++) {
                float sumX = 0.0, sumY = 0.0;
                for (int ky = -1; ky <= 1; ky++) {
                    for (int kx = -1; kx <= 1; kx++) {
                        unsigned char val = src_data[(y + ky) * step + (x + kx) * nChannels + c];
                        sumX += val * Gx[ky + 1][kx + 1];
                        sumY += val * Gy[ky + 1][kx + 1];
                    }
                }
                float mag = sqrt(sumX*sumX + sumY*sumY);
                if (mag > 255) mag = 255;
                dst_data[y * step + x * nChannels + c] = (unsigned char)mag;
            }
        }
    }
}

// =================================================================
// MODE 1: IMAGE PROCESSING (Standard Benchmark)
// =================================================================
void process_image_mode() {
    string image_path;
    cout << "Enter image filename (e.g., test_image2.jpg): ";
    cin >> image_path;

    Mat img = imread(image_path, IMREAD_COLOR);
    if (img.empty()) { cout << "Error loading image.\n"; return; }

    ofstream f("results_image.csv");
    f << "Filter,SerialTime,ParallelTime,Speedup\n";

    // Helper lambda to run test and write to file
    auto run_test = [&](const char* name, float k[3][3], bool is_sobel) {
        Mat outS = Mat::zeros(img.size(), img.type());
        Mat outP = Mat::zeros(img.size(), img.type());
        
        double t1 = omp_get_wtime();
        if(is_sobel) apply_sobel_serial(img, outS); else apply_serial_filter(img, outS, k);
        double ts = omp_get_wtime() - t1;

        double t2 = omp_get_wtime();
        if(is_sobel) apply_sobel_parallel(img, outP); else apply_parallel_filter(img, outP, k);
        double tp = omp_get_wtime() - t2;

        char fname[100]; sprintf(fname, "%s_result.jpg", name); imwrite(fname, outP);
        cout << name << ": Serial=" << ts << "s, Parallel=" << tp << "s, Speedup=" << ts/tp << "x\n";
        f << name << "," << ts << "," << tp << "," << ts/tp << "\n";
    };

    run_test("Sharpen", sharpen_kernel, false);
    run_test("Gaussian", gaussian_kernel, false);
    run_test("Edge", edge_kernel, false);
    run_test("Sobel", NULL, true);
    
    f.close();
    cout << "Image processing complete.\n";
}

// =================================================================
// MODE 2: AUTOMATED VIDEO BENCHMARK (The requested change)
// =================================================================
void process_video_mode() {
    string video_path;
    cout << "Enter video filename (e.g., test_video.mp4): ";
    cin >> video_path;

    VideoCapture cap(video_path);
    if (!cap.isOpened()) { cout << "Error opening video file.\n"; return; }

    int w = (int)cap.get(CAP_PROP_FRAME_WIDTH);
    int h = (int)cap.get(CAP_PROP_FRAME_HEIGHT);
    double fps = cap.get(CAP_PROP_FPS);
    if (fps <= 0) fps = 24.0;

    // 1. Create Writers for MP4
    // 'm','p','4','v' is a standard code for MP4 container
    int fourcc = VideoWriter::fourcc('m', 'p', '4', 'v'); 
    
    VideoWriter w_sharp("out_sharpen.mp4", fourcc, fps, Size(w, h));
    VideoWriter w_gauss("out_gaussian.mp4", fourcc, fps, Size(w, h));
    VideoWriter w_edge("out_edge.mp4", fourcc, fps, Size(w, h));
    VideoWriter w_sobel("out_sobel.mp4", fourcc, fps, Size(w, h));

    // 2. Variables for Timings
    double t_sharp_s = 0, t_sharp_p = 0;
    double t_gauss_s = 0, t_gauss_p = 0;
    double t_edge_s = 0, t_edge_p = 0;
    double t_sobel_s = 0, t_sobel_p = 0;

    Mat frame, temp_dest;
    temp_dest = Mat::zeros(Size(w, h), CV_8UC3);
    int frame_count = 0;

    cout << "Starting Batch Processing (This applies Serial AND Parallel for every frame)..." << endl;

    while (true) {
        cap >> frame;
        if (frame.empty()) break;

        double start, end;

        // --- 1. Sharpen ---
        // Measure Serial
        start = omp_get_wtime(); apply_serial_filter(frame, temp_dest, sharpen_kernel); end = omp_get_wtime();
        t_sharp_s += (end - start);
        // Measure Parallel (and save this one)
        start = omp_get_wtime(); apply_parallel_filter(frame, temp_dest, sharpen_kernel); end = omp_get_wtime();
        t_sharp_p += (end - start);
        w_sharp.write(temp_dest);

        // --- 2. Gaussian ---
        start = omp_get_wtime(); apply_serial_filter(frame, temp_dest, gaussian_kernel); end = omp_get_wtime();
        t_gauss_s += (end - start);
        start = omp_get_wtime(); apply_parallel_filter(frame, temp_dest, gaussian_kernel); end = omp_get_wtime();
        t_gauss_p += (end - start);
        w_gauss.write(temp_dest);

        // --- 3. Edge ---
        start = omp_get_wtime(); apply_serial_filter(frame, temp_dest, edge_kernel); end = omp_get_wtime();
        t_edge_s += (end - start);
        start = omp_get_wtime(); apply_parallel_filter(frame, temp_dest, edge_kernel); end = omp_get_wtime();
        t_edge_p += (end - start);
        w_edge.write(temp_dest);

        // --- 4. Sobel ---
        start = omp_get_wtime(); apply_sobel_serial(frame, temp_dest); end = omp_get_wtime();
        t_sobel_s += (end - start);
        start = omp_get_wtime(); apply_sobel_parallel(frame, temp_dest); end = omp_get_wtime();
        t_sobel_p += (end - start);
        w_sobel.write(temp_dest);

        frame_count++;
        if(frame_count % 10 == 0) cout << "Processed " << frame_count << " frames..." << endl;
    }

    // 3. Save Results to CSV
    ofstream csv("results_video_benchmark.csv");
    csv << "Filter,TotalSerialTime(s),TotalParallelTime(s),Speedup\n";
    csv << "Sharpen," << t_sharp_s << "," << t_sharp_p << "," << t_sharp_s/t_sharp_p << "\n";
    csv << "Gaussian," << t_gauss_s << "," << t_gauss_p << "," << t_gauss_s/t_gauss_p << "\n";
    csv << "Edge," << t_edge_s << "," << t_edge_p << "," << t_edge_s/t_edge_p << "\n";
    csv << "Sobel," << t_sobel_s << "," << t_sobel_p << "," << t_sobel_s/t_sobel_p << "\n";
    csv.close();

    cap.release();
    w_sharp.release(); w_gauss.release(); w_edge.release(); w_sobel.release();

    cout << "\n=============================================" << endl;
    cout << "VIDEO BENCHMARK COMPLETE" << endl;
    cout << "1. Output Videos saved as: out_sharpen.mp4, out_gaussian.mp4, etc." << endl;
    cout << "2. Timing Data saved to: results_video_benchmark.csv" << endl;
    cout << "=============================================" << endl;
}

int main(int argc, char** argv) {
    int mode;
    cout << "1. Process IMAGE\n2. Process VIDEO (Auto Benchmark All Filters)\nChoice: ";
    cin >> mode;
    if (mode == 1) process_image_mode();
    else if (mode == 2) process_video_mode();
    else cout << "Invalid choice.\n";
    return 0;
}