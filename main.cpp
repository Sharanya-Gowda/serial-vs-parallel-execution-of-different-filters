#include <stdio.h>    // For printf
#include <stdlib.h>   // For general utilities
#include <omp.h>      // For OpenMP functions
#include <opencv2/opencv.hpp> // For OpenCV
#include <string.h>   // For sprintf
#include <fstream>    // For file I/O
#include <cmath>      // For sqrt()

// Use the cv namespace
using namespace cv;

/*
 * =================================================================
 * Section 1: Single-Kernel Convolution Functions
 * (Used for Sharpen, Gaussian, Edge_Detection)
 * =================================================================
 */

/**
 * @brief Applies a 3x3 convolution kernel in SERIAL.
 */
void apply_serial_filter(Mat& src, Mat& dst, float kernel[3][3]) {
    int width = src.cols;
    int height = src.rows;
    int nChannels = src.channels();
    int step = src.step;
    unsigned char* src_data = src.data;
    unsigned char* dst_data = dst.data;

    for (int y = 1; y < height - 1; y++) {
        for (int x = 1; x < width - 1; x++) {
            for (int c = 0; c < nChannels; c++) {
                float sum = 0.0;
                for (int ky = -1; ky <= 1; ky++) {
                    for (int kx = -1; kx <= 1; kx++) {
                        unsigned char val = src_data[(y + ky) * step + (x + kx) * nChannels + c];
                        sum += val * kernel[ky + 1][kx + 1];
                    }
                }
                if (sum < 0) sum = 0;
                if (sum > 255) sum = 255;
                dst_data[y * step + x * nChannels + c] = (unsigned char)sum;
            }
        }
    }
}

/**
 * @brief Applies a 3x3 convolution kernel in PARALLEL.
 */
void apply_parallel_filter(Mat& src, Mat& dst, float kernel[3][3]) {
    int width = src.cols;
    int height = src.rows;
    int nChannels = src.channels();
    int step = src.step;
    unsigned char* src_data = src.data;
    unsigned char* dst_data = dst.data;

    #pragma omp parallel
    {
        #pragma omp master
        {
            printf("Running with %d threads.\n", omp_get_num_threads());
        }
        
        #pragma omp for
        for (int y = 1; y < height - 1; y++) {
            for (int x = 1; x < width - 1; x++) {
                for (int c = 0; c < nChannels; c++) {
                    float sum = 0.0;
                    for (int ky = -1; ky <= 1; ky++) {
                        for (int kx = -1; kx <= 1; kx++) {
                            unsigned char val = src_data[(y + ky) * step + (x + kx) * nChannels + c];
                            sum += val * kernel[ky + 1][kx + 1];
                        }
                    }
                    if (sum < 0) sum = 0;
                    if (sum > 255) sum = 255;
                    dst_data[y * step + x * nChannels + c] = (unsigned char)sum;
                }
            }
        }
    } // end parallel region
}


/*
 * =================================================================
 * Section 2: NEW Sobel Filter Functions
 * (These are more complex, using two kernels)
 * =================================================================
 */

// Define the Sobel kernels
float Gx[3][3] = {
    {-1, 0, 1},
    {-2, 0, 2},
    {-1, 0, 1}
};

float Gy[3][3] = {
    {-1, -2, -1},
    { 0,  0,  0},
    { 1,  2,  1}
};


/**
 * @brief Applies the Sobel edge detection algorithm in SERIAL.
 */
void apply_sobel_serial(Mat& src, Mat& dst) {
    int width = src.cols;
    int height = src.rows;
    int nChannels = src.channels();
    int step = src.step;
    unsigned char* src_data = src.data;
    unsigned char* dst_data = dst.data;

    for (int y = 1; y < height - 1; y++) {
        for (int x = 1; x < width - 1; x++) {
            for (int c = 0; c < nChannels; c++) {
                
                float sumX = 0.0;
                float sumY = 0.0;
                
                // Apply both kernels
                for (int ky = -1; ky <= 1; ky++) {
                    for (int kx = -1; kx <= 1; kx++) {
                        unsigned char val = src_data[(y + ky) * step + (x + kx) * nChannels + c];
                        sumX += val * Gx[ky + 1][kx + 1];
                        sumY += val * Gy[ky + 1][kx + 1];
                    }
                }
                
                // Calculate magnitude and clamp
                float magnitude = sqrt(sumX * sumX + sumY * sumY);
                if (magnitude < 0) magnitude = 0;
                if (magnitude > 255) magnitude = 255;
                
                dst_data[y * step + x * nChannels + c] = (unsigned char)magnitude;
            }
        }
    }
}


/**
 * @brief Applies the Sobel edge detection algorithm in PARALLEL.
 */
void apply_sobel_parallel(Mat& src, Mat& dst) {
    int width = src.cols;
    int height = src.rows;
    int nChannels = src.channels();
    int step = src.step;
    unsigned char* src_data = src.data;
    unsigned char* dst_data = dst.data;
    
    #pragma omp parallel
    {
        #pragma omp master
        {
            printf("Running with %d threads.\n", omp_get_num_threads());
        }

        #pragma omp for
        for (int y = 1; y < height - 1; y++) {
            for (int x = 1; x < width - 1; x++) {
                for (int c = 0; c < nChannels; c++) {
                    
                    float sumX = 0.0;
                    float sumY = 0.0;
                    
                    for (int ky = -1; ky <= 1; ky++) {
                        for (int kx = -1; kx <= 1; kx++) {
                            unsigned char val = src_data[(y + ky) * step + (x + kx) * nChannels + c];
                            sumX += val * Gx[ky + 1][kx + 1];
                            sumY += val * Gy[ky + 1][kx + 1];
                        }
                    }
                    
                    float magnitude = sqrt(sumX * sumX + sumY * sumY);
                    if (magnitude < 0) magnitude = 0;
                    if (magnitude > 255) magnitude = 255;
                    
                    dst_data[y * step + x * nChannels + c] = (unsigned char)magnitude;
                }
            }
        }
    } // end parallel region
}


/*
 * =================================================================
 * Section 3: Test-Runner Functions
 * =================================================================
 */

/**
 * @brief Runs tests for a SINGLE-KERNEL filter.
 */
void run_filter_test(Mat& src, float kernel[3][3], const char* filter_name, std::ofstream& file_stream) {
    
    printf("\n--- Testing Filter: %s ---\n", filter_name);
    Mat serial_out = Mat::zeros(src.size(), src.type());
    Mat parallel_out = Mat::zeros(src.size(), src.type());

    // Serial Test
    double start_serial = omp_get_wtime();
    apply_serial_filter(src, serial_out, kernel);
    double end_serial = omp_get_wtime();
    double serial_time = end_serial - start_serial;
    
    // Parallel Test
    double start_parallel = omp_get_wtime();
    apply_parallel_filter(src, parallel_out, kernel);
    double end_parallel = omp_get_wtime();
    double parallel_time = end_parallel - start_parallel;

    // Save Results
    char serial_filename[100];
    char parallel_filename[100];
    sprintf(serial_filename, "%s_serial.jpg", filter_name);
    sprintf(parallel_filename, "%s_parallel.jpg", filter_name);
    imwrite(serial_filename, serial_out);
    imwrite(parallel_filename, parallel_out);

    // Analysis
    double speedup = serial_time / parallel_time;
    printf("Serial Time:   %f s\n", serial_time);
    printf("Parallel Time: %f s\n", parallel_time);
    printf("Speedup:       %.2fx\n", speedup);
    file_stream << filter_name << "," << serial_time << "," << parallel_time << "," << speedup << "\n";
    printf("Results saved to %s and %s\n", serial_filename, parallel_filename);
}


/**
 * @brief NEW: Runs tests for the SOBEL filter.
 */
void run_sobel_test(Mat& src, const char* filter_name, std::ofstream& file_stream) {
    
    printf("\n--- Testing Filter: %s ---\n", filter_name);
    Mat serial_out = Mat::zeros(src.size(), src.type());
    Mat parallel_out = Mat::zeros(src.size(), src.type());

    // Serial Test
    double start_serial = omp_get_wtime();
    apply_sobel_serial(src, serial_out);
    double end_serial = omp_get_wtime();
    double serial_time = end_serial - start_serial;
    
    // Parallel Test
    double start_parallel = omp_get_wtime();
    apply_sobel_parallel(src, parallel_out);
    double end_parallel = omp_get_wtime();
    double parallel_time = end_parallel - start_parallel;

    // Save Results
    char serial_filename[100];
    char parallel_filename[100];
    sprintf(serial_filename, "%s_serial.jpg", filter_name);
    sprintf(parallel_filename, "%s_parallel.jpg", filter_name);
    imwrite(serial_filename, serial_out);
    imwrite(parallel_filename, parallel_out);

    // Analysis
    double speedup = serial_time / parallel_time;
    printf("Serial Time:   %f s\n", serial_time);
    printf("Parallel Time: %f s\n", parallel_time);
    printf("Speedup:       %.2fx\n", speedup);
    file_stream << filter_name << "," << serial_time << "," << parallel_time << "," << speedup << "\n";
    printf("Results saved to %s and %s\n", serial_filename, parallel_filename);
}


/*
 * =================================================================
 * Section 4: Main Function
 * =================================================================
 */

int main(int argc, char** argv) {

    // --- 1. SETUP ---
    const char* image_path = "test_image.jpg"; 
    Mat img = imread(image_path, IMREAD_COLOR);

    if (img.empty()) {
        printf("Error: Could not load image at %s\n", image_path);
        return -1;
    }
    printf("Image loaded: %d x %d\n", img.cols, img.rows);
    printf("Total threads available: %d\n", omp_get_max_threads());

    // Open results file
    std::ofstream result_file("results.csv");
    if (!result_file.is_open()) {
        printf("Error: Could not open results.csv for writing.\n");
        return -1;
    }
    result_file << "Filter,SerialTime,ParallelTime,Speedup\n";

    // --- 2. DEFINE KERNELS ---
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

    float edge_kernel[3][3] = { // Laplacian
        { 0,  1,  0},
        { 1, -4,  1},
        { 0,  1,  0}
    };

    // --- 3. RUN ALL TESTS ---
    run_filter_test(img, sharpen_kernel, "Sharpen", result_file);
    run_filter_test(img, gaussian_kernel, "Gaussian_Blur", result_file);
    run_filter_test(img, edge_kernel, "Edge_Detection", result_file);
    
    // NEW: Add the Sobel test
    run_sobel_test(img, "Sobel", result_file);
    
    // --- 4. CLEANUP ---
    result_file.close();
    printf("\nAll filters applied.\n");
    return 0;
}