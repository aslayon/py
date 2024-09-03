#include <iostream>
#include <fstream>
#include <vector>
#include <windows.h>
#include <cmath>
#include <algorithm> 



class ModifyImg {
public:
    BITMAPFILEHEADER fileHeader;
    BITMAPINFOHEADER infoHeader;
    std::ifstream bmpFile;

    ModifyImg(const std::string& filename) {
        bmpFile.open(filename, std::ios::binary);
        if (!bmpFile) {
            throw std::runtime_error("파일을 열 수 없습니다.");
        }
        bmpFile.read(reinterpret_cast<char*>(&fileHeader), sizeof(fileHeader));
        bmpFile.read(reinterpret_cast<char*>(&infoHeader), sizeof(infoHeader));
    }

    ~ModifyImg() {
        if (bmpFile.is_open()) {
            bmpFile.close();
        }
    }

    void invertColors() {
        int width = infoHeader.biWidth;
        int height = std::abs(infoHeader.biHeight);
        int bytesPerPixel = infoHeader.biBitCount / 8;
        int rowStride = ((width * bytesPerPixel + 3) / 4) * 4;

        std::vector<BYTE> rowData(rowStride);
        std::ofstream outFile("output_GRAY.bmp", std::ios::binary);

        outFile.write(reinterpret_cast<const char*>(&fileHeader), sizeof(BITMAPFILEHEADER));
        outFile.write(reinterpret_cast<const char*>(&infoHeader), sizeof(BITMAPINFOHEADER));

        bmpFile.seekg(fileHeader.bfOffBits, std::ios::beg); //픽셀 데이터 시작 위치로.

        bool isTopDown = infoHeader.biHeight < 0;
        for (int y = 0; y < height; ++y) {
            int readRow = isTopDown ? y : (height - 1 - y);
            bmpFile.seekg(fileHeader.bfOffBits + readRow * rowStride, std::ios::beg); //픽셀 시작위치. 에서 몇번째 ?
            bmpFile.read(reinterpret_cast<char*>(rowData.data()), rowStride); // 읽어오고.
            
            // 색상 반전
            for (int x = 0; x < width; ++x) {
                BYTE r = rowData[x * bytesPerPixel + 2];
                BYTE g = rowData[x * bytesPerPixel + 1];
                BYTE b = rowData[x * bytesPerPixel + 0];

                BYTE gray = (r + g + b) / 3;

                rowData[x * bytesPerPixel + 0] = gray;
                rowData[x * bytesPerPixel + 1] = gray;
                rowData[x * bytesPerPixel + 2] = gray;
            }
            
            int writeRow = isTopDown ? y : (height - 1 - y);
            outFile.seekp(fileHeader.bfOffBits + writeRow * rowStride, std::ios::beg);
            outFile.write(reinterpret_cast<const char*>(rowData.data()), rowStride);
        }

        outFile.close();
    }

    void avgfilter(const std::string& filename) {
        int width = infoHeader.biWidth;
        int height = std::abs(infoHeader.biHeight);
        int bytesPerPixel = infoHeader.biBitCount / 8;
        int rowStride = ((width * bytesPerPixel + 3) / 4) * 4;

        std::vector<BYTE> rowData(rowStride * height);  // 이미지 전체를 위한 벡터
        std::vector<BYTE> tempRowData(rowStride * height);  // 임시 데이터 저장을 위한 벡터
        std::ofstream outFile(filename, std::ios::binary);

        outFile.write(reinterpret_cast<const char*>(&fileHeader), sizeof(BITMAPFILEHEADER));
        outFile.write(reinterpret_cast<const char*>(&infoHeader), sizeof(BITMAPINFOHEADER));

        bmpFile.seekg(fileHeader.bfOffBits, std::ios::beg);
        bmpFile.read(reinterpret_cast<char*>(rowData.data()), rowData.size());

        bool isTopDown = infoHeader.biHeight < 0;

        // 임시 데이터를 rowData에서 복사
        std::copy(rowData.begin(), rowData.end(), tempRowData.begin());

        // 3x3 블러 필터 적용
        for (int y = 1; y < height - 1; ++y) {
            for (int x = 1; x < width - 1; ++x) {
                int sumR = 0, sumG = 0, sumB = 0;
                int count = 0;

                for (int dy = -1; dy <= 1; ++dy) {
                    for (int dx = -1; dx <= 1; ++dx) {
                        if (dx == 0 && dy == 0) continue;
                        int nx = x + dx;
                        int ny = y + dy;
                        int offset = (ny * rowStride) + (nx * bytesPerPixel);

                        sumB += tempRowData[offset + 0];
                        sumG += tempRowData[offset + 1];
                        sumR += tempRowData[offset + 2];
                        ++count;
                    }
                }

                BYTE avgB = static_cast<BYTE>(sumB / count);
                BYTE avgG = static_cast<BYTE>(sumG / count);
                BYTE avgR = static_cast<BYTE>(sumR / count);

                int centerOffset = (y * rowStride) + (x * bytesPerPixel);
                rowData[centerOffset + 0] = avgB;
                rowData[centerOffset + 1] = avgG;
                rowData[centerOffset + 2] = avgR;
            }
        }

        // 결과를 파일에 쓰기
        outFile.seekp(fileHeader.bfOffBits, std::ios::beg);
        outFile.write(reinterpret_cast<const char*>(rowData.data()), rowData.size());

        outFile.close();
    }

    void sobel1(const std::string& filename) {
        int width = infoHeader.biWidth;
        int height = std::abs(infoHeader.biHeight);
        int bytesPerPixel = infoHeader.biBitCount / 8;
        int rowStride = ((width * bytesPerPixel + 3) / 4) * 4;

        std::vector<BYTE> rowData(rowStride * height);  // 이미지 전체를 위한 벡터
        std::vector<BYTE> tempRowData(rowStride * height);  // 임시 데이터 저장을 위한 벡터
        std::ofstream outFile(filename, std::ios::binary);

        outFile.write(reinterpret_cast<const char*>(&fileHeader), sizeof(BITMAPFILEHEADER));
        outFile.write(reinterpret_cast<const char*>(&infoHeader), sizeof(BITMAPINFOHEADER));

        bmpFile.seekg(fileHeader.bfOffBits, std::ios::beg);
        bmpFile.read(reinterpret_cast<char*>(rowData.data()), rowData.size());

        bool isTopDown = infoHeader.biHeight < 0;

        // 임시 데이터를 rowData에서 복사
        std::copy(rowData.begin(), rowData.end(), tempRowData.begin());

        // 3x3 블러 필터 적용
        for (int y = 1; y < height - 1; ++y) {
            for (int x = 1; x < width - 1; ++x) {
                int sumR = 0, sumG = 0, sumB = 0;
                int count = 0;

                for (int dy = -1; dy <= 1; ++dy) {
                    for (int dx = -1; dx <= 1; ++dx) {
                        
                        int nx = x + dx;
                        int ny = y + dy;
                        int offset = (ny * rowStride) + (nx * bytesPerPixel);
                        // 필터 적용: -1, 0, 1
                        int filterValue = (dy == -1) ? -1 : (dy == 1) ? 1 : 0;

                        sumB += tempRowData[offset + 0] * filterValue;
                        sumG += tempRowData[offset + 1] * filterValue;
                        sumR += tempRowData[offset + 2] * filterValue;
                    }
                }

                

                int centerOffset = (y * rowStride) + (x * bytesPerPixel);
                rowData[centerOffset + 0] = std::abs(sumB);
                rowData[centerOffset + 1] = std::abs(sumG);
                rowData[centerOffset + 2] = std::abs(sumR);
            }
        }

        // 결과를 파일에 쓰기
        outFile.seekp(fileHeader.bfOffBits, std::ios::beg);
        outFile.write(reinterpret_cast<const char*>(rowData.data()), rowData.size());

        outFile.close();
    }

    void sobel2(const std::string& filename) {
        int width = infoHeader.biWidth;
        int height = std::abs(infoHeader.biHeight);
        int bytesPerPixel = infoHeader.biBitCount / 8;
        int rowStride = ((width * bytesPerPixel + 3) / 4) * 4;

        std::vector<BYTE> rowData(rowStride * height);  // 이미지 전체를 위한 벡터
        std::vector<BYTE> tempRowData(rowStride * height);  // 임시 데이터 저장을 위한 벡터
        std::ofstream outFile(filename, std::ios::binary);

        outFile.write(reinterpret_cast<const char*>(&fileHeader), sizeof(BITMAPFILEHEADER));
        outFile.write(reinterpret_cast<const char*>(&infoHeader), sizeof(BITMAPINFOHEADER));

        bmpFile.seekg(fileHeader.bfOffBits, std::ios::beg);
        bmpFile.read(reinterpret_cast<char*>(rowData.data()), rowData.size());

        bool isTopDown = infoHeader.biHeight < 0;

        // 임시 데이터를 rowData에서 복사
        std::copy(rowData.begin(), rowData.end(), tempRowData.begin());

        // 3x3 블러 필터 적용
        for (int y = 1; y < height - 1; ++y) {
            for (int x = 1; x < width - 1; ++x) {
                int sumR = 0, sumG = 0, sumB = 0;
                int count = 0;

                for (int dy = -1; dy <= 1; ++dy) {
                    for (int dx = -1; dx <= 1; ++dx) {

                        int nx = x + dx;
                        int ny = y + dy;
                        int offset = (ny * rowStride) + (nx * bytesPerPixel);
                        // 필터 적용: -1, 0, 1
                        int filterValue = (dx == -1) ? -1 : (dx == 1) ? 1 : 0;

                        sumB += tempRowData[offset + 0] * filterValue;
                        sumG += tempRowData[offset + 1] * filterValue;
                        sumR += tempRowData[offset + 2] * filterValue;
                    }
                }



                int centerOffset = (y * rowStride) + (x * bytesPerPixel);
                rowData[centerOffset + 0] = std::abs(sumB);
                rowData[centerOffset + 1] = std::abs(sumG);
                rowData[centerOffset + 2] = std::abs(sumR);
            }
        }

        // 결과를 파일에 쓰기
        outFile.seekp(fileHeader.bfOffBits, std::ios::beg);
        outFile.write(reinterpret_cast<const char*>(rowData.data()), rowData.size());

        outFile.close();
    }
};

int main() {
    try {
        ModifyImg img("test.bmp"); 
        img.invertColors();
        img.avgfilter("avfiltered.bmp");
        img.sobel1("sobel1.bmp");
        img.sobel2("sobel2.bmp");
        ModifyImg img2("avfiltered.bmp");
        img2.avgfilter("avfiltered2.bmp");
        ModifyImg img3("avfiltered2.bmp");
        img3.avgfilter("avfiltered3.bmp");
        std::cout << "complete" << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "오류 발생: " << e.what() << std::endl;
        return 1;
    }
    return 0;
}