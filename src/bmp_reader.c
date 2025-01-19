#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>

#pragma pack(push, 1) // Выравнивание по 1 байту

typedef struct {
    uint16_t bfType;      // Тип файла (должен быть 'BM')
    uint32_t bfSize;      // Размер файла в байтах
    uint16_t bfReserved1; // Зарезервировано
    uint16_t bfReserved2; // Зарезервировано
    uint32_t bfOffBits;   // Смещение до начала данных пикселей
} BMPHeader;

typedef struct {
    uint32_t biSize;          // Размер структуры DIB
    int32_t  biWidth;         // Ширина изображения
    int32_t  biHeight;        // Высота изображения
    uint16_t biPlanes;        // Количество цветовых плоскостей
    uint16_t biBitCount;      // Количество бит на пиксель
    uint32_t biCompression;    // Метод сжатия
    uint32_t biSizeImage;     // Размер изображения в байтах
    int32_t  biXPelsPerMeter;  // Горизонтальное разрешение
    int32_t  biYPelsPerMeter;  // Вертикальное разрешение
    uint32_t biClrUsed;       // Количество используемых цветов
    uint32_t biClrImportant;   // Важные цвета
} DIBHeader;

#pragma pack(pop)

void readBMP(const char *filename, uint8_t** pixelData, int* size_of_data) {
    FILE *file = fopen(filename, "rb");

    BMPHeader bmpHeader;
    DIBHeader dibHeader;

    // Чтение заголовков
    fread(&bmpHeader, sizeof(BMPHeader), 1, file);
    fread(&dibHeader, sizeof(DIBHeader), 1, file);

    // Перемещение указателя файла на начало данных пикселей
    fseek(file, bmpHeader.bfOffBits, SEEK_SET);

    // Выделение памяти для пикселей
    int rowSize = ((dibHeader.biBitCount * dibHeader.biWidth + 31) / 32) * 4;
    *pixelData = (uint8_t *)malloc(rowSize * dibHeader.biHeight);

    *size_of_data = rowSize * dibHeader.biHeight;

    fread(*pixelData, rowSize, dibHeader.biHeight, file);
    fclose(file);
}