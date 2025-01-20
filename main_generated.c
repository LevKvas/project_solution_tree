#include <stdio.h>
#include <stdlib.h>

#pragma pack(push, 1) // Упаковка структуры без выравнивания

typedef struct {
    unsigned short bfType;      // Тип файла (должен быть 'BM')
    unsigned int bfSize;       // Размер файла в байтах
    unsigned short bfReserved1; // Зарезервировано
    unsigned short bfReserved2; // Зарезервировано
    unsigned int bfOffBits;     // Смещение до данных пикселей
} BITMAPFILEHEADER;

typedef struct {
    unsigned int biSize;          // Размер структуры
    int biWidth;                  // Ширина изображения
    int biHeight;                 // Высота изображения
    unsigned short biPlanes;      // Количество плоскостей
    unsigned short biBitCount;    // Количество бит на пиксель
    unsigned int biCompression;    // Тип сжатия
    unsigned int biSizeImage;      // Размер изображения в байтах
    int biXPelsPerMeter;          // Горизонтальное разрешение
    int biYPelsPerMeter;          // Вертикальное разрешение
    unsigned int biClrUsed;       // Количество используемых цветов
    unsigned int biClrImportant;   // Количество важных цветов
} BITMAPINFOHEADER;

#pragma pack(pop) // Возврат к стандартному выравниванию

void readBMP(const char *filename, int* image_test) {
    FILE *file = fopen(filename, "rb");
    if (!file) {
        perror("Unable to open file");
        return;
    }

    BITMAPFILEHEADER bfh;
    fread(&bfh, sizeof(BITMAPFILEHEADER), 1, file);
    
    if (bfh.bfType != 0x4D42) { // 'BM' в шестнадцатеричном представлении
        printf("Not a valid BMP file\n");
        fclose(file);
        return;
    }

    BITMAPINFOHEADER bih;
    fread(&bih, sizeof(BITMAPINFOHEADER), 1, file);

    printf("Width: %d\n", bih.biWidth);
    printf("Height: %d\n", bih.biHeight);
    printf("Bit Count: %d\n", bih.biBitCount);

    // Переход к началу данных пикселей
    fseek(file, bfh.bfOffBits, SEEK_SET);

    // Вычисляем размер строки с учетом выравнивания
    int rowSize = (bih.biWidth * bih.biBitCount / 8 + 3) & ~3;
    
    // Чтение пикселей
    unsigned char *pixels = (unsigned char *)malloc(rowSize * bih.biHeight);
    
    for (int i = 0; i < bih.biHeight; i++) {
        fread(pixels + i * rowSize, rowSize, 1, file);
    }

    fclose(file);

    for (int i = 0; i < 784 && i < bih.biWidth * bih.biHeight; i++) {
        unsigned char b = pixels[i * 3];     // Синий компонент
        unsigned char g = pixels[i * 3 + 1]; // Зеленый компонент
        unsigned char r = pixels[i * 3 + 2]; // Красный компонент

        int pixel_value = 0.299 * r + 0.587 * g + 0.114 * b;
        image_test[i] = pixel_value;
    }

    free(pixels);
}

int main() {
    int image_test[784];
    readBMP("images/five_test.bmp", image_test); // Укажите имя вашего BMP файла

    for(int i = 0; i < 784; i ++)
    {
        printf("%d \n", image_test[i]);
    }
    return 0;
}
