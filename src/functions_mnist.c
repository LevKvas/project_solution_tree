#include <stdio.h>
#include <stdlib.h>

#define NUM_IMAGES 4
#define NUM_PIXELS 4


// Функция для загрузки изображений MNIST
void load_mnist_images(const char *filename, unsigned char images[NUM_IMAGES][NUM_PIXELS]) {
    FILE *file = fopen(filename, "rb");
    if (!file) {
        perror("Cannot open file");
        exit(EXIT_FAILURE);
    }

    fseek(file, 16, SEEK_SET); // Пропускаем заголовок (16 байт)
    fread(images, sizeof(unsigned char), NUM_IMAGES * NUM_PIXELS, file);
    fclose(file);
}

void load_mnist_labels(const char *filename, unsigned char* labels) {
    FILE *file = fopen(filename, "rb");

    fseek(file, 8, SEEK_SET); // Пропускаем заголовок (8 байт)
    fread(labels, sizeof(unsigned char), NUM_IMAGES, file);
    fclose(file);
}

// Функция для отображения изображения в текстовом виде
void print_image(unsigned char image[NUM_PIXELS]) {
    for (int x = 0; x < NUM_PIXELS; x++) {
      unsigned char pixel = image[x];
      if (pixel > 128) {
        printf("  "); // Пробел для светлых пикселей
      } else {
        printf("**"); // Звездочки для темных пикселей
      }
    }
    printf("\n");
}

