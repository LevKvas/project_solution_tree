#pragma once

#define NUM_IMAGES 4
#define NUM_PIXELS 4

void load_mnist_images(const char *filename, unsigned char images[NUM_IMAGES][NUM_PIXELS]);
void print_image(unsigned char image[NUM_PIXELS]);
void load_mnist_labels(const char *filename, unsigned char* labels);