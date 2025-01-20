#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <stdio.h>

//#include "includes/mnist_main.h"


#define ADDRESS_TRAIN "data/train-images.idx3-ubyte"
#define ADDRESS_TRAIN_LABEL "data/train-labels.idx1-ubyte"

#define NUM_IMAGES 200
#define NUM_PIXELS 784

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

int max_num(unsigned char* data, int size) {
    int max = 0;
    for (int i = 0; i < size; i++) {
        if (data[i] > max) {
            max = data[i];
        }
    }
    return max;
}

double entropy(unsigned char* data, int size) {
    double entropy = 0;
    if (size == 0) return 0;

    int maxi = max_num(data, size) + 1; // +1, because must include zero

    double* bin_count = malloc(maxi * sizeof(double));
    memset(bin_count, 0, maxi * sizeof(double));


    for (int i = 0; i < size; i++) {
        bin_count[data[i]] ++;
    }

    // count probability of other element
    for (int i = 0; i < maxi; i++) {
        bin_count[i] /= size;
    }

    for (int i=0; i < maxi; i++) {
        double p = bin_count[i];
        if (p != 0) {
            entropy += p * (log10(p) / log10(2));
        }
    }
    free(bin_count);
    return -entropy;
}
typedef struct Node Node;

struct Node
{
    int feather; // this is the best column for splitting
    int result; // possible answer for image on this node
    Node* true_node;
    Node* false_node;
};


// must add feather, because look at one determinant column
void split_data(unsigned char test_image[][NUM_PIXELS], unsigned char* test_label,\
    int feature, unsigned char (*true_images)[NUM_PIXELS], int* num_true_images,\
    unsigned char (*false_images)[NUM_PIXELS], int* num_false_images,\
    unsigned char* true_labels, unsigned char* false_labels, int num_of_images)
{
    // feature - number of column
    // next I find columns, that pixels > 0.5 = true, and < 0.5 = false
    // I want get +- equals splits

    int* true_pixels = malloc(num_of_images * sizeof(int)); // first part
    int index_for_true = 0;

    int* false_pixels = malloc(num_of_images * sizeof(int)); // second part
    int index_for_false = 0;

    // at the start I am finding true and false indexes
    for(int i = 0; i < num_of_images; i++)
    {
        // test_image[i][feature] - elements by one column
        if (test_image[i][feature] > 127) // it mens 1
        {
            true_pixels[index_for_true] = i;
            index_for_true++;
        }
        else // it mens 0
        {
            false_pixels[index_for_false] = i;
            index_for_false++;
        }
    }
    // next I am splitting my data into two parts - true/false
    for (int i = 0; i < index_for_true; i++) {
        for (int j = 0;  j < NUM_PIXELS; j++) {
            true_images[i][j] = test_image[true_pixels[i]][j];
        }
    }
    for (int i = 0; i < index_for_false; i++) {
        for (int j = 0;  j < NUM_PIXELS; j++) {
            false_images[i][j] = test_image[false_pixels[i]][j];
        }
    }
    // must split labels
    for (int i = 0; i < index_for_true; i++) {
        true_labels[i] = test_label[true_pixels[i]];
    }
    for (int i = 0; i < index_for_false; i++) {
        false_labels[i] = test_label[false_pixels[i]];
    }
    *num_true_images = index_for_true;
    *num_false_images = index_for_false;

    free(true_pixels);
    free(false_pixels);
}

Node* create_Node() {
    Node* newNode = (Node*)malloc(sizeof(Node)); // create a new node
    newNode->true_node = NULL;
    newNode->false_node = NULL;
    newNode->result = -1;
    return newNode;

}

Node* build_tree(unsigned char test_image[][NUM_PIXELS], unsigned char* test_label,\
    Node* node, int num_of_images)
{
    int number_of_elements = 0;
    unsigned char prev_ele = 0;
    int best_gain = 0; // the best information gain
    double current_entropy = entropy(test_label, num_of_images);
    int best_feather = 0; // I will find the best splits and remember num of column

    unsigned char true_images[num_of_images][NUM_PIXELS]; // num_of_images is the sup of true and false images
    int num_true_images = 0;

    unsigned char false_images[num_of_images][NUM_PIXELS];
    int num_false_images = 0;

    unsigned char true_labels[num_of_images];
    unsigned char false_labels[num_of_images];

    // data for best split
    unsigned char true_images_best[num_of_images][NUM_PIXELS];
    unsigned char false_images_best[num_of_images][NUM_PIXELS];
    unsigned char true_labels_best[num_of_images];
    unsigned char false_labels_best[num_of_images];
    int num_true_images_best = 0;
    int num_false_images_best = 0;


    for (int i = 0; i < num_of_images; i++) {
        if (i == 0) {
            prev_ele = test_label[0];
            number_of_elements++;
        }
        else {
            if (prev_ele != test_label[i]) {
                number_of_elements++;
            }
            prev_ele = test_label[i];
        }
    }
    if (number_of_elements == 1) {
        node->result = test_label[0];
        return node; // filling in the tree
    }

    for (int i = 0; i < NUM_PIXELS; i++) {
        split_data(test_image, test_label, i, true_images,\
            &num_true_images, false_images, &num_false_images, true_labels, false_labels, num_of_images);
        double true_entropy = entropy(true_labels, num_true_images);
        double false_entropy = entropy(false_labels, num_false_images);


        double p = (double)num_true_images / num_of_images;
        double gain = current_entropy - p * true_entropy - (1 - p) * false_entropy;
        if (gain > best_gain) {
            // I must remember the best split
            best_gain = gain;
            best_feather = i;
            memcpy(true_images_best, true_images, sizeof(true_images));
            memcpy(false_images_best, false_images, sizeof(false_images));
            memcpy(true_labels_best, true_labels, sizeof(true_labels));
            memcpy(false_labels_best, false_labels, sizeof(false_labels));
            num_true_images_best = num_true_images;
            num_false_images_best = num_false_images;
        }
    }
    // building a tree

    Node* new_node = NULL;
    new_node = create_Node();

    if (best_gain > 0) {
        // this is splitting on two branches
        node->true_node = (struct Node*)build_tree(true_images_best, true_labels_best, new_node, num_true_images_best); // it is a node
        node->false_node = (struct Node*)build_tree(false_images_best, false_labels_best, new_node, num_false_images_best);
        node->feather = best_feather;

        return node;
    }
    node->result = test_label[0];
    return node;
}

int predict(Node* tree, unsigned char* image) {
    // I have a decision_tree and I want to say, what num on image is
    // I have to go around the tree and come to the end node
    if (tree->result != -1) {
        return tree->result;
    }
    else {
        Node* branch = tree->false_node;
        if (image[tree->feather] > 127) { // it is true
            branch = tree->true_node;
        }
        return predict(branch, image);
    }
}

void freeTree(Node* root) {
    if (root != NULL) {
        freeTree((Node*)root->true_node);
        freeTree((Node*)root->false_node);
        free(root);
    }
}

void main() {
    unsigned char images[NUM_IMAGES][NUM_PIXELS];
    unsigned char labels[NUM_IMAGES];

    load_mnist_images(ADDRESS_TRAIN, images);
    load_mnist_labels(ADDRESS_TRAIN_LABEL, labels);

    //for (int i = 0; i < NUM_PIXELS; i++) {
        //printf("%1.1hhu ", images[2][i]);
        //if ((i+1) % 28 == 0) putchar('\n');
    //}
    //printf("%d \n", labels[2]);

    //unsigned char images_trial[4][4] = {{240, 12, 67, 123},
                                        //{173, 200, 1, 255},
                                        //{8, 9, 10, 112},
                                        //{46, 12, 34, 12}};
    //unsigned char label_trial[4] = {1, 2, 3, 4};

    Node* root = NULL;
    root = create_Node();
    root = build_tree(images, labels, root, NUM_IMAGES);

    unsigned char images_trial[784] = {0, 29, 149, 76, 0, 29, 149, 76, 0, 29, 149, 76, 0, 29, 149, 76, 0, 29, 149, 76, 0, 29, 149, 76, 0, 29, 149, 76, 0, 29, 149, 76, 0, 29, 149, 76, 0, 29, 149, 76, 0, 29, 149, 76, 0, 29, 149, 76, 0, 29, 149, 76, 0, 29, 149, 76, 0, 29, 149, 76, 0, 29, 149, 76, 0, 29, 149, 76, 0, 29, 149, 76, 0, 29, 149, 76, 0, 29, 149, 76, 0, 29, 149, 76, 0, 29, 149, 76, 0, 29, 149, 76, 0, 29, 149, 76, 0, 29, 149, 76, 0, 29, 149, 76, 0, 29, 149, 76, 0, 29, 149, 76, 0, 29, 149, 76, 0, 29, 149, 76, 0, 29, 149, 76, 0, 29, 149, 76, 0, 29, 149, 76, 0, 29, 149, 76, 0, 29, 149, 76, 0, 29, 149, 76, 0, 29, 149, 76, 0, 29, 149, 76, 0, 29, 149, 76, 0, 29, 185, 160, 254, 254, 254, 254, 254, 254, 254, 254, 150, 29, 149, 76, 0, 29, 149, 76, 0, 29, 149, 76, 0, 29, 149, 76, 0, 29, 149, 76, 0, 29, 149, 76, 0, 29, 185, 160, 254, 254, 254, 254, 254, 254, 254, 254, 254, 254, 254, 254, 150, 29, 149, 76, 0, 29, 149, 76, 0, 29, 149, 76, 0, 29, 149, 76, 0, 29, 149, 76, 0, 29, 149, 76, 222, 254, 254, 254, 254, 254, 254, 254, 254, 254, 254, 254, 254, 254, 223, 182, 0, 29, 149, 76, 0, 29, 149, 76, 0, 29, 149, 76, 0, 29, 149, 76, 0, 29, 149, 76, 0, 92, 224, 232, 222, 114, 160, 76, 0, 92, 224, 232, 254, 254, 254, 254, 254, 162, 166, 76, 0, 29, 149, 76, 0, 29, 149, 76, 0, 29, 149, 76, 0, 29, 149, 76, 0, 29, 149, 76, 0, 29, 149, 76, 0, 29, 149, 76, 0, 92, 229, 243, 254, 254, 254, 254, 0, 29, 149, 76, 0, 29, 149, 76, 0, 29, 149, 76, 0, 29, 149, 76, 0, 29, 149, 76, 0, 29, 149, 76, 0, 29, 149, 76, 0, 29, 149, 76, 158, 254, 254, 254, 254, 29, 149, 76, 0, 29, 149, 76, 0, 29, 149, 76, 0, 29, 149, 76, 0, 29, 149, 76, 0, 29, 149, 76, 0, 29, 149, 76, 0, 29, 149, 76, 0, 247, 254, 254, 254, 247, 177, 76, 0, 29, 149, 76, 0, 29, 149, 76, 0, 29, 149, 76, 0, 29, 149, 76, 0, 29, 149, 76, 119, 254, 254, 254, 150, 29, 149, 76, 0, 135, 239, 254, 254, 254, 207, 143, 0, 29, 149, 76, 0, 29, 149, 76, 0, 29, 149, 76, 0, 29, 149, 76, 0, 29, 149, 76, 0, 247, 254, 254, 254, 254, 254, 254, 254, 254, 254, 254, 254, 254, 252, 249, 0, 29, 149, 76, 0, 29, 149, 76, 0, 29, 149, 76, 0, 29, 149, 76, 0, 29, 149, 76, 0, 29, 225, 254, 254, 254, 254, 254, 254, 254, 254, 254, 254, 254, 252, 249, 95, 29, 149, 76, 0, 29, 149, 76, 0, 29, 149, 76, 0, 29, 149, 76, 0, 29, 149, 76, 0, 29, 149, 76, 254, 254, 254, 254, 254, 254, 254, 254, 254, 254, 245, 232, 95, 29, 149, 76, 0, 29, 149, 76, 0, 29, 149, 76, 0, 29, 149, 76, 0, 29, 149, 76, 0, 29, 149, 76, 0, 254, 254, 254, 254, 29, 149, 76, 0, 29, 149, 76, 0, 29, 149, 76, 0, 29, 149, 76, 0, 29, 149, 76, 0, 29, 149, 76, 0, 29, 149, 76, 0, 29, 149, 76, 0, 29, 225, 254, 254, 254, 178, 76, 0, 29, 149, 76, 0, 29, 149, 76, 0, 29, 149, 76, 0, 29, 149, 76, 0, 29, 149, 76, 0, 29, 149, 76, 0, 29, 149, 76, 0, 29, 149, 76, 254, 254, 254, 254, 0, 29, 149, 76, 0, 29, 149, 76, 0, 29, 149, 76, 0, 29, 149, 76, 0, 29, 149, 76, 0, 29, 149, 76, 0, 29, 149, 76, 0, 29, 149, 76, 0, 254, 254, 254, 254, 29, 149, 76, 0, 29, 149, 76, 0, 29, 149, 76, 0, 29, 149, 76, 0, 29, 149, 76, 0, 29, 149, 76, 0, 29, 149, 76, 0, 29, 149, 76, 0, 29, 225, 254, 254, 254, 178, 76, 0, 29, 149, 76, 0, 29, 149, 76, 0, 29, 149, 76, 0, 29, 149, 76, 0, 29, 149, 76, 0, 29, 149, 76, 0, 29, 149, 76, 0, 29, 149, 76, 254, 254, 254, 254, 254, 254, 254, 254, 254, 254, 254, 254, 254, 254, 254, 254, 150, 29, 149, 76, 0, 29, 149, 76, 0, 29, 149, 76};
    int answer = predict(root, images_trial); // to get answer
    printf("%d predict\n", answer);

    freeTree(root);
}
