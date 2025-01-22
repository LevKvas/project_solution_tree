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

double gini_coefficient(unsigned char* data, int size) {
    double gini = 0;

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

    for (int i = 0; i < maxi; i++) {
        gini += bin_count[i] * bin_count[i];
    }

    free(bin_count);
    return 1 - gini;
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
    unsigned char value;
    Node* true_node;
    Node* false_node;

};


// must add feather, because look at one determinant column
void split_data(unsigned char test_image[][NUM_PIXELS], unsigned char* test_label,\
    int feature, unsigned char (*true_images)[NUM_PIXELS], int* num_true_images,\
    unsigned char (*false_images)[NUM_PIXELS], int* num_false_images,\
    unsigned char* true_labels, unsigned char* false_labels, int num_of_images, unsigned char separating_value)
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
        if (test_image[i][feature] <= separating_value) // it mens 1
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
    Node* newNode = malloc(sizeof(Node)); // create a new node
    newNode->true_node = NULL;
    newNode->false_node = NULL;
    newNode->result = -1;
    newNode->value = 0;
    return newNode;

}

Node* build_tree(unsigned char test_image[][NUM_PIXELS], unsigned char* test_label,\
    Node* node, int num_of_images)
{
    int number_of_elements = 0;
    unsigned char prev_ele = 0;
    double best_gini = 1; // very bad situation
    unsigned char best_separating_value = 0;
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
        Node* new_node = NULL;
        new_node = create_Node(); // because node will be NULL always

        new_node->result = test_label[0];
        return new_node; // filling in the tree
    }

    for (int i = 0; i < NUM_PIXELS; i++) {
        for (int j = 0; j < num_of_images; j++) {
            split_data(test_image, test_label, i, true_images,\
                    &num_true_images, false_images, &num_false_images, true_labels, false_labels, num_of_images, test_image[j][i]);
            double true_gini = gini_coefficient(true_labels, num_true_images);
            double false_gini = gini_coefficient(false_labels, num_false_images);


            double p_true = (double)num_true_images / num_of_images; // probability of true
            double p_false = (double)num_true_images / num_of_images;

            double weighted_gini = p_true * true_gini + p_false * false_gini;
            // gini must limited to 0(ideal situation)

            if (weighted_gini < best_gini) {
                // I must remember the best split
                best_gini = weighted_gini;
                best_feather = i;
                best_separating_value = test_image[j][i];
                memcpy(true_images_best, true_images, sizeof(true_images));
                memcpy(false_images_best, false_images, sizeof(false_images));
                memcpy(true_labels_best, true_labels, sizeof(true_labels));
                memcpy(false_labels_best, false_labels, sizeof(false_labels));
                num_true_images_best = num_true_images;
                num_false_images_best = num_false_images;
            }
        }
    }
    // ПОЛАГАЮ, ГДЕ-ТО ЗДЕСЬ ОШИБКА

    if (best_gini < 1) {
        // this is splitting on two branches
        if (node == NULL) {
            node = create_Node();
        }
        node->true_node = build_tree(true_images_best, true_labels_best, node->true_node, num_true_images_best); // it is a node
        node->false_node = build_tree(false_images_best, false_labels_best, node->false_node, num_false_images_best);
        node->feather = best_feather;
        node->value = best_separating_value;

        return node;
    }
    printf("%d \n", test_label[0]);
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
        if (image[tree->feather] <= tree->value) { // it is true
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

void convert_to_b_w(unsigned char** images) {
    for (int i = 0; i < NUM_IMAGES; i++) {
        for (int j = 0; j < NUM_PIXELS; j++) {
            if (images[i][j] > 127) {
                images[i][j] = 1;
            }
            else {
                images[i][j] = 0;
            }
        }
    }

}

void main() {
    unsigned char images[NUM_IMAGES][NUM_PIXELS];
    unsigned char labels[NUM_IMAGES];

    load_mnist_images(ADDRESS_TRAIN, images); // loading data from mnist
    load_mnist_labels(ADDRESS_TRAIN_LABEL, labels);


    //for (int i = 0; i < NUM_PIXELS; i++) {
    //printf("%1.1hhu ", images[300][i]);
    //if ((i+1) % 28 == 0) putchar('\n');
    //}
    //printf("%d \n", labels[300]);

    //unsigned char images_trial[4][4] = {{240, 12, 67, 123},
    //{173, 200, 1, 255},
    //{8, 9, 10, 112},
    //{46, 12, 34, 12}};
    //unsigned char label_trial[4] = {1, 2, 3, 4};

    Node* root = NULL;
    root = create_Node();
    root = build_tree(images, labels, root, NUM_IMAGES); // building a tree

    //six in white/black format
    unsigned char image_for_find[784] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
    int answer = predict(root, image_for_find); // to get answer
    printf("%d predict\n", answer);
}
