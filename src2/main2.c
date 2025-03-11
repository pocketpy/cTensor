#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <string.h>
#include "cten_internal.h"

// Define the neural network architecture
#define INPUT_SIZE 4
#define HIDDEN_SIZE 8
#define OUTPUT_SIZE 3
#define LEARNING_RATE 0.01
#define EPOCHS 1000
#define BATCH_SIZE 10

// Function declarations
void initialize_weights(float *W1, float *b1, float *W2, float *b2);
void shuffle_data(int *indices, int dataset_size);
float sigmoid(float x);
float sigmoid_derivative(float x);
void softmax(float *input, float *output, int size);
int argmax(float *array, int size);
float categorical_cross_entropy(float *predictions, int actual_class);
void forward_pass(float *input, float *W1, float *b1, float *W2, float *b2, 
                 float *hidden_output, float *final_output);
void backpropagation(float *input, int actual_class, float *hidden_output, float *final_output,
                    float *W1, float *b1, float *W2, float *b2, 
                    float *dW1, float *db1, float *dW2, float *db2);
void update_weights(float *W1, float *b1, float *W2, float *b2, 
                   float *dW1, float *db1, float *dW2, float *db2, 
                   int batch_size);
int predict(float *input, float *W1, float *b1, float *W2, float *b2);
float evaluate(const float (*X)[4], const int *y, int dataset_size, 
              float *W1, float *b1, float *W2, float *b2);

int main_() {
    // Load the Iris dataset
    const float (*X)[4];
    const int *y;
    int dataset_size = load_iris_dataset(&X, &y);
    
    printf("Dataset loaded with %d samples\n", dataset_size);

    // Allocate memory for weights and biases
    float *W1 = (float *)malloc(INPUT_SIZE * HIDDEN_SIZE * sizeof(float));
    float *b1 = (float *)malloc(HIDDEN_SIZE * sizeof(float));
    float *W2 = (float *)malloc(HIDDEN_SIZE * OUTPUT_SIZE * sizeof(float));
    float *b2 = (float *)malloc(OUTPUT_SIZE * sizeof(float));
    
    // Allocate memory for gradients
    float *dW1 = (float *)malloc(INPUT_SIZE * HIDDEN_SIZE * sizeof(float));
    float *db1 = (float *)malloc(HIDDEN_SIZE * sizeof(float));
    float *dW2 = (float *)malloc(HIDDEN_SIZE * OUTPUT_SIZE * sizeof(float));
    float *db2 = (float *)malloc(OUTPUT_SIZE * sizeof(float));
    
    // Allocate memory for intermediate outputs
    float *hidden_output = (float *)malloc(HIDDEN_SIZE * sizeof(float));
    float *final_output = (float *)malloc(OUTPUT_SIZE * sizeof(float));
    
    // Initialize weights with random values
    initialize_weights(W1, b1, W2, b2);
    
    // Create an array of indices for shuffling
    int *indices = (float *)malloc(dataset_size * sizeof(int));
    for (int i = 0; i < dataset_size; i++) {
        indices[i] = i;
    }
    
    // Training loop
    printf("Starting training for %d epochs...\n", EPOCHS);
    for (int epoch = 0; epoch < EPOCHS; epoch++) {
        // Shuffle the data for each epoch
        shuffle_data(indices, dataset_size);
        
        // Mini-batch training
        for (int batch_start = 0; batch_start < dataset_size; batch_start += BATCH_SIZE) {
            int batch_end = batch_start + BATCH_SIZE;
            if (batch_end > dataset_size) batch_end = dataset_size;
            int current_batch_size = batch_end - batch_start;
            
            // Reset gradients
            memset(dW1, 0, INPUT_SIZE * HIDDEN_SIZE * sizeof(float));
            memset(db1, 0, HIDDEN_SIZE * sizeof(float));
            memset(dW2, 0, HIDDEN_SIZE * OUTPUT_SIZE * sizeof(float));
            memset(db2, 0, OUTPUT_SIZE * sizeof(float));
            
            // Process each sample in the batch
            for (int i = batch_start; i < batch_end; i++) {
                int idx = indices[i];
                
                // Forward pass
                forward_pass((float*)X[idx], W1, b1, W2, b2, hidden_output, final_output);
                
                // Backward pass
                backpropagation((float*)X[idx], y[idx], hidden_output, final_output, 
                               W1, b1, W2, b2, dW1, db1, dW2, db2);
            }
            
            // Update weights with the accumulated gradients
            update_weights(W1, b1, W2, b2, dW1, db1, dW2, db2, current_batch_size);
        }
        
        // Calculate accuracy every 100 epochs
        if (epoch % 100 == 0 || epoch == EPOCHS - 1) {
            float accuracy = evaluate(X, y, dataset_size, W1, b1, W2, b2);
            printf("Epoch %d, Accuracy: %.2f%%\n", epoch, accuracy * 100);
        }
    }
    
    printf("Training completed.\n");
    
    // Free allocated memory
    free(W1);
    free(b1);
    free(W2);
    free(b2);
    free(dW1);
    free(db1);
    free(dW2);
    free(db2);
    free(hidden_output);
    free(final_output);
    free(indices);
    
    return 0;
}

// Initialize weights with random values
void initialize_weights(float *W1, float *b1, float *W2, float *b2) {
    srand(time(NULL));
    
    // Xavier initialization for W1
    float scale1 = sqrt(6.0 / (INPUT_SIZE + HIDDEN_SIZE));
    for (int i = 0; i < INPUT_SIZE * HIDDEN_SIZE; i++) {
        W1[i] = ((float)rand() / RAND_MAX) * 2 * scale1 - scale1;
    }
    
    // Initialize b1 to zeros
    for (int i = 0; i < HIDDEN_SIZE; i++) {
        b1[i] = 0.0;
    }
    
    // Xavier initialization for W2
    float scale2 = sqrt(6.0 / (HIDDEN_SIZE + OUTPUT_SIZE));
    for (int i = 0; i < HIDDEN_SIZE * OUTPUT_SIZE; i++) {
        W2[i] = ((float)rand() / RAND_MAX) * 2 * scale2 - scale2;
    }
    
    // Initialize b2 to zeros
    for (int i = 0; i < OUTPUT_SIZE; i++) {
        b2[i] = 0.0;
    }
}

// Shuffle the data indices for each epoch
void shuffle_data(int *indices, int dataset_size) {
    for (int i = dataset_size - 1; i > 0; i--) {
        int j = rand() % (i + 1);
        int temp = indices[i];
        indices[i] = indices[j];
        indices[j] = temp;
    }
}

// Sigmoid activation function
float sigmoid(float x) {
    return 1.0 / (1.0 + exp(-x));
}

// Derivative of sigmoid function
float sigmoid_derivative(float x) {
    float s = sigmoid(x);
    return s * (1 - s);
}

// Softmax activation for output layer
void softmax(float *input, float *output, int size) {
    float max_val = input[0];
    for (int i = 1; i < size; i++) {
        if (input[i] > max_val) {
            max_val = input[i];
        }
    }
    
    float sum = 0.0;
    for (int i = 0; i < size; i++) {
        output[i] = exp(input[i] - max_val);
        sum += output[i];
    }
    
    for (int i = 0; i < size; i++) {
        output[i] /= sum;
    }
}

// Find the index of the maximum value
int argmax(float *array, int size) {
    int max_idx = 0;
    for (int i = 1; i < size; i++) {
        if (array[i] > array[max_idx]) {
            max_idx = i;
        }
    }
    return max_idx;
}

// Calculate categorical cross entropy loss
float categorical_cross_entropy(float *predictions, int actual_class) {
    return -log(predictions[actual_class] + 1e-10); // Adding a small epsilon to prevent log(0)
}

// Forward pass through the neural network
void forward_pass(float *input, float *W1, float *b1, float *W2, float *b2, 
                 float *hidden_output, float *final_output) {
    // Hidden layer pre-activation
    float hidden_pre[HIDDEN_SIZE];
    for (int i = 0; i < HIDDEN_SIZE; i++) {
        hidden_pre[i] = b1[i];
        for (int j = 0; j < INPUT_SIZE; j++) {
            hidden_pre[i] += input[j] * W1[j * HIDDEN_SIZE + i];
        }
        // Apply sigmoid activation
        hidden_output[i] = sigmoid(hidden_pre[i]);
    }
    
    // Output layer pre-activation
    float output_pre[OUTPUT_SIZE];
    for (int i = 0; i < OUTPUT_SIZE; i++) {
        output_pre[i] = b2[i];
        for (int j = 0; j < HIDDEN_SIZE; j++) {
            output_pre[i] += hidden_output[j] * W2[j * OUTPUT_SIZE + i];
        }
    }
    
    // Apply softmax activation to output layer
    softmax(output_pre, final_output, OUTPUT_SIZE);
}

// Backpropagation to compute gradients
void backpropagation(float *input, int actual_class, float *hidden_output, float *final_output,
                    float *W1, float *b1, float *W2, float *b2, 
                    float *dW1, float *db1, float *dW2, float *db2) {
    // Create one-hot encoding for the actual class
    float y_true[OUTPUT_SIZE] = {0};
    y_true[actual_class] = 1.0;
    
    // Output layer error (derivative of cross entropy with softmax is simple)
    float d_output[OUTPUT_SIZE];
    for (int i = 0; i < OUTPUT_SIZE; i++) {
        d_output[i] = final_output[i] - y_true[i];
    }
    
    // Gradient for W2 and b2
    for (int i = 0; i < HIDDEN_SIZE; i++) {
        for (int j = 0; j < OUTPUT_SIZE; j++) {
            dW2[i * OUTPUT_SIZE + j] += hidden_output[i] * d_output[j];
        }
    }
    
    for (int i = 0; i < OUTPUT_SIZE; i++) {
        db2[i] += d_output[i];
    }
    
    // Hidden layer error
    float d_hidden[HIDDEN_SIZE] = {0};
    for (int i = 0; i < HIDDEN_SIZE; i++) {
        for (int j = 0; j < OUTPUT_SIZE; j++) {
            d_hidden[i] += d_output[j] * W2[i * OUTPUT_SIZE + j];
        }
        d_hidden[i] *= hidden_output[i] * (1 - hidden_output[i]); // Sigmoid derivative
    }
    
    // Gradient for W1 and b1
    for (int i = 0; i < INPUT_SIZE; i++) {
        for (int j = 0; j < HIDDEN_SIZE; j++) {
            dW1[i * HIDDEN_SIZE + j] += input[i] * d_hidden[j];
        }
    }
    
    for (int i = 0; i < HIDDEN_SIZE; i++) {
        db1[i] += d_hidden[i];
    }
}

// Update weights using SGD
void update_weights(float *W1, float *b1, float *W2, float *b2, 
                   float *dW1, float *db1, float *dW2, float *db2, 
                   int batch_size) {
    float scale = LEARNING_RATE / batch_size;
    
    // Update W1
    for (int i = 0; i < INPUT_SIZE * HIDDEN_SIZE; i++) {
        W1[i] -= scale * dW1[i];
    }
    
    // Update b1
    for (int i = 0; i < HIDDEN_SIZE; i++) {
        b1[i] -= scale * db1[i];
    }
    
    // Update W2
    for (int i = 0; i < HIDDEN_SIZE * OUTPUT_SIZE; i++) {
        W2[i] -= scale * dW2[i];
    }
    
    // Update b2
    for (int i = 0; i < OUTPUT_SIZE; i++) {
        b2[i] -= scale * db2[i];
    }
}

// Make a prediction for a single input
int predict(float *input, float *W1, float *b1, float *W2, float *b2) {
    float hidden_output[HIDDEN_SIZE];
    float final_output[OUTPUT_SIZE];
    
    forward_pass(input, W1, b1, W2, b2, hidden_output, final_output);
    
    return argmax(final_output, OUTPUT_SIZE);
}

// Evaluate the model on the dataset
float evaluate(const float (*X)[4], const int *y, int dataset_size, 
              float *W1, float *b1, float *W2, float *b2) {
    int correct = 0;
    
    for (int i = 0; i < dataset_size; i++) {
        int prediction = predict((float*)X[i], W1, b1, W2, b2);
        if (prediction == y[i]) {
            correct++;
        }
    }
    
    return (float)correct / dataset_size;
}