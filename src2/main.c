#include "cten.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

#define N_SAMPLES 150
#define N_FEATURES 4

void swap(float *a, float *b, int n) {
    for (int i = 0; i < n; i++) {
        float temp = a[i];
        a[i] = b[i];
        b[i] = temp;
    }
}

void shuffle(float X[N_SAMPLES][N_FEATURES], int y[N_SAMPLES]) {
    for (int i = N_SAMPLES - 1; i > 0; i--) {
        int j = rand() % (i + 1);

        swap(X[i], X[j], N_FEATURES);

        int temp = y[i];
        y[i] = y[j];
        y[j] = temp;
    }
}

// void print_tensor(Tensor t, void* ctx) {
//     printf("Tensor at %p, shape: (%d)\n", (void*)&t, t.data->numel);
// }

// void debug_computation_graph(Tensor loss) {
//     printf("Traversing computation graph:\n");
//     int count = Tensor_backward_apply(loss, print_tensor, NULL);
//     printf("Total tensors in graph: %d\n", count);
// }

enum MemoryPoolIds {
    PoolId_Default = 0,
    PoolId_Model = 1,
    PoolId_Optimizer = 2,
};

typedef struct Model {
    Tensor weight_1, weight_2;//, weight_3;
    Tensor bias_1, bias_2;//, bias_3
} Model;

Tensor Model_forward(Model* model, Tensor x) {
    x = nn_linear(x, model->weight_1, model->bias_1);
    x = nn_relu(x);
    x = nn_linear(x, model->weight_2, model->bias_2);
    // x = nn_relu(x);
    // x = nn_linear(x, model->weight_3, model->bias_3);
    x = nn_softmax(x);    // Compute mean

    return x;
}

void normalize(float X[][4], int num_samples) {
    float mean[4] = {0}, stddev[4] = {0};

    for (int j = 0; j < 4; j++) {
        for (int i = 0; i < num_samples; i++) {
            mean[j] += X[i][j];
        }
        mean[j] /= num_samples;
    }

    for (int j = 0; j < 4; j++) {
        for (int i = 0; i < num_samples; i++) {
            stddev[j] += (X[i][j] - mean[j]) * (X[i][j] - mean[j]);
        }
        stddev[j] = sqrt(stddev[j] / num_samples);
    }

    for (int j = 0; j < 4; j++) {
        for (int i = 0; i < num_samples; i++) {
            X[i][j] = (X[i][j] - mean[j]) / (stddev[j] + 1e-8);
        }
    }
}

int main() {
    cten_initilize();
    srand(time(NULL));

    // load iris dataset
    const float(*X)[4];
    const int* y;

    int n_samples = load_iris_dataset(&X, &y);
    int n_features = 4;
    int n_classes = 3;

    int n_train_samples = n_samples * 0.8;
    int n_test_samples = n_samples - n_train_samples;

    float Xt[DATASET_SIZE][4];  // Mutable copy
    int yt[DATASET_SIZE];

    for (int i = 0; i < n_samples; i++) {
        for (int j = 0; j < 4; j++) {
            Xt[i][j] = X[i][j];  // Copy features
        }
        yt[i] = y[i];
    }

    normalize(Xt, DATASET_SIZE);

    printf("n_samples: %d\n", n_samples);
    printf("n_train_samples: %d\n", n_train_samples);
    printf("n_test_samples: %d\n", n_test_samples);

    // create model
    Model model;
    cten_begin_malloc(PoolId_Model);
    model.weight_1 = Tensor_init_he((TensorShape){n_features, 8}, true);
    model.bias_1 = Tensor_zeros((TensorShape){1, 8}, true);
    // model.weight_2 = Tensor_init_he((TensorShape){32, 16}, true);
    // model.bias_2 = Tensor_zeros((TensorShape){1, 16}, true);
    model.weight_2 = Tensor_init_he((TensorShape){8, n_classes}, true);
    model.bias_2 = Tensor_zeros((TensorShape){1, n_classes}, true);
    Tensor_print(model.weight_1);

    cten_end_malloc();

    // create optimizer
    cten_begin_malloc(PoolId_Optimizer);
    optim_sgd* optimizer = optim_sgd_new(4, (Tensor*)&model);
    optim_sgd_config(optimizer, 0.001f, 0.0f);
    cten_end_malloc();
    shuffle(Xt, yt);

    // train model
    int batch_size = 8;
    for(int epoch = 0; epoch < 3; epoch++) {
        printf("==> epoch: %d\n", epoch);
        for(int i = 0; i < n_train_samples; i += batch_size) {
            printf("    batch: %d/%d samples\n", i, n_train_samples);
            cten_begin_malloc(PoolId_Default);
            // prepare input and target
            Tensor input = Tensor_zeros((TensorShape){batch_size, n_features}, false);
            Tensor y_true = Tensor_zeros((TensorShape){batch_size, n_classes}, false);
            for(int j = 0; j < batch_size; j++) {
                for(int k = 0; k < n_features; k++) {
                    input.data->flex[j * n_features + k] = Xt[i + j][k];
                }
                // one-hot encoding
                y_true.data->flex[j * n_classes + yt[i + j]] = 1.0f;
            }
            // Tensor_print(input);
            // Tensor_print(y_true);
            // zero the gradients
            optim_sgd_zerograd(optimizer); // kinda - for now it does zeroes the grad
            // forward pass
            Tensor y_pred = Model_forward(&model, input); // good - kinda
            // printf("y_pred Tensor: ");
            // Tensor_print(y_pred);
            Tensor loss = nn_crossentropy(y_true, y_pred);
            // printf("=== Checking Computation Graph ===\n");
            // int nodes_visited = Tensor_backward_apply(loss, NULL, NULL);
            // printf("Total nodes in the graph: %d\n", nodes_visited);
            printf("loss: %.4f\n", Tensor_get(loss, 0, 0, 0, 0));
            // backward pass
            // printf("Weight_1 tensor: \n");
            // Tensor_print(model.weight_1);
            // printf("bias_1 tensor: \n");
            // Tensor_print(model.bias_1);
            // printf("Weight_2 tensor: \n");
            // Tensor_print(model.weight_2);
            // printf("bias_2 tensor: \n");
            // Tensor_print(model.bias_2);
            // debug_computation_graph(loss);
            Tensor_backward(loss, (Tensor){});
            optim_sgd_step(optimizer);
            // printf("Weight_1 tensor: \n");
            // Tensor_print(model.weight_1);
            // printf("bias_1 tensor: \n");
            // Tensor_print(model.bias_1);
            // printf("Weight_2 tensor: \n");
            // Tensor_print(model.weight_2);
            // printf("bias_2 tensor: \n");
            // Tensor_print(model.bias_2);
            // Tensor_print(optimizer->params[0]);
            // Tensor_print(optimizer->params[1]);
            // Tensor_print(optimizer->params[2]);
            // Tensor_print(optimizer->params[3]);
            cten_end_malloc();
            // free temporary tensors
            cten_free(PoolId_Default);
        }
    }

    // free optimizer
    cten_free(PoolId_Optimizer);

    // evaluate model
    cten_begin_eval();
    int correct = 0;
    for(int i = n_train_samples; i < n_samples; i++) {
        cten_begin_malloc(PoolId_Default);
        // prepare input and target
        Tensor input = Tensor_zeros((TensorShape){1, n_features}, false);
        Tensor y_true = Tensor_zeros((TensorShape){1, n_classes}, false);
        for(int j = 0; j < n_features; j++) {
            input.data->flex[j] = X[i][j];
        }
        y_true.data->flex[y[i]] = 1.0f;
        // forward pass
        Tensor y_pred = Model_forward(&model, input);
        Tensor loss = nn_crossentropy(y_true, y_pred);
        // calculate accuracy
        int pred_classes[1];
        Tensor_argmax(y_pred, pred_classes);
        if(pred_classes[0] == y[i]) correct++;
        cten_end_malloc();
        // free temporary tensors
        cten_free(PoolId_Default);
    }
    printf("accuracy: %.4f\n", (float)correct / n_test_samples);
    cten_end_eval();

    // free model
    cten_free(PoolId_Model);

    cten_finalize();
    return 0;
}
