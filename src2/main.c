#include "cten.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

enum MemoryPoolIds {
    PoolId_Default = 0,
    PoolId_Model = 1,
    PoolId_Optimizer = 2,
};

typedef struct Model {
    Tensor weight_1, weight_2;
    Tensor bias_1, bias_2;
} Model;

Tensor Model_forward(Model* model, Tensor x) {
    x = nn_linear(x, model->weight_1, model->bias_1);
    x = nn_relu(x);
    x = nn_linear(x, model->weight_2, model->bias_2);
    x = nn_softmax(x);
    return x;
}

int main() {
    cten_initilize();
    
    // load iris dataset
    const float(*X)[4];
    const int* y;

    int n_samples = load_iris_dataset(&X, &y);
    int n_features = 4;
    int n_classes = 3;

    int n_train_samples = n_samples * 0.8;
    int n_test_samples = n_samples - n_train_samples;

    printf("n_samples: %d\n", n_samples);
    printf("n_train_samples: %d\n", n_train_samples);
    printf("n_test_samples: %d\n", n_test_samples);

    float mean[4] = {0}, std[4] = {0};
    for (int i = 0; i < n_train_samples; i++) {
        for (int j = 0; j < 4; j++) {
            mean[j] += X[i][j];
        }
    }
    for (int j = 0; j < 4; j++) {
        mean[j] /= n_train_samples;
    }
    for (int i = 0; i < n_train_samples; i++) {
        for (int j = 0; j < 4; j++) {
            std[j] += (X[i][j] - mean[j]) * (X[i][j] - mean[j]);
        }
    }
    for (int j = 0; j < 4; j++) {
        std[j] = sqrtf(std[j] / n_train_samples);
        // Avoid division by zero
        if (std[j] == 0) std[j] = 1.0f;
    }

    float(*X_norm)[4] = malloc(n_samples * sizeof(*X_norm));
    for (int i = 0; i < n_samples; i++) {
        for (int j = 0; j < 4; j++) {
            X_norm[i][j] = (X[i][j] - mean[j]) / std[j];
        }
    }
    //Normalize the input
    X = (const float(*)[4])X_norm;

    // create model
    Model model;
    cten_begin_malloc(PoolId_Model);
    model.weight_1 = Tensor_new((TensorShape){n_features, 32}, true);
    model.bias_1 = Tensor_zeros((TensorShape){1, 32}, true);
    model.weight_2 = Tensor_new((TensorShape){32, n_classes}, true);
    model.bias_2 = Tensor_zeros((TensorShape){1, n_classes}, true);
    cten_end_malloc();

    // create optimizer
    cten_begin_malloc(PoolId_Optimizer);
    optim_sgd* optimizer = optim_sgd_new(4, (Tensor*)&model);
    optim_sgd_config(optimizer, 0.001f, 0.0f);
    cten_end_malloc();

    // train model
    int batch_size = 8;
    for(int epoch = 0; epoch < 3; epoch++) {
        printf("==> epoch: %d\n", epoch);
        for(int i = 0; i < n_train_samples; i += batch_size) {
            printf("    batch: %d/%d samples\n", i, n_train_samples);
            cten_begin_malloc(PoolId_Default);
            // prepare input and target
            Tensor input = Tensor_new((TensorShape){batch_size, n_features}, false);
            Tensor y_true = Tensor_zeros((TensorShape){batch_size, n_classes}, false);
            for(int j = 0; j < batch_size; j++) {
                for(int k = 0; k < n_features; k++) {
                    Tensor_set(input, j, k, 0, 0, X[i + j][k]);
                }
                // one-hot encoding
                Tensor_set(y_true, j, y[i + j], 0, 0, 1.0f);
            }
            // zero the gradients
            optim_sgd_zerograd(optimizer);
            // forward pass
            Tensor y_pred = Model_forward(&model, input);
            Tensor loss = nn_crossentropy(y_true, y_pred);
            // backward pass
            Tensor grad = Tensor_ones((TensorShape){1}, false);
            Tensor_backward(loss, grad);
            optim_sgd_step(optimizer);
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
        Tensor input = Tensor_new((TensorShape){1, n_features}, false);
        Tensor y_true = Tensor_zeros((TensorShape){1, n_classes}, false);
        for(int j = 0; j < n_features; j++) {
            Tensor_set(input, 0, j, 0, 0, X[i][j]);
        }
        Tensor_set(y_true, 0, y[i], 0, 0, 1.0f);
        // forward pass
        Tensor y_pred = Model_forward(&model, input);
        Tensor loss = nn_crossentropy(y_true, y_pred);
        // calculate accuracy
        int pred_classes[1];
        Tensor_argmax(y_pred, pred_classes);
        if(pred_classes[0] == y[i]) correct++;
        printf("Sample %d - True: %d, Pred: %d\n", i - n_train_samples, y[i], pred_classes[0]);
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
