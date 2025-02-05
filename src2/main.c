#include "cten.h"
#include <stdio.h>
#include <stdlib.h>

typedef struct Model {
    Tensor weight_1, weight_2;
    Tensor bias_1, bias_2;
} Model;

Tensor model_forward(Model* model, Tensor input) {
    Tensor x = nn_linear(input, model->weight_1, model->bias_1);
    x = nn_relu(x);
    x = nn_linear(x, model->weight_2, model->bias_2);
    x = nn_softmax(x);
    return x;
}

int main_disable() {
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

    // create model
    Model model;
    model.weight_1 = Tensor_new((TensorShape){n_features, 32}, true);
    model.bias_1 = Tensor_zeros((TensorShape){1, 32}, true);
    model.weight_2 = Tensor_new((TensorShape){32, n_classes}, true);
    model.bias_2 = Tensor_zeros((TensorShape){1, n_classes}, true);

    // train model
    optim_sgd* optimizer = optim_sgd_new(4, (Tensor*)&model);
    optim_sgd_config(optimizer, 0.001f, 0.0f);

    int batch_size = 8;
    for(int epoch = 0; epoch < 3; epoch++) {
        for(int i = 0; i < n_train_samples; i += batch_size) {
            // create input and target tensors
            Tensor input = Tensor_new((TensorShape){batch_size, n_features}, false);
            Tensor y_true = Tensor_zeros((TensorShape){batch_size, n_classes}, false);
            for(int j = 0; j < batch_size; j++) {
                for(int k = 0; k < n_features; k++) {
                    input.data->flex[j * n_features + k] = X[i + j][k];
                }
                y_true.data->flex[j * n_classes + y[i + j]] = 1.0f;
            }
            optim_sgd_zerograd(optimizer);
            Tensor y_pred = model_forward(&model, input);
            Tensor loss = nn_crossentropy(y_true, y_pred);
            Tensor_backward(loss, (Tensor){NULL});
            optim_sgd_step(optimizer);
        }
    }

    // evaluate model
    cten_begin_eval();

    int correct = 0;
    for(int i = n_train_samples; i < n_samples; i++) {
        Tensor input = Tensor_new((TensorShape){1, n_features}, false);
        Tensor y_true = Tensor_zeros((TensorShape){1, n_classes}, false);
        for(int j = 0; j < n_features; j++) {
            input.data->flex[j] = X[i][j];
        }
        y_true.data->flex[y[i]] = 1.0f;

        Tensor y_pred = model_forward(&model, input);
        Tensor loss = nn_crossentropy(y_true, y_pred);
        int* pred_classes = Tensor_argmax(y_pred, -1);
        if(pred_classes[0] == y[i]) correct++;
        free(pred_classes);
    }

    printf("accuracy: %.4f\n", (float)correct / n_test_samples);
    cten_end_eval();

    return 0;
}
