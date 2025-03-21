#include "cten.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>  

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

    //shuffling
    int* indices = malloc(n_samples * sizeof(int));
    for (int i = 0; i < n_samples; i++) {
        indices[i] = i;
    }
    // Seed Used and doing Fisher-Yates shuffle
    srand((unsigned)time(NULL));
    for (int i = n_samples - 1; i > 0; i--) {
        int j = rand() % (i + 1);
        int tmp = indices[i];
        indices[i] = indices[j];
        indices[j] = tmp;
    }

    float (*X_shuffled)[4] = malloc(n_samples * sizeof(*X_shuffled));
    int* y_shuffled = malloc(n_samples * sizeof(int));
    for (int i = 0; i < n_samples; i++) {
        int idx = indices[i];
        for (int j = 0; j < n_features; j++) {
            X_shuffled[i][j] = X[idx][j];
        }
        y_shuffled[i] = y[idx];
    }
    free(indices);
    X = (const float(*)[4])X_shuffled;
    y = (const int*)y_shuffled;

    int n_train_samples = n_samples * 0.8; 
    int n_test_samples = n_samples - n_train_samples; 
    
    printf("n_samples: %d\n", n_samples);
    printf("n_train_samples: %d\n", n_train_samples);
    printf("n_test_samples: %d\n", n_test_samples);

    //normalize the dataset
    float(*X_norm)[4] = malloc(n_samples * sizeof(*X_norm));
    Tensor_normalize_dataset(X, X_norm, n_samples, n_train_samples, n_features);
    X = (const float(*)[4])X_norm;

    // create model
    Model model;
    cten_begin_malloc(PoolId_Model);
    model.weight_1 = Glorot_init((TensorShape){n_features, 32}, true);
    model.bias_1 = Tensor_zeros((TensorShape){1, 32}, true);
    model.weight_2 = Glorot_init((TensorShape){32, n_classes}, true);
    model.bias_2 = Tensor_zeros((TensorShape){1, n_classes}, true);
    cten_end_malloc();

    // create optimizer
    cten_begin_malloc(PoolId_Optimizer);
    optim_sgd* optimizer = optim_sgd_new(4, (Tensor*)&model);
    optim_sgd_config(optimizer, 0.01f, 0.0f);
    cten_end_malloc();

    // train model
    int batch_size = 8;
    for(int epoch = 0; epoch < 3; epoch++) {
        printf("==> epoch: %d\n", epoch);
        float epoch_loss = 0.0f;
        int num_batches = 0;
        for(int i = 0; i < n_train_samples; i += batch_size) {
            int actual_batch_size = i + batch_size <= n_train_samples ? batch_size : n_train_samples - i;
            printf(" batch: %d/%d samples\n", i, n_train_samples);
            cten_begin_malloc(PoolId_Default);            
            Tensor input = Tensor_zeros((TensorShape){actual_batch_size, n_features}, false);
            Tensor y_true = Tensor_zeros((TensorShape){actual_batch_size, n_classes}, false);

            for(int j = 0; j < actual_batch_size; j++) {
                for(int k = 0; k < n_features; k++) {
                    input.data->flex[j * n_features + k] = X[i + j][k];
                }
                // one-hot encoding
                y_true.data->flex[j * n_classes + y[i + j]] = 1.0f;

            }
            // zero the gradients
            optim_sgd_zerograd(optimizer);
            // forward pass
            Tensor logit = Model_forward(&model, input);
            Tensor loss = nn_softmax_crossentropy(y_true, logit);
            epoch_loss += loss.data->flex[0];
            num_batches++;
            
            Tensor grad = Tensor_ones((TensorShape){1}, false);
            Tensor_backward(loss, grad);

            
            optim_sgd_step(optimizer);
            cten_end_malloc();
            // free temporary tensors
            cten_free(PoolId_Default);
        }
        printf("Epoch %d average loss: %.6f\n", epoch, epoch_loss / num_batches);
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
        y_true.data->flex[0 * n_classes + y[i]] = 1.0f; //Writing 0 here just to follow the architecture of the code

        // forward pass
        Tensor logit = Model_forward(&model, input);
        Tensor y_pred = nn_softmax(logit);
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
