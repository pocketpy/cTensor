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
    x = nn_softmax(x);
    return x;
}

//TESTING PURPOSE ONLY
void Model_backward(Model* model, Tensor input, Tensor y_true, Tensor y_pred) {
    int batch_size = input.shape[0];
    int n_features = input.shape[1];
    int n_hidden = model->weight_1.shape[1];
    int n_classes = model->weight_2.shape[1];

    Tensor output_grad = Tensor_ones((TensorShape){batch_size, n_classes}, false);
    for (int i = 0; i < batch_size; i++) {
        for (int j = 0; j < n_classes; j++) {
            output_grad.data->flex[i * n_classes + j] = 
                y_pred.data->flex[i * n_classes + j] - y_true.data->flex[i * n_classes + j];
        }
    }
    printf("Output gradient:\n");
    Tensor_print(output_grad);
    
    Tensor hidden_output = nn_linear(input, model->weight_1, model->bias_1);
    hidden_output = nn_relu(hidden_output);
    
    Tensor weight2_grad = Tensor_zeros(model->weight_2.shape, false);
    for (int b = 0; b < batch_size; b++) {
        for (int i = 0; i < n_hidden; i++) {
            for (int j = 0; j < n_classes; j++) {
                weight2_grad.data->flex[i * n_classes + j] += 
                    hidden_output.data->flex[b * n_hidden + i] * output_grad.data->flex[b * n_classes + j];
            }
        }
    }
    
    Tensor bias2_grad = Tensor_zeros(model->bias_2.shape, false);
    for (int b = 0; b < batch_size; b++) {
        for (int j = 0; j < n_classes; j++) {
            bias2_grad.data->flex[j] += output_grad.data->flex[b * n_classes + j];
        }
    }

    
    Tensor weight2_T = Tensor_transpose(model->weight_2);
    Tensor hidden_grad = Tensor_zeros(hidden_output.shape, false);
    for (int i = 0; i < batch_size; i++) {
        for (int j = 0; j < n_hidden; j++) {
            for (int k = 0; k < n_classes; k++) {
                hidden_grad.data->flex[i * n_hidden + j] += 
                    output_grad.data->flex[i * n_classes + k] * weight2_T.data->flex[j * n_classes + k];
            }
        }
    }
    
    Tensor hidden_input = nn_linear(input, model->weight_1, model->bias_1);
    for (int i = 0; i < batch_size * n_hidden; i++) {
        if (hidden_input.data->flex[i] <= 0) {
            hidden_grad.data->flex[i] = 0;
        }
    }

    Tensor weight1_grad = Tensor_zeros(model->weight_1.shape, false);
    for (int b = 0; b < batch_size; b++) {
        for (int i = 0; i < n_features; i++) {
            for (int j = 0; j < n_hidden; j++) {
                weight1_grad.data->flex[i * n_hidden + j] += 
                    input.data->flex[b * n_features + i] * hidden_grad.data->flex[b * n_hidden + j];
            }
        }
    }
    printf("Weight 1 gradient:\n");
    Tensor_print(weight1_grad);
    
    Tensor bias1_grad = Tensor_zeros(model->bias_1.shape, false);
    for (int b = 0; b < batch_size; b++) {
        for (int j = 0; j < n_hidden; j++) {
            bias1_grad.data->flex[j] += hidden_grad.data->flex[b * n_hidden + j];
        }
    }
    printf("Bias 1 gradient:\n");
    Tensor_print(bias1_grad);
    
    if (model->weight_1.node != NULL) {
        model->weight_1.node->grad = weight1_grad;
    }
    if (model->bias_1.node != NULL) {
        model->bias_1.node->grad = bias1_grad;
    }
    if (model->weight_2.node != NULL) {
        model->weight_2.node->grad = weight2_grad;
    }
    if (model->bias_2.node != NULL) {
        model->bias_2.node->grad = bias2_grad;
    }
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
    float mean[4] = {0}, std[4] = {0};
    for (int i = 0; i < n_train_samples; i++) {
        for (int j = 0; j < n_features; j++) {
            mean[j] += X[i][j];
        }
    }
    for (int j = 0; j < n_features; j++) {
        mean[j] /= n_train_samples;
    }
    for (int i = 0; i < n_train_samples; i++) {
        for (int j = 0; j < n_features; j++) {
            std[j] += (X[i][j] - mean[j]) * (X[i][j] - mean[j]);
        }
    }
    for (int j = 0; j < n_features; j++) {
        std[j] = sqrtf(std[j] / n_train_samples);
        // Avoid division by zero
        if (std[j] == 0) std[j] = 1.0f;
    }

    // Normalize the entire dataset
    float(*X_norm)[4] = malloc(n_samples * sizeof(*X_norm));
    for (int i = 0; i < n_samples; i++) {
        for (int j = 0; j < n_features; j++) {
            X_norm[i][j] = (X[i][j] - mean[j]) / std[j];
        }
    }
    X = (const float(*)[4])X_norm;

    // create model
    Model model;
    cten_begin_malloc(PoolId_Model);
    model.weight_1 = nn_random_init((TensorShape){n_features, 32}, true);
    model.bias_1 = Tensor_zeros((TensorShape){1, 32}, true);
    model.weight_2 = nn_random_init((TensorShape){32, n_classes}, true);
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
        float epoch_loss = 0.0f;
        int num_batches = 0;
        for(int i = 0; i < n_train_samples; i += batch_size) {
            int actual_batch_size = i + batch_size <= n_train_samples ? batch_size : n_train_samples - i;
            
            cten_begin_malloc(PoolId_Default);
            
            // Debug print
            // printf("Batch %d: using %d samples\n", i/batch_size, actual_batch_size);
            
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
            Tensor y_pred = Model_forward(&model, input);
            Tensor loss = nn_crossentropy(y_true, y_pred);
            epoch_loss += loss.data->flex[0];
            num_batches++;
            
            // backward pass
            printf("Backward pass\n");
            Tensor grad = Tensor_ones((TensorShape){actual_batch_size,n_classes}, false);
            Tensor_backward(loss, grad);


            // printf("\nStarting custom backpropagation\n");
            // Model_backward(&model, input, y_true, y_pred);
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
