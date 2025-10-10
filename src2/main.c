#include "cten.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>  

#define PI 3.14159265358979323846

enum MemoryPoolIds {
    PoolId_Default = 0,
    PoolId_Model = 1,
    PoolId_Optimizer = 2,
};

typedef struct {
    Tensor w1, b1;
    Tensor w2, b2;
    Tensor w3, b3;
} Model;

Tensor Model_forward(Model* model, Tensor x) {
    x = nn_linear(x, model->w1, model->b1);
    x = nn_elu(x, 1.0f);
    x = nn_linear(x, model->w2, model->b2);
    x = nn_elu(x, 1.0f);
    x = nn_linear(x, model->w3, model->b3);
    return x;
}

float rand_float() {
    return (float)rand() / (RAND_MAX / 2.0f) - 1.0f;
}

void generate_sine_data(float* x_data, float* y_data, int n_samples, float noise_level) {
    for (int i = 0; i < n_samples; i++) {
        x_data[i] = rand_float() * 4.0f * PI;
        
        // Generate Gaussian noise using the Box-Muller transform
        float u1 = ((float)rand() + 1.0f) / ((float)RAND_MAX + 2.0f);
        float u2 = ((float)rand() + 1.0f) / ((float)RAND_MAX + 2.0f);
        float z = sqrtf(-2.0f * logf(u1)) * cosf(2.0f * PI * u2);
        
        y_data[i] = sin(x_data[i]) + z * noise_level;
    }
}

int main() {
    cten_initilize();

    // Generating Sine Data
    int n_samples = 2048;
    int n_train_samples = n_samples * 0.8;
    int n_test_samples = n_samples - n_train_samples;
    float* x_data = malloc(n_samples * sizeof(float));
    float* y_data = malloc(n_samples * sizeof(float));
    generate_sine_data(x_data, y_data, n_samples, 0.05f);

    // create model
    Model model;
    cten_begin_malloc(PoolId_Model);
    model.w1 = Glorot_init((TensorShape){1, 64}, true);
    model.b1 = Tensor_zeros((TensorShape){1, 64}, true);
    model.w2 = Glorot_init((TensorShape){64, 32}, true);
    model.b2 = Tensor_zeros((TensorShape){1, 32}, true);
    model.w3 = Glorot_init((TensorShape){32, 1}, true);
    model.b3 = Tensor_zeros((TensorShape){1, 1}, true);
    cten_end_malloc();

    // create optimizer
    float learning_rate = 0.01f;
    cten_begin_malloc(PoolId_Optimizer);
    optim_adam* optimizer = optim_adam_new(6, (Tensor*)&model, learning_rate, 0.9f, 0.999f, 1e-8f, 0.0f);
    cten_end_malloc();

    // train model
    int batch_size = 64;
    for (int epoch = 0; epoch < 200; epoch++) {
        // Manual Learning Rate Scheduler
        if (epoch > 0 && epoch % 100 == 0) {
            learning_rate *= 0.7f;
            printf("Epoch %d: Learning rate decreased to %f\n", epoch, learning_rate);
        }

        float total_loss = 0.0f;
        int num_batches = 0;
        for (int i = 0; i < n_train_samples; i += batch_size) {
            int current_batch_size = (i + batch_size > n_train_samples) ? (n_train_samples - i) : batch_size;

            cten_begin_malloc(PoolId_Default);
            Tensor input = Tensor_zeros((TensorShape){current_batch_size, 1}, false);
            Tensor y_true = Tensor_zeros((TensorShape){current_batch_size, 1}, false);

            for (int j = 0; j < current_batch_size; j++) {
                input.data->flex[j] = x_data[i + j];
                y_true.data->flex[j] = y_data[i + j];
            }

            optim_adam_zerograd(optimizer);
            Tensor y_pred = Model_forward(&model, input);
            
            // Combined Loss: Huber + 30% MAE
            Tensor huber = nn_huber_loss(y_true, y_pred, 1.0f);
            Tensor mae = nn_mae_loss(y_true, y_pred);
            Tensor loss = Tensor_add(huber, Tensor_mulf(mae, 0.3f));

            total_loss += loss.data->flex[0];
            num_batches++;

            Tensor_backward(loss, Tensor_ones((TensorShape){1}, false));
            
            // Gradient Clipping
            cten_clip_grad_norm((Tensor*)&model, 6, 5.0f);

            optim_adam_step(optimizer);
            cten_end_malloc();
            // free temporary tensors
            cten_free(PoolId_Default);
        }
        if (epoch % 50 == 0) {
            printf("Epoch %d, Average Loss: %.6f\n", epoch, total_loss / num_batches);
        }
    }

    // free optimizer
    cten_free(PoolId_Optimizer);

    // evaluate model
    cten_begin_eval();
    float total_test_mse = 0;
    for (int i = n_train_samples; i < n_samples; i++) {
        cten_begin_malloc(PoolId_Default);
        Tensor input = Tensor_zeros((TensorShape){1, 1}, false);
        input.data->flex[0] = x_data[i];
        
        Tensor y_pred = Model_forward(&model, input);

        float true_val = y_data[i];
        float pred_val = y_pred.data->flex[0];
        total_test_mse += (true_val - pred_val) * (true_val - pred_val);
        
        if (i%50 == 0) {
             printf("Input: %.3f, True: %.3f, Predicted: %.3f\n", x_data[i], true_val, pred_val);
        }
        cten_free(PoolId_Default);
    }
    printf("Final Test MSE: %.6f\n", total_test_mse / n_test_samples);
    cten_end_eval();

    // free model
    cten_free(PoolId_Model); 

    cten_finalize();
    return 0;
}
