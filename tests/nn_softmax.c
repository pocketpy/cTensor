#include "cten.h"
#include <stdio.h>
#include <stdlib.h>

void print_grad_data(Tensor input)
{
	int cnt = input.node->grad.data->numel;
	printf("grad elments count : %d\t", cnt);
	for (int i = 0; i < cnt; i++)
	{
		printf("%f\t", input.node->grad.data->flex[i]);
	}
	printf("\n");
}


int main()
{
	Tensor a = Tensor_new((TensorShape) { 4 }, true);
	a.data->flex[0] = 1.0;
	a.data->flex[1] = 2.0;
	a.data->flex[2] = 3.0;
	a.data->flex[3] = 4.0;

	Tensor b = nn_softmax(a);

	printf("a\t\tafter softmax\n");
	for (int i = 0; i < a.data->numel; i++)
	{
		printf("%f\t%f\n", a.data->flex[i], b.data->flex[i]);
	}

	Tensor c = Tensor_sum(b);

	printf("%f\n", c.data->flex[0]);

	Tensor_backward(c, (Tensor) { NULL });

	printf("%d\n", TensorShape_dim(b.shape));

	print_grad_data(a);
	print_grad_data(b);
	print_grad_data(c);

	Tensor aa = a;
	aa = Tensor_mul(aa, a);
	for (int i = 0; i < a.data->numel; i++)
	{
		printf("%f\n", aa.data->flex[i]);
	}


	Tensor_delete(a);
	Tensor_delete(b);
	Tensor_delete(c);

	return 0;
}