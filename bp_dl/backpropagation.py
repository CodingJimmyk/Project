#utf-8 code
import random
import math

class Neuron:
    def __init__(self, bias):
        self.bias = bias
        self.weights = []

    def calculate_output(self, inputs):
        self.inputs = inputs
        self.output = self.activator(self.calculate_total_net_input())
        return self.output

    def calculate_total_net_input(self):
        total = 0
        for i in range(len(self.inputs)):
            total += self.inputs[i] * self.weights[i]
        total += self.bias
        return total

    # 激活函数sigmoid
    def activator(self, total_net_input):
        return 1 / (1 + math.exp(-total_net_input))

    def calculate_pd_errtotal_mul_pd_out(self, target_output):
        # (∂Etotal/∂Out_o) * (∂Out_o/∂Net_o)
        pd_errtotal = self.calculate_pd_error_total(target_output)
        pd_out      = self.calculate_pd_out_net()
        return pd_errtotal * pd_out

    # 每一个神经元的误差是由平方差公式计算的
    def calculate_error(self, target_output):
        return 0.5 * (target_output - self.output) ** 2

    def calculate_pd_error_total(self, target_output):
        return -(target_output - self.output)

    def calculate_pd_out_net(self):
        return self.output * (1 - self.output)

    def calculate_net_formula_pd_weight(self, index):
        return self.inputs[index]

class NeuronLayer:
    def __init__(self, num_neurons, bias):
        # 同一层的神经元共享一个截距项b
        if bias != 0.0:
            self.bias = bias
        else:
            self.bias = random.random()

        self.neurons = []
        for i in range(num_neurons):
            self.neurons.append(Neuron(self.bias))

    def inspect_NL(self):
        for n in range(len(self.neurons)):
            print(' Neuron', n)
            for w in range(len(self.neurons[n].weights)):
                print('  Weight:', self.neurons[n].weights[w])
            print('  Bias:', self.bias)

    def feed_forward(self, inputs):
        outputs = []
        for neuron in self.neurons:
            outputs.append(neuron.calculate_output(inputs))
        return outputs

    def get_outputs(self):
        outputs = []
        for neuron in self.neurons:
            outputs.append(neuron.output)
        return outputs

class NeuralNetwork:
    LEARNING_RATE = 0.5

    def __init__(self, num_inputs, num_hidden, num_outputs, hidden_layer_weights = None, hidden_layer_bias = None, output_layer_weights = None, output_layer_bias = None):
        self.num_inputs = num_inputs

        self.hidden_layer = NeuronLayer(num_hidden, hidden_layer_bias)
        self.output_layer = NeuronLayer(num_outputs, output_layer_bias)

        self.init_weights_from_inputs_to_hidden_layer_neurons(hidden_layer_weights)
        self.init_weights_from_hidden_layer_neurons_to_output_layer_neurons(output_layer_weights)

    def init_weights_from_inputs_to_hidden_layer_neurons(self, hidden_layer_weights):
        weight_num = 0
        for h in range(len(self.hidden_layer.neurons)):
            for i in range(self.num_inputs):
                if not hidden_layer_weights:
                    self.hidden_layer.neurons[h].weights.append(random.random())
                else:
                    self.hidden_layer.neurons[h].weights.append(hidden_layer_weights[weight_num])
                weight_num += 1

    def init_weights_from_hidden_layer_neurons_to_output_layer_neurons(self, output_layer_weights):
        weight_num = 0
        for o in range(len(self.output_layer.neurons)):
            for h in range(len(self.hidden_layer.neurons)):
                if not output_layer_weights:
                    self.output_layer.neurons[o].weights.append(random.random())
                else:
                    self.output_layer.neurons[o].weights.append(output_layer_weights[weight_num])
                weight_num += 1

    def feed_forward(self, inputs):
        hidden_layer_outputs = self.hidden_layer.feed_forward(inputs)
        return self.output_layer.feed_forward(hidden_layer_outputs)

    def feed_backward(self, targets_outputs):
        # 1. 输出神经元的值 backward
        pd_errtotal_mul_pd_out = [None] * len(self.output_layer.neurons)
        for o in range(len(self.output_layer.neurons)):
            # (∂Etotal/∂Out_o) * (∂Out_o/∂Net_o), partial derivative
            pd_errtotal_mul_pd_out[o] = self.output_layer.neurons[o].calculate_pd_errtotal_mul_pd_out(targets_outputs[o])

        # 2. 隐含層神经元的值 do before weight update
        pd_errtotal_mul_pd_hidden_output = [None] * len(self.hidden_layer.neurons)
        for h in range(len(self.hidden_layer.neurons)):
            sum_error_total_hidden_neuron_output = 0
            for o in range(len(self.output_layer.neurons)):
                # ∂Etotal / ∂Out_h = (∂Etotal / ∂Out_o) * (∂Out_o / ∂Net_o) *  [(∂Net_o/∂Out_h)]
                pd_errtotal_outh = pd_errtotal_mul_pd_out[o] * self.output_layer.neurons[o].weights[h]
                # Σ ∂Etotal / ∂Out_h
                sum_error_total_hidden_neuron_output += pd_errtotal_outh
            # (∂Out_h/∂Net_h)
            pd_outh_neth = self.hidden_layer.neurons[h].calculate_pd_out_net()
            # ∂Etotal/∂Net_h = (∂Etotal / ∂Out_h) * (∂Out_h / ∂Net_h)
            pd_errtotal_mul_pd_hidden_output[h] = sum_error_total_hidden_neuron_output * pd_outh_neth

        # 3. 更新输出層權重系数
        for o in range(len(self.output_layer.neurons)):
            for w_h2o in range(len(self.output_layer.neurons[o].weights)):
                # [(∂Net_o / ∂W)]
                pd_neto_w = self.output_layer.neurons[o].calculate_net_formula_pd_weight(w_h2o)
                # ∂Etotal/∂W = (∂Etotal/∂Out_o) * (∂Out_o/∂Net_o) * [(∂Net_o/∂W)]
                pd_errortotal_weight = pd_errtotal_mul_pd_out[o] * pd_neto_w
                # Δw = η * ∂Etotal/∂W
                self.output_layer.neurons[o].weights[w_h2o] -= self.LEARNING_RATE * pd_errortotal_weight

        # 4. 更新隱含層的權重系数
        for h in range(len(self.hidden_layer.neurons)):
            for w_i2h in range(len(self.hidden_layer.neurons[h].weights)):
                # (∂Net_h / ∂W)
                pd_neth_w = self.hidden_layer.neurons[h].calculate_net_formula_pd_weight(w_i2h)
                # ∂Etotal/∂W = ∂Etotal/∂Net_h * (∂Net_h/∂W)
                pd_errortotal_weight = pd_errtotal_mul_pd_hidden_output[h] * pd_neth_w
                # Δw = η * ∂Etotal/∂W
                self.hidden_layer.neurons[h].weights[w_i2h] -= self.LEARNING_RATE * pd_errortotal_weight

    def train(self, training_inputs, targets_outputs):
        self.feed_forward(training_inputs)
        self.feed_backward(targets_outputs)

    def inspect_net(self):
        print('-Hidden Layer-')
        self.hidden_layer.inspect_NL()
        print('-Output Layer-')
        self.output_layer.inspect_NL()

    def calculate_total_loss(self, training_sets):
        total_error = 0
        for t in range(len(training_sets)):
            training_inputs, expect_outputs = training_sets[t]
            self.feed_forward(training_inputs)
            for o in range(len(expect_outputs)):
                total_error += self.output_layer.neurons[o].calculate_error(expect_outputs[o])
        return total_error

if __name__ == "__main__":
    nn = NeuralNetwork(num_inputs=2,
                       num_hidden=2,
                       num_outputs=2,
                       hidden_layer_weights=[0.15, 0.2, 0.25, 0.3],
                       hidden_layer_bias=0.35,
                       output_layer_weights=[0.4, 0.45, 0.5, 0.55],
                       output_layer_bias=0.6)

    nn.inspect_net()
    for i in range(1000):
        nn.train([0.05, 0.1], [0.01, 0.99])
        print('***After Train***')
        nn.inspect_net()
        # after point show 9 digits
        print("Iter No." ,i ,"| total_loss =", round(nn.calculate_total_loss([[[0.05, 0.1], [0.01, 0.99]]]), 9))
        print('======================')
