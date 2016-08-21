using System;
using System.Collections.Generic;
using System.Text;
using System.IO;

namespace Complex_Neural_Network
{
    class Program
    {
        static void Main()
        {
            /*
             *      Test Platform
             *      The current test is to map 8 points around a circle to a single line horizontal line
             */

            // set constant PI
            const double PI = 3.141592653589793238462643383279502884197169399375;

            // Declare Neural Network
            NeuralNetwork NN = new NeuralNetwork(new int[] { 1, 360, 1 });

            // Declare and initialize input array
            Complex[,] input = new Complex[,] { { 0.9 * Complex.fromAngle(2 * PI * 0 / 8),
                                                  0.9 * Complex.fromAngle(2 * PI * 1 / 8),
                                                  0.9 * Complex.fromAngle(2 * PI * 2 / 8),
                                                  0.9 * Complex.fromAngle(2 * PI * 3 / 8),
                                                  0.9 * Complex.fromAngle(2 * PI * 4 / 8),
                                                  0.9 * Complex.fromAngle(2 * PI * 5 / 8),
                                                  0.9 * Complex.fromAngle(2 * PI * 6 / 8),
                                                  0.9 * Complex.fromAngle(2 * PI * 7 / 8) } };

            // Declare and initialize target output array
            Complex[,] output = new Complex[,] { {0.9 * Complex.fromAngle(2 * PI * 0 / 8),
                                                  0.675 * Complex.fromAngle(2 * PI * 0 / 8),
                                                  0.45 * Complex.fromAngle(2 * PI * 0 / 8),
                                                  0.225 * Complex.fromAngle(2 * PI * 0 / 8),
                                                  0.225 * Complex.fromAngle(2 * PI * 4 / 8),
                                                  0.45 * Complex.fromAngle(2 * PI * 4 / 8),
                                                  0.675 * Complex.fromAngle(2 * PI * 4 / 8),
                                                  0.9 * Complex.fromAngle(2 * PI * 4 / 8) } };

            // Train network with backpropagation
            NN.backProp(input, output, 0.01, 0.01);

            //  Declare output variable
            Complex[,] testOut;

            //  Get output with feed forward
            NN.forwardProp(input, out testOut, -1);

        }


    }
}
