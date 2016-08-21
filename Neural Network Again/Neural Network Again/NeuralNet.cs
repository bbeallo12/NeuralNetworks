using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Neural_Network_Again
{
    class NeuralNet
    {
        // initialize variables
        Random rand = new Random();
        int inputCount, outputCount, layerCount;
        int topologyHash;
        int biasHash;
        int[] topology;

        // Weights
        double[][,] weights;
        double[][,] deltaWeights;
        // Bias
        double[][,] bias;
        double[][,] deltaBias;
        // Node Values
        double[][,] values;

        double randomMultiplier { get; set; }

        // constructor, create neural network based on topology
        public NeuralNet(int[] Topology)
        {
            randomMultiplier = 1.0;
            topology = Topology;
            layerCount = topology.Length;
            inputCount = topology[0];
            outputCount = topology[layerCount - 1];
            weights = new double[layerCount - 1][,];
            bias = new double[layerCount - 1][,];
            deltaWeights = new double[layerCount - 1][,];
            deltaBias = new double[layerCount - 1][,];
            values = new double[layerCount][,];
            topologyHash = 0;
            biasHash = 1;
            // The hash is just a simple hash to compare the topologies of a network with the topologies of loaded weight values
            // to make sure they are compatible
            for (int i = 0; i < topology.Length - 1; i++)
            {
                if (i == 0)
                {
                    topologyHash += topology[topology.Length - 1] * topology[topology.Length - 1];
                }
                topologyHash += topology[i] * topology[i];
                biasHash += topology[i + 1] * topology[i + 1];
                // initialize weights and bias matrices
                // also deltaweights and deltabias for the error adjustments
                weights[i] = new double[topology[i + 1], topology[i]];
                bias[i] = new double[topology[i + 1], 1];
                deltaWeights[i] = new double[topology[i + 1], topology[i]];
                deltaBias[i] = new double[topology[i + 1], 1];
                for (int x = 0; x < topology[i + 1]; x++)
                {
                    bias[i][x, 0] = randomMultiplier * (rand.NextDouble() - 0.5);
                    for (int y = 0; y < topology[i]; y++)
                    {
                        // set random weights
                        weights[i][x, y] = randomMultiplier * (rand.NextDouble() - 0.5);
                    }
                }
            }

        }
        public NeuralNet(int[] Topology, double randomness)
        {
            randomMultiplier = randomness;
            topology = Topology;
            layerCount = topology.Length;
            inputCount = topology[0];
            outputCount = topology[layerCount - 1];
            weights = new double[layerCount - 1][,];
            bias = new double[layerCount - 1][,];
            deltaWeights = new double[layerCount - 1][,];
            deltaBias = new double[layerCount - 1][,];
            values = new double[layerCount][,];
            topologyHash = 0;
            biasHash = 1;
            for (int i = 0; i < topology.Length - 1; i++)
            {
                if (i == 0)
                {
                    topologyHash += topology[topology.Length - 1] * topology[topology.Length - 1];
                }
                topologyHash += topology[i] * topology[i];
                biasHash += topology[i + 1] * topology[i + 1];
                weights[i] = new double[topology[i + 1], topology[i]];
                bias[i] = new double[topology[i + 1], 1];
                deltaWeights[i] = new double[topology[i + 1], topology[i]];
                deltaBias[i] = new double[topology[i + 1], 1];
                for (int x = 0; x < topology[i + 1]; x++)
                {
                    bias[i][x, 0] = randomMultiplier * rand.NextDouble() - 1;
                    for (int y = 0; y < topology[i]; y++)
                    {
                        weights[i][x, y] = randomMultiplier * rand.NextDouble() - 1;
                    }
                }
            }

        }

        // feed forwad function to calculate outputs
        public void feedForward(double[,] Input, out double[,] Output)
        {
            for (int i = 0; i < layerCount; i++)
            {
                // declare values arrays
                values[i] = new double[topology[i], Input.GetLength(1)];
            }
            // copy input values to first layer of values
            copyArray(Input, out values[0]);

            for (int w = 0; w < weights.Length; w++)
            {
                // Values(n+1) = activate(Values(n) * Weights(n) + bias(n))
                matrixMult(values[w], weights[w], out values[w + 1]);
                matrixAddRow(values[w + 1], bias[w]);
                activate(values[w + 1]);
            }
            // copy last layer of values to output
            copyArray(values[layerCount - 1], out Output);

        }

        // back propagation
        public bool backPropogation(double[,] Input, double[,] Output, double LearnRate, out double Error)
        {
            // make sure input and outputs are the right size
            if (Input.GetLength(0) != inputCount)
            {
                Error = -1;
                return false;
            }
            if (Output.GetLength(0) != outputCount)
            {
                Error = -1;
                return false;
            }

            double[,] testOutput;
            feedForward(Input, out testOutput); // get current output
            double[,] backPropError;
            double[,] temp;
            // (y_hat - y) * f'(Z_3) * dZ_3/dW_2
            // calculate error
            Error = matrixError(Output, testOutput);
            // calculate act'
            activateDer(testOutput, out backPropError);
            matrixSub(testOutput, Output);
            scalarMult(backPropError, testOutput);

            for (int w = weights.Length - 1; w >= 0; w--)
            {
                // calculate back propagations
                matrixTranspose(values[w], out temp);
                matrixMult(temp, backPropError, out deltaWeights[w]);
                matrixSet(temp);
                matrixMult(temp, backPropError, out deltaBias[w]);
                matrixTranspose(weights[w], out temp);
                matrixMult(ref backPropError, temp);
                activateDer(values[w], out temp);
                scalarMult(backPropError, temp);
            }

            for (int w = 0; w < weights.Length; w++)
            {
                // add change in weights
                scalarMult(deltaWeights[w], LearnRate);
                matrixSub(weights[w], deltaWeights[w]);
                scalarMult(deltaBias[w], LearnRate);
                matrixSub(bias[w], deltaBias[w]);
            }

            return true;
        }

        // set weights and bias from files
        public bool setWeights(double[][,] W)
        {
            if (W.Length != weights.Length)
            {
                return false;
            }
            if (matrixHash(W) != topologyHash)
            {
                return false;
            }

            for (int w = 0; w < W.Length; w++)
            {
                for (int x = 0; x < W[w].GetLength(0); x++)
                {
                    for (int y = 0; y < W[w].GetLength(1); y++)
                    {
                        weights[w][x, y] = W[w][x, y];
                    }
                }
            }

            return true;
        }
        public bool setBias(double[][,] B)
        {
            if (B.Length != bias.Length)
            {
                return false;
            }
            if (matrixHash(B) != biasHash)
            {
                return false;
            }

            for (int w = 0; w < B.Length; w++)
            {
                for (int x = 0; x < B[w].GetLength(0); x++)
                {
                    for (int y = 0; y < B[w].GetLength(1); y++)
                    {
                        bias[w][x, y] = B[w][x, y];
                    }
                }
            }

            return true;
        }
        // return weights and bias for output to file
        public void getWeights(out double[][,] OUT)
        {
            OUT = weights;
        }
        public void getBias(out double[][,] OUT)
        {
            OUT = bias;
        }
        // copy array function
        static private void copyArray(double[,] A, out double[,] B)
        {
            B = new double[A.GetLength(0), A.GetLength(1)];
            for (int x = 0; x < A.GetLength(0); x++)
            {
                for (int y = 0; y < A.GetLength(1); y++)
                {
                    B[x, y] = A[x, y];
                }
            }
        }
        static private void copyArray(double[,] A, double[,] B)
        {
            for (int x = 0; x < B.GetLength(0); x++)
            {
                for (int y = 0; y < B.GetLength(1); y++)
                {
                    B[x, y] = A[x, y];
                }
            }
        }
        // activation function and derivative
        static private void activate(double[,] Matrix)
        {
            for (int x = 0; x < Matrix.GetLength(0); x++)
            {
                for (int y = 0; y < Matrix.GetLength(1); y++)
                {
                    Matrix[x, y] = 1.0 / (1 + Math.Exp(-Matrix[x, y]));
                }
            }
        }
        static private void activateDer(double[,] Matrix, out double[,] C)
        {
            C = new double[Matrix.GetLength(0), Matrix.GetLength(1)];
            for (int x = 0; x < Matrix.GetLength(0); x++)
            {
                for (int y = 0; y < Matrix.GetLength(1); y++)
                {
                    C[x, y] = Matrix[x, y] * (1 - Matrix[x, y]);
                }
            }
        }
        // calculate hash
        static private int matrixHash(double[][,] M)
        {
            int hash = 0;

            for (int i = 0; i < M.Length; i++)
            {
                if (i == 0)
                {
                    hash += M[0].GetLength(1) * M[0].GetLength(1);
                }
                hash += M[i].GetLength(0) * M[i].GetLength(0);
            }

            return hash;
        }
        // calculate error
        static private double matrixError(double[,] A, double[,] B)
        {
            double error = 0;
            for (int x = 0; x < A.GetLength(0); x++)
            {
                for (int y = 0; y < A.GetLength(1); y++)
                {
                    error += (A[x, y] - B[x, y]) * (A[x, y] - B[x, y]);
                }
            }
            return error / 2;
        }
        // matrix operations
        static private void matrixMult(ref double[,] A, double[,] B)
        {
            int X = B.GetLength(0), Y = A.GetLength(1), Z = A.GetLength(0);
            double[,] C = new double[X, Y];
            for (int x = 0; x < X; x++)
            {
                for (int y = 0; y < Y; y++)
                {
                    C[x, y] = 0;
                    for (int z = 0; z < Z; z++)
                    {
                        C[x, y] += A[z, y] * B[x, z];
                    }
                }
            }
            A = C;
        }
        static private void matrixMult(double[,] A, double[,] B, out double[,] C)
        {
            int X = B.GetLength(0), Y = A.GetLength(1), Z = A.GetLength(0);
            C = new double[X, Y];
            for (int x = 0; x < X; x++)
            {
                for (int y = 0; y < Y; y++)
                {
                    C[x, y] = 0;
                    for (int z = 0; z < Z; z++)
                    {
                        C[x, y] += A[z, y] * B[x, z];
                    }
                }
            }
        }
        static private void matrixTranspose(double[,] A, out double[,] B)
        {
            int X = A.GetLength(0), Y = A.GetLength(1);
            B = new double[Y, X];
            for (int x = 0; x < X; x++)
            {
                for (int y = 0; y < Y; y++)
                {
                    B[y, x] = A[x, y];
                }
            }
            A = B;
        }
        static private void scalarMult(double[,] A, double B)
        {
            for (int x = 0; x < A.GetLength(0); x++)
            {
                for (int y = 0; y < A.GetLength(1); y++)
                {
                    A[x, y] *= B;
                }
            }
        }
        static private void scalarMult(double[,] A, double[,] B)
        {
            for (int x = 0; x < A.GetLength(0); x++)
            {
                for (int y = 0; y < A.GetLength(1); y++)
                {
                    A[x, y] *= B[x, y];
                }
            }
        }
        static private void matrixAdd(double[,] A, double[,] B)
        {
            for (int x = 0; x < A.GetLength(0); x++)
            {
                for (int y = 0; y < A.GetLength(1); y++)
                {
                    A[x, y] += B[x, y];
                }
            }
        }
        static private void matrixAddRow(double[,] A, double[,] B)
        {
            for (int x = 0; x < A.GetLength(0); x++)
            {
                for (int y = 0; y < A.GetLength(1); y++)
                {
                    A[x, y] += B[x, 0];
                }
            }
        }
        static private void matrixSub(double[,] A, double[,] B)
        {
            for (int x = 0; x < A.GetLength(0); x++)
            {
                for (int y = 0; y < A.GetLength(1); y++)
                {
                    A[x, y] -= B[x, y];
                }
            }
        }
        static private void matrixSet(double[,] A)
        {
            for (int x = 0; x < A.GetLength(0); x++)
            {
                for (int y = 0; y < A.GetLength(1); y++)
                {
                    A[x, y] = 1;
                }
            }
        }
    }
}
