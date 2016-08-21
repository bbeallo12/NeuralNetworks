using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NeuralNet_BallTraining
{
    class NeuralNet
    {
        Random rand = new Random();
        int inputCount, outputCount, layerCount;
        int topologyHash;
        public double gravity { get; set; }
        public double friction { get; set; }
        int biasHash;
        double randomMultiplier { get; set; }
        double randomOffset { get; set; }
        int[] topology;

        // Weight attributes
        double[][,] weights;
        double[][,] weightDeltas;
        double[][,] weightVelocities;
        double[][,] weightAccelerations;

        // Bias attributes
        double[][,] bias;
        double[][,] biasDeltas;
        double[][,] biasVelocities;
        double[][,] biasAccelerations;

        // Node attributes
        double[][,] values;

        public NeuralNet(int[] Topology)
        {
            randomMultiplier = 1.0;
            randomOffset = 0.0;
            topology = Topology;
            layerCount = topology.Length;
            inputCount = topology[0];
            outputCount = topology[layerCount - 1];

            //Weights
            weights = new double[layerCount - 1][,];
            weightDeltas = new double[layerCount - 1][,];
            weightVelocities = new double[layerCount - 1][,];
            weightAccelerations = new double[layerCount - 1][,];


            //Bias
            bias = new double[layerCount - 1][,];
            biasDeltas = new double[layerCount - 1][,];
            biasVelocities = new double[layerCount - 1][,];
            biasAccelerations = new double[layerCount - 1][,];


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

                // Weights
                weights[i] = new double[topology[i + 1], topology[i]];
                weightDeltas[i] = new double[topology[i + 1], topology[i]];
                weightVelocities[i] = new double[topology[i + 1], topology[i]];
                weightAccelerations[i] = new double[topology[i + 1], topology[i]];

                // Bias
                bias[i] = new double[topology[i + 1], 1];
                biasDeltas[i] = new double[topology[i + 1], 1];
                biasVelocities[i] = new double[topology[i + 1], 1];
                biasAccelerations[i] = new double[topology[i + 1], 1];

                for (int x = 0; x < topology[i + 1]; x++)
                {
                    bias[i][x, 0] = randomMultiplier * (rand.NextDouble() - 0.5) + randomOffset;
                    for (int y = 0; y < topology[i]; y++)
                    {
                        weights[i][x, y] = randomMultiplier * (rand.NextDouble() - 0.5) + randomOffset;
                    }
                }
            }

        }
        public NeuralNet(int[] Topology, double randomness)
        {
            randomMultiplier = randomness;
            randomOffset = 0.0;
            topology = Topology;
            layerCount = topology.Length;
            inputCount = topology[0];
            outputCount = topology[layerCount - 1];

            //Weights
            weights = new double[layerCount - 1][,];
            weightDeltas = new double[layerCount - 1][,];
            weightVelocities = new double[layerCount - 1][,];
            weightAccelerations = new double[layerCount - 1][,];


            //Bias
            bias = new double[layerCount - 1][,];
            biasDeltas = new double[layerCount - 1][,];
            biasVelocities = new double[layerCount - 1][,];
            biasAccelerations = new double[layerCount - 1][,];


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

                // Weights
                weights[i] = new double[topology[i + 1], topology[i]];
                weightDeltas[i] = new double[topology[i + 1], topology[i]];
                weightVelocities[i] = new double[topology[i + 1], topology[i]];
                weightAccelerations[i] = new double[topology[i + 1], topology[i]];

                // Bias
                bias[i] = new double[topology[i + 1], 1];
                biasDeltas[i] = new double[topology[i + 1], 1];
                biasVelocities[i] = new double[topology[i + 1], 1];
                biasAccelerations[i] = new double[topology[i + 1], 1];

                for (int x = 0; x < topology[i + 1]; x++)
                {
                    bias[i][x, 0] = randomMultiplier * (rand.NextDouble() - 0.5) + randomOffset;
                    for (int y = 0; y < topology[i]; y++)
                    {
                        weights[i][x, y] = randomMultiplier * (rand.NextDouble() - 0.5) + randomOffset;
                    }
                }
            }

        }

        public void resetVelocities()
        {
            for (int i = 0; i < topology.Length - 1; i++)
            {
                // Weights
                weightVelocities[i] = new double[topology[i + 1], topology[i]];
                // Bias
                biasVelocities[i] = new double[topology[i + 1], 1];
            }
        }

        public void feedForward(double[,] Input, out double[,] Output)
        {
            for (int i = 0; i < layerCount; i++)
            {
                values[i] = new double[topology[i], Input.GetLength(1)];
            }

            copyArray(Input, out values[0]);

            for (int w = 0; w < weights.Length; w++)
            {
                matrixMult(values[w], weights[w], out values[w + 1]);
                matrixAddRow(values[w + 1], bias[w]);
                activate(values[w + 1]);
            }

            copyArray(values[layerCount - 1], out Output);

        }

        public bool backPropogationBall(double[,] Input, double[,] Output, double G, double MU, out double Error)
        {
            gravity = G;
            friction = MU;
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
            feedForward(Input, out testOutput);
            double[,] backPropError;
            double[,] temp;
            // (y_hat - y) * f'(Z_3) * dZ_3/dW_2
            Error = matrixError(Output, testOutput);
            activateDer(testOutput, out backPropError);
            matrixSub(testOutput, Output);
            scalarMult(backPropError, testOutput);

            for (int w = weights.Length - 1; w >= 0; w--)
            {
                matrixTranspose(values[w], out temp);
                matrixMult(temp, backPropError, out weightDeltas[w]);
                matrixSet(ref temp);
                matrixMult(temp, backPropError, out biasDeltas[w]);
                matrixTranspose(weights[w], out temp);
                matrixMult(ref backPropError, temp);
                activateDer(values[w], out temp);
                scalarMult(backPropError, temp);
            }

            for (int w = 0; w < weights.Length; w++)
            {
                accelCalc(weightDeltas[w], weightVelocities[w], weightAccelerations[w]);
                matrixAdd(weightVelocities[w], weightAccelerations[w]);
                matrixSin(weightVelocities[w], weightDeltas[w], out temp);
                matrixAdd(weights[w], temp);

                accelCalc(biasDeltas[w], biasVelocities[w], biasAccelerations[w]);
                matrixAdd(biasVelocities[w], biasAccelerations[w]);
                matrixSin(biasVelocities[w], biasDeltas[w], out temp);
                matrixAdd(bias[w], temp);
            }

            return true;
        }

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

        public void getWeights(out double[][,] OUT)
        {
            OUT = weights;
        }
        public void getBias(out double[][,] OUT)
        {
            OUT = bias;
        }

        private void copyArray(double[,] A, out double[,] B)
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
        private void copyArray(double[,] A, double[,] B)
        {
            for (int x = 0; x < B.GetLength(0); x++)
            {
                for (int y = 0; y < B.GetLength(1); y++)
                {
                    B[x, y] = A[x, y];
                }
            }
        }

        private void activate(double[,] Matrix)
        {
            for (int x = 0; x < Matrix.GetLength(0); x++)
            {
                for (int y = 0; y < Matrix.GetLength(1); y++)
                {
                    Matrix[x, y] = 1.0 / (1 + Math.Exp(-Matrix[x, y]));
                }
            }
        }
        private void activateDer(double[,] Matrix, out double[,] C)
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

        private int matrixHash(double[][,] M)
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

        private double matrixError(double[,] A, double[,] B)
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

        private void matrixMult(ref double[,] A, double[,] B)
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
        private void matrixMult(double[,] A, double[,] B, out double[,] C)
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

        private void matrixSin(double[,] A, double[,] delta, out double[,] B)
        {
            B = new double[A.GetLength(0), A.GetLength(1)];
            for (int x = 0; x < A.GetLength(0); x++)
            {
                for (int y = 0; y < A.GetLength(1); y++)
                {
                    B[x, y] = A[x, y] / Math.Sqrt(1 + delta[x, y] * delta[x, y]);
                }
            }
        }
        private void matrixCos(double[,] A, double[,] delta, out double[,] B)
        {
            B = new double[A.GetLength(0), A.GetLength(1)];
            for (int x = 0; x < A.GetLength(0); x++)
            {
                for (int y = 0; y < A.GetLength(1); y++)
                {
                    B[x, y] = A[x, y] * delta[x, y] / Math.Sqrt(1 + delta[x, y] * delta[x, y]);
                }
            }
        }

        private void matrixTranspose(double[,] A, out double[,] B)
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

        private void scalarMult(double[,] A, double B)
        {
            for (int x = 0; x < A.GetLength(0); x++)
            {
                for (int y = 0; y < A.GetLength(1); y++)
                {
                    A[x, y] *= B;
                }
            }
        }
        private void scalarMult(double[,] A, double[,] B)
        {
            for (int x = 0; x < A.GetLength(0); x++)
            {
                for (int y = 0; y < A.GetLength(1); y++)
                {
                    A[x, y] *= B[x, y];
                }
            }
        }

        private void matrixAdd(double[,] A, double[,] B)
        {
            for (int x = 0; x < A.GetLength(0); x++)
            {
                for (int y = 0; y < A.GetLength(1); y++)
                {
                    A[x, y] += B[x, y];
                }
            }
        }

        private void matrixAddRow(double[,] A, double[,] B)
        {
            for (int x = 0; x < A.GetLength(0); x++)
            {
                for (int y = 0; y < A.GetLength(1); y++)
                {
                    A[x, y] += B[x, 0];
                }
            }
        }

        private void matrixSub(double[,] A, double[,] B)
        {
            for (int x = 0; x < A.GetLength(0); x++)
            {
                for (int y = 0; y < A.GetLength(1); y++)
                {
                    A[x, y] -= B[x, y];
                }
            }
        }

        private void matrixSet(ref double[,] A)
        {
            A = new double[A.GetLength(0), 1];
            for (int x = 0; x < A.GetLength(0); x++)
            {
                A[x, 0] = 1;
            }
        }

        private void accelCalc(double[,] Y, double[,] V, double[,] A)
        {
            for (int x = 0; x < Y.GetLength(0); x++)
            {
                for (int y = 0; y < Y.GetLength(1); y++)
                {
                    A[x, y] = -gravity * (Y[x, y] + Math.Sign(V[x, y]) * friction) / Math.Sqrt(1 + Y[x, y] * Y[x, y]);
                }
            }
        }
    }
}
