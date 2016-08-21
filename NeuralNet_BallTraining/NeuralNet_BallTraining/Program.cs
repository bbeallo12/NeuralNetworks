using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.IO;

namespace NeuralNet_BallTraining
{
    class Program
    {
        static void Main(string[] args)
        {
            string appdata = Environment.GetFolderPath(Environment.SpecialFolder.MyDocuments) + "\\Neural Net Ball";
            string inputPath = appdata + "\\input.txt";
            string trainingInputPath = appdata + "\\trainingInput.txt";
            string targetOutputPath = appdata + "\\targetOutput.txt";
            string outputPath = appdata + "\\output.txt";
            string weightInputPath = appdata + "\\weightInput.txt";
            string weightOutputPath = appdata + "\\weightOutput.txt";
            string biasInputPath = appdata + "\\biasInput.txt";
            string biasOutputPath = appdata + "\\biasOutput.txt";
            string scalingOutputPath = appdata + "\\scalingOutput.txt";
            string scalingInputPath = appdata + "\\scalingInput.txt";
            if (!Directory.Exists(appdata) || !File.Exists(trainingInputPath) || !File.Exists(scalingOutputPath) || !File.Exists(scalingInputPath) || !File.Exists(targetOutputPath) || !File.Exists(inputPath) || !File.Exists(outputPath) || !File.Exists(weightInputPath) || !File.Exists(weightOutputPath) || !File.Exists(biasInputPath) || !File.Exists(biasOutputPath))
            {
                if (!Directory.Exists(appdata))
                {
                    Console.WriteLine("Created Neural Net Ball folder at: \n\t{0}", appdata);
                    Directory.CreateDirectory(appdata);
                }
                if (!File.Exists(trainingInputPath))
                {
                    Console.WriteLine("Created target.txt at: \n\t{0}", trainingInputPath);
                    File.Create(trainingInputPath);
                }
                if (!File.Exists(scalingOutputPath))
                {
                    Console.WriteLine("Created target.txt at: \n\t{0}", scalingOutputPath);
                    File.Create(scalingOutputPath);
                }
                if (!File.Exists(scalingInputPath))
                {
                    Console.WriteLine("Created target.txt at: \n\t{0}", scalingInputPath);
                    File.Create(scalingInputPath);
                }
                if (!File.Exists(targetOutputPath))
                {
                    Console.WriteLine("Created target.txt at: \n\t{0}", targetOutputPath);
                    File.Create(targetOutputPath);
                }
                if (!File.Exists(inputPath))
                {
                    Console.WriteLine("Created input.txt at \n\t{0}", inputPath);
                    File.Create(inputPath);
                }
                if (!File.Exists(outputPath))
                {
                    Console.WriteLine("Created output.txt at \n\t{0}", outputPath);
                    File.Create(outputPath);
                }
                if (!File.Exists(weightInputPath))
                {
                    Console.WriteLine("Created weightInput.txt at \n\t{0}", weightInputPath);
                    File.Create(weightInputPath);
                }
                if (!File.Exists(weightOutputPath))
                {
                    Console.WriteLine("Created weightOutput.txt at \n\t{0}", weightOutputPath);
                    File.Create(weightOutputPath);
                }
                if (!File.Exists(biasInputPath))
                {
                    Console.WriteLine("Created biasInput.txt at \n\t{0}", biasInputPath);
                    File.Create(biasInputPath);
                }
                if (!File.Exists(biasOutputPath))
                {
                    Console.WriteLine("Created biasOutput.txt at \n\t{0}", biasOutputPath);
                    File.Create(biasOutputPath);
                }
            }
            else
            {
                int[] topology;
                double[,] input;
                double[,] trainingInput;
                double[,] output;
                double[,] targetOutput;
                double[][,] weights = null;
                double[][,] bias = null;
                double[,] offsetAndScaling = new double[2, 2];
                double error;
                double Gravity;
                double Friction;
                double targetError;
                double randomness;
                int cycles;
                string userInput;
                bool loop;
                bool trainOrUse = false;
                double bestError = -1;

                Console.WriteLine("Would you like to load (L) previously determined weights and bias values to calculate an output\nor train (T) a completely new network? (L/T)");

                loop = true;
                while (loop)
                {
                    userInput = Console.ReadLine();
                    if (userInput.ToUpper() == "T")
                    {
                        trainOrUse = true;
                        loop = false;
                    }
                    else if (userInput.ToUpper() == "L")
                    {
                        trainOrUse = false;
                        loop = false;
                    }
                    else
                    {
                        Console.WriteLine("Unrecognized Input");
                    }
                }

                if (!trainOrUse)  // Using an already trained network
                {
                    Console.WriteLine("Input Neural Network Topology (EX: 2 3 1 ):");
                    do
                    {
                        loop = !partIntegers(Console.ReadLine(), out topology);
                        if (loop)
                        {
                            Console.WriteLine("Incorrect input");
                        }
                    } while (loop);

                    NeuralNet NN = new NeuralNet(topology);

                    Console.WriteLine("Neural Network has been generated");

                    Console.WriteLine("Inporting Weights from File location: {0}", weightInputPath);
                    importMatrixMultiple(weightInputPath, out weights);
                    if (NN.setWeights(weights))
                    {
                        Console.WriteLine("\tWeights Successfully Loaded");
                        Console.WriteLine("Inporting Bias from File location: {0}", biasInputPath);
                        importMatrixMultiple(biasInputPath, out bias);
                        if (NN.setBias(bias))
                        {
                            Console.WriteLine("\tBias Successfully Loaded");
                            Console.WriteLine("Inporting Offsets and Scalings from File location: {0}", scalingInputPath);
                            if (importMatrix(scalingInputPath, out offsetAndScaling) && offsetAndScaling.GetLength(0) == 2 && offsetAndScaling.GetLength(1) == 2)
                            {
                                Console.WriteLine("\tOffsets and Scalings Successfully Loaded");
                                Console.WriteLine("Inporting Inputs from File location: {0}", inputPath);
                                if (importMatrix(inputPath, out input))
                                {
                                    Console.WriteLine("\tInputs Successfully Loaded");
                                    normalizeData(input, offsetAndScaling[0, 0], offsetAndScaling[1, 0]);
                                    Console.WriteLine("Normalizing inputs");
                                    NN.feedForward(input, out output);
                                    Console.WriteLine("Outputs Calculated");
                                    deNormalizeData(output, offsetAndScaling[0, 1], offsetAndScaling[1, 1]);
                                    Console.WriteLine("Denormalizing outputs");
                                    exportMatrix(outputPath, output, false);
                                    Console.WriteLine("\tOutputs have been exported to\n\t{0}", outputPath);
                                }
                                else
                                {
                                    Console.WriteLine("******************** Inputs Failed to Load! ********************");
                                }
                            }
                            else
                            {
                                Console.WriteLine("******************** Offsets and Scalings Failed to Load! ********************");
                            }
                        }
                        else
                        {
                            Console.WriteLine("******************** Bias Failed to Load! ********************");
                        }
                    }
                    else
                    {
                        Console.WriteLine("******************** Weights Failed to Load! ********************");
                    }
                }
                else // Training a new Network
                {
                    Console.WriteLine("Input Neural Network Topology (EX: 2 3 1 ):");
                    do
                    {
                        loop = !partIntegers(Console.ReadLine(), out topology);
                        if (loop)
                        {
                            Console.WriteLine("Incorrect input");
                        }
                    } while (loop);

                    Console.WriteLine("Input Randomness Factor:");
                    do
                    {
                        loop = !double.TryParse(Console.ReadLine(), out randomness);
                        if (loop)
                        {
                            Console.WriteLine("Incorrect input");
                        }
                        if (randomness <= 0)
                        {
                            Console.WriteLine("Randomness must be positive");
                            loop = true;
                        }
                    } while (loop);

                    NeuralNet NN = new NeuralNet(topology, randomness);
                    cycles = 0;
                    Console.WriteLine("Neural Network has been generated");

                    Console.WriteLine("Inporting training Inputs from File location: {0}", trainingInputPath);
                    if (importMatrix(trainingInputPath, out trainingInput))
                    {
                        Console.WriteLine("\tTraining Inputs Successfully Loaded");
                        Console.WriteLine("Inporting target Outputs from File location: {0}", targetOutputPath);
                        if (importMatrix(targetOutputPath, out targetOutput))
                        {
                            Console.WriteLine("\tTarget Outputs Successfully Loaded");
                            Console.WriteLine("Normalizing training inputs and target outputs");
                            normalizeData(trainingInput, 2, out offsetAndScaling[0, 0], out offsetAndScaling[1, 0]);
                            normalizeData(targetOutput, 2, out offsetAndScaling[0, 1], out offsetAndScaling[1, 1]);
                            Console.WriteLine("Exporting Offset and Scaling information to\n\t{0}", scalingOutputPath);
                            exportMatrix(scalingOutputPath, offsetAndScaling, false);

                            bool temp;

                            Console.WriteLine("Input Gravity (Example 0.01)");
                            do
                            {
                                temp = double.TryParse(Console.ReadLine(), out Gravity);
                                if (!temp)
                                {
                                    Console.WriteLine("Invalid Input");
                                }
                                if (Gravity <= 0)
                                {
                                    temp = false;
                                    Console.WriteLine("Gravity must be positive");
                                }
                            } while (!temp);

                            Console.WriteLine("Input Friction (Example 0.2)");
                            do
                            {
                                temp = double.TryParse(Console.ReadLine(), out Friction);
                                if (!temp)
                                {
                                    Console.WriteLine("Invalid Input");
                                }
                                if (Friction <= 0)
                                {
                                    temp = false;
                                    Console.WriteLine("Friction must be positive");
                                }
                            } while (!temp);

                            Console.WriteLine("Input Target Error (Example 0.00001)");
                            do
                            {
                                temp = double.TryParse(Console.ReadLine(), out targetError);
                                if (!temp)
                                {
                                    Console.WriteLine("Invalid Input");
                                }
                                if (targetError <= 0)
                                {
                                    temp = false;
                                    Console.WriteLine("Target Error must be positive");
                                }
                            } while (!temp);

                            Console.WriteLine("Training...");
                            cycles = 0;
                            do
                            {
                                if (cycles == 99999)
                                {
                                //    Friction += 0.01;
                                }
                                NN.backPropogationBall(trainingInput, targetOutput, Gravity, Friction, out error);
                                if (bestError == -1)
                                {
                                    bestError = error;
                                    NN.getWeights(out weights);
                                    NN.getBias(out bias);
                                }
                                else if (bestError > error)
                                {
                                    bestError = error;
                                    NN.getWeights(out weights);
                                    NN.getBias(out bias);
                                }
                                Console.WriteLine("Current Error: {0,-10:0.00000000}\tBest Error: {1,-10:0.00000000}\tFriction: {2,-6:0.0000}", error, bestError, Friction);
                                cycles = (cycles + 1) % 100000;
                            } while (bestError > targetError);

                            exportMatrixMultiple(weightOutputPath, weights, false);
                            exportMatrixMultiple(biasOutputPath, bias, false);
                            Console.WriteLine("Trained weights and bias have been exported to \n\t{0}\n\t{1}\n", weightOutputPath, biasOutputPath);

                        }
                        else
                        {
                            Console.WriteLine("******************** Target Outputs Failed to Load! ********************");
                        }
                    }
                    else
                    {
                        Console.WriteLine("******************** Training Inputs Failed to Load! ********************");
                    }
                }
            }

        }

        static bool importMatrix(string fileLocation, out double[,] Matrix)
        {
            StreamReader reader = new StreamReader(fileLocation);
            int x, y, itA, itB, itC;
            double temp;
            string input;

            if (!int.TryParse(reader.ReadLine(), out x))
            {
                Matrix = null;
                reader.Close();
                return false;
            }
            if (!int.TryParse(reader.ReadLine(), out y))
            {
                Matrix = null;
                reader.Close();
                return false;
            }
            Matrix = new double[x, y];
            for (int i = 0; i < y; i++)
            {
                input = reader.ReadLine();
                itA = 0;
                itB = 0;
                itC = 0;
                while (itA < input.Length)
                {
                    if (itB < input.Length && ((input[itB] >= '0' && input[itB] <= '9') || input[itB] == '.' || input[itB] == '-'))
                    {
                        itB++;
                    }
                    else
                    {
                        if (itB - itA > 0)
                        {
                            if (double.TryParse(input.Substring(itA, itB - itA), out temp))
                            {
                                Matrix[itC++, i] = temp;
                            }
                            else
                            {
                                Matrix = null;
                                reader.Close();
                                return false;
                            }
                        }
                        itA = ++itB;
                    }
                }
            }
            reader.Close();
            return true;
        }
        static bool importMatrixMultiple(string fileLocation, out double[][,] Matrix)
        {
            StreamReader reader = new StreamReader(fileLocation);
            int x, y, matrixCount, itA, itB, itC;
            double temp;
            string input;

            if (!int.TryParse(reader.ReadLine(), out matrixCount))
            {
                Matrix = null;
                reader.Close();
                return false;
            }

            Matrix = new double[matrixCount][,];

            for (int z = 0; z < matrixCount; z++)
            {
                if (!int.TryParse(reader.ReadLine(), out x))
                {
                    Matrix = null;
                    reader.Close();
                    return false;
                }
                if (!int.TryParse(reader.ReadLine(), out y))
                {
                    Matrix = null;
                    reader.Close();
                    return false;
                }
                Matrix[z] = new double[x, y];
                for (int i = 0; i < y; i++)
                {
                    input = reader.ReadLine();
                    itA = 0;
                    itB = 0;
                    itC = 0;
                    while (itA < input.Length)
                    {
                        if (itB < input.Length && ((input[itB] >= '0' && input[itB] <= '9') || input[itB] == '.' || input[itB] == '-'))
                        {
                            itB++;
                        }
                        else
                        {
                            if (itB - itA > 0)
                            {
                                if (double.TryParse(input.Substring(itA, itB - itA), out temp))
                                {
                                    Matrix[z][itC++, i] = temp;
                                }
                                else
                                {
                                    Matrix = null;
                                    reader.Close();
                                    return false;
                                }
                            }
                            itA = ++itB;
                        }
                    }
                }
            }
            reader.Close();
            return true;
        }
        static void exportMatrix(string fileLocation, double[,] Matrix, bool Round)
        {
            File.WriteAllText(fileLocation, String.Empty);
            StreamWriter writer = new StreamWriter(fileLocation, false);
            writer.WriteLine(Matrix.GetLength(0));
            writer.WriteLine(Matrix.GetLength(1));
            for (int y = 0; y < Matrix.GetLength(1); y++)
            {
                for (int x = 0; x < Matrix.GetLength(0); x++)
                {
                    if (Round)
                    {
                        writer.Write("{0:0.00000} ", Matrix[x, y]);
                    }
                    else
                    {
                        writer.Write("{0} ", Matrix[x, y]);
                    }
                }
                writer.WriteLine();
            }
            writer.Close();
        }
        static void exportMatrixMultiple(string fileLocation, double[][,] Matrix, bool Round)
        {
            File.WriteAllText(fileLocation, String.Empty);
            StreamWriter writer = new StreamWriter(fileLocation, false);
            writer.WriteLine(Matrix.Length);
            for (int z = 0; z < Matrix.Length; z++)
            {
                writer.WriteLine(Matrix[z].GetLength(0));
                writer.WriteLine(Matrix[z].GetLength(1));
                for (int y = 0; y < Matrix[z].GetLength(1); y++)
                {
                    for (int x = 0; x < Matrix[z].GetLength(0); x++)
                    {
                        if (Round)
                        {
                            writer.Write("{0:0.00000} ", Matrix[z][x, y]);
                        }
                        else
                        {
                            writer.Write("{0} ", Matrix[z][x, y]);
                        }
                    }
                    writer.WriteLine();
                }
            }
            writer.Close();
        }
        static bool partIntegers(string input, out int[] Integers)
        {
            List<int> builder = new List<int>();
            int itA = 0, itB = 0, temp;
            while (itA < input.Length)
            {
                if (itB < input.Length)
                {
                    if (input[itB] >= '0' && input[itB] <= '9')
                    {
                        itB++;
                    }
                    else if (input[itB] == ' ')
                    {
                        if (itB - itA > 0)
                        {
                            if (int.TryParse(input.Substring(itA, itB - itA), out temp))
                            {
                                if (temp > 0)
                                {
                                    builder.Add(temp);
                                }
                                else
                                {
                                    Console.WriteLine("Integers must be positive");
                                    Integers = null;
                                    return false;
                                }
                            }
                            else
                            {
                                Console.WriteLine("Not an integer");
                                Integers = null;
                                return false;
                            }
                        }
                        itA = ++itB;
                    }
                    else
                    {
                        Console.WriteLine("Unrecognized Character");
                        Integers = null;
                        return false;
                    }
                }
                else
                {
                    if (itB - itA > 0)
                    {
                        if (int.TryParse(input.Substring(itA, itB - itA), out temp))
                        {
                            if (temp > 0)
                            {
                                builder.Add(temp);
                            }
                            else
                            {
                                Console.WriteLine("Integers must be positive");
                                Integers = null;
                                return false;
                            }
                        }
                        else
                        {
                            Console.WriteLine("Not an integer");
                            Integers = null;
                            return false;
                        }
                    }
                    itA = ++itB;
                }
            }
            Integers = builder.ToArray();
            return true;
        }
        static void normalizeData(double[,] Data, double pad, out double offset, out double scale)
        {
            double min = Data[0, 0], max = Data[0, 0];
            for (int x = 0; x < Data.GetLength(0); x++)
            {

                for (int y = 0; y < Data.GetLength(1); y++)
                {
                    if (min > Data[x, y])
                    {
                        min = Data[x, y];
                    }
                    if (max < Data[x, y])
                    {
                        max = Data[x, y];
                    }
                }
            }
            offset = min - pad;
            scale = (max + pad) - offset;
            for (int x = 0; x < Data.GetLength(0); x++)
            {

                for (int y = 0; y < Data.GetLength(1); y++)
                {
                    Data[x, y] = (Data[x, y] - offset) / scale;
                }
            }
        }
        static void normalizeData(double[,] Data, double offset, double scale)
        {
            for (int x = 0; x < Data.GetLength(0); x++)
            {

                for (int y = 0; y < Data.GetLength(1); y++)
                {
                    Data[x, y] = (Data[x, y] - offset) / scale;
                }
            }
        }
        static void deNormalizeData(double[,] Data, double offset, double scale)
        {
            for (int x = 0; x < Data.GetLength(0); x++)
            {

                for (int y = 0; y < Data.GetLength(1); y++)
                {
                    Data[x, y] = scale * Data[x, y] + offset;
                }
            }
        }
    }
}
