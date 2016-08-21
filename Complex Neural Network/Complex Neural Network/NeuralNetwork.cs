using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Complex_Neural_Network
{
    class NeuralNetwork
    {
        // 1 - Define delegate
        // 2 - Define an event based on that delegate
        // 3 - Raise the event

        //  Declare events
        public delegate void BackPropStartEventHandler(object source, EventArgs e);
        public event BackPropStartEventHandler BackPropStart;
        protected virtual void OnBackPropStart()
        {
            if (BackPropStart != null)
                BackPropStart(this, EventArgs.Empty);

        }
        public delegate void ForPropStartEventHandler(object source, EventArgs e);
        public event ForPropStartEventHandler ForPropStart;
        protected virtual void OnForPropStart()
        {
            if (ForPropStart != null)
                ForPropStart(this, EventArgs.Empty);

        }

        const double stdDev = 3.462;    // constant for controlling standard deviation of weights and bias
        Random rand = new Random();     // random generator
        int inputCount, outputCount, layerCount, topologyHash, biasHash;
        int[] topology;

        public Neuron[][] network { get; set; } // Declare network array of neurons

        public NeuralNetwork(int[] Topology)    // constructor
        {
            // set local variables
            topology = Topology;    // save topology
            layerCount = topology.Length;
            inputCount = topology[0];
            outputCount = topology[layerCount - 1];

            // initialize first dimension of network array
            network = new Neuron[layerCount][];

            //  Initialize neurons in network array
            for (int i = 0; i < layerCount; i++)
            {
                network[i] = new Neuron[topology[i]];
                for (int j = 0; j < topology[i]; j++)
                {
                    if (i == 0)
                    {
                        network[i][j] = new Neuron(j, topology[i], 0, topology[i + 1], 0);
                    }
                    else if (i == layerCount - 1)
                    {
                        network[i][j] = new Neuron(j, topology[i], topology[i - 1], 0, 2);
                    }
                    else
                    {
                        network[i][j] = new Neuron(j, topology[i], topology[i - 1], topology[i + 1], 1);
                    }
                    BackPropStart += network[i][j].OnBackPropStart;
                    ForPropStart += network[i][j].OnForPropStart;
                }
            }
            for (int i = 0; i < layerCount; i++)
            {
                for (int j = 0; j < topology[i]; j++)
                {
                    if (i == 0)
                    {
                        network[i][j].forwardLayer = network[i + 1];
                    }
                    else if (i == layerCount - 1)
                    {
                        network[i][j].backLayer = network[i - 1];
                    }
                    else
                    {
                        network[i][j].forwardLayer = network[i + 1];
                        network[i][j].backLayer = network[i - 1];
                    }
                }
            }

        }

        // Feed forward
        public bool forwardProp(Complex[,] Input, out Complex[,] Output, int n)
        {
            // Check if inputs match the number of input neurons
            if (Input.GetLength(0) != inputCount)
            {
                Console.WriteLine("Wrong input size");
                Output = null;
                return false;
            }
            Output = new Complex[outputCount, Input.GetLength(1)];
            if (n == -1)
            {
                // activate sendForward() function of each neuron
                for (int k = 0; k < Input.GetLength(1); k++)
                {
                    setInputs(Input, k);
                    // Call OnForPropStart event
                    OnForPropStart();
                    for (int i = 0; i < layerCount; i++)
                    {
                        for (int j = 0; j < topology[i]; j++)
                        {
                            if (i == 0)
                            {
                                network[i][j].sendForward();
                            }
                            else if (i == layerCount - 1)
                            {
                                network[i][j].act();
                            }
                            else
                            {
                                network[i][j].act();
                                network[i][j].sendForward();
                            }
                        }
                    }

                    getOutputs(Output, k);
                }
            }
            else
            {

                setInputs(Input, n);
                OnForPropStart();
                for (int i = 0; i < layerCount; i++)
                {
                    for (int j = 0; j < topology[i]; j++)
                    {
                        if (i == 0)
                        {
                            network[i][j].sendForward();
                        }
                        else if (i == layerCount - 1)
                        {
                            network[i][j].act();
                        }
                        else
                        {
                            network[i][j].act();
                            network[i][j].sendForward();
                        }
                    }
                }

                getOutputs(Output, n);

            }

            return true;
        }

        // BackPropagation training
        public bool backProp(Complex[,] Input, Complex[,] Output, double Error, double LR)
        {
            // Make sure inputs and outputs match the network size
            if (Input.GetLength(0) != inputCount || Output.GetLength(0) != outputCount)
            {
                Console.WriteLine("Wrong input or output count");
                return false;
            }
            if (Input.GetLength(1) != Output.GetLength(1))
            {
                Console.WriteLine("Input Output size mismatch");
                return false;
            }
            double currentError = -1;
            Complex[,] testOutput;
            // call OnBackPropStart event
            OnBackPropStart();
            do
            {
                currentError = 0;
                // loop through examples
                for (int k = 0; k < Input.GetLength(1); k++)
                {
                    // calculate output
                    forwardProp(Input, out testOutput, k);
                    // loop through network layers
                    for (int i = layerCount - 1; i > 0; i--)
                    {
                        // loop though neurons in layer
                        for (int j = 0; j < network[i].Length; j++)
                        {
                            if (i == layerCount - 1)
                            {
                                // call sendBackward function of neuron
                                // add returned error to current error
                                currentError += network[i][j].sendBackward(LR, Output, k);
                            }
                            else if (i == 1)
                            {
                                network[i][j].sendBackward(LR);
                            }
                        }
                    }

                }
                // divide error by number of outputs
                currentError /= Output.GetLength(1);
                Console.WriteLine("Current Error: {0}", currentError);
            } while (currentError > Error);

            return true;
        }

        // input and output functions
        private void setInputs(Complex[,] Input, int n)
        {
            for (int j = 0; j < inputCount; j++)
            {
                network[0][j].value = Input[j, n];
            }
        }
        private void getOutputs(Complex[,] Output, int n)
        {
            for (int j = 0; j < outputCount; j++)
            {
                Output[j, n] = network[layerCount - 1][j].value;
            }
        }

    }
}
