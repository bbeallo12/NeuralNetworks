using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Complex_Neural_Network
{
    class Neuron
    {
        // declare local variables
        static Random rand = new Random();
        const double stdDev = 3.462;
        public Neuron[] backLayer { get; set; }
        public Neuron[] forwardLayer { get; set; }
        int ID;
        int layerSize;
        int type;
        public Complex net { get; set; }
        public Complex value { get; set; }
        public Complex[,] weights { get; set; }
        public Complex bias { get; set; }
        Complex[,] dW;
        Complex dB;
        public Complex backPropError { get; set; }

        // constructor
        public Neuron(int id, int LayerSize, int preLayerSize, int postLayerSize, int Type)
        {
            // set local variables
            type = Type;
            backPropError = Complex.zero;
            net = Complex.zero;
            value = Complex.zero;
            ID = id;
            layerSize = LayerSize;
            if (type > 0) // if it is a hidden or output neuron
            {
                weights = new Complex[2, preLayerSize];
                dW = new Complex[2, preLayerSize];
                bias = stdDev * (rand.NextDouble() - 0.5) * Complex.randC / Math.Sqrt(preLayerSize);
                dB = Complex.zero;
                for (int i = 0; i < preLayerSize; i++)
                {
                    weights[0, i] = stdDev * (rand.NextDouble() - 0.5) * Complex.randC / Math.Sqrt(preLayerSize);
                    dW[0, i] = Complex.zero;
                    weights[1, i] = stdDev * (rand.NextDouble() - 0.5) * Complex.randC / Math.Sqrt(preLayerSize);
                    dW[1, i] = Complex.zero;
                }

            }
        }

        public void addNet(int ID, Complex Value) // add to net value
        {
            net += Value * weights[0, ID] + Value.conj * weights[1, ID];
        }
        public void act() // activation function
        {
            net += bias;
            value = net / (1 + net.abs);
        }
        public double derAct()  // activation function derivative
        {
            return 1 / ((1 + net.abs) * (1 + net.abs));
        }
        public void sendForward() // feed forward operation for each neuron
        {
            for (int i = 0; i < forwardLayer.Length; i++)
            {
                forwardLayer[i].addNet(ID, value);
            }
        }
        public double sendBackward(double LR, Complex[,] Target, int n)// back propagation operation for each neuron
        {
            // Calculate backprop Error
            backPropError = (value - Target[ID, n]).abs * Complex.fromAngle((value - Target[ID, n]).arg);
            /** derAct()*/
            // Calculate dW
            dB = 1 * backPropError;

            for (int i = 0; i < dW.GetLength(1); i++)
            {
                dW[0, i] = backPropError.abs * backLayer[i].value.abs * Complex.fromAngle(backPropError.arg - backLayer[i].value.arg);
                dW[1, i] = backPropError.abs * backLayer[i].value.abs * Complex.fromAngle(backPropError.arg + backLayer[i].value.arg);
            }
            // Pass back prop error
            for (int i = 0; i < backLayer.Length; i++)
            {
                //backLayer[i].backPropError += backPropError.abs * weights[i].abs * Complex.fromAngle(backPropError.arg - weights[i].arg);
                backLayer[i].backPropError += new Complex(backPropError.abs * (weights[0, i] + weights[1, i]).abs * Math.Cos(backPropError.arg - (weights[0, i] + weights[1, i]).arg), backPropError.abs * (weights[0, i] - weights[1, i]).abs * Math.Sin(backPropError.arg - (weights[0, i] - weights[1, i]).arg));
            }

            // Change Weights
            bias -= LR * dB;
            for (int i = 0; i < weights.GetLength(1); i++)
            {
                weights[0, i] -= LR * dW[0, i];
                weights[1, i] -= LR * dW[1, i];
            }
            
            Complex error = (-1 * Target[ID, n] * net.abs + net - Math.Log(net.abs + 1) * Complex.fromAngle(net.arg));
            Complex error2 = (-1 * Target[ID, n] * (Target[ID, n] / (1 - Target[ID, n].abs)).abs + (Target[ID, n] / (1 - Target[ID, n].abs)) - Math.Log((Target[ID, n] / (1 - Target[ID, n].abs)).abs + 1) * Complex.fromAngle((Target[ID, n] / (1 - Target[ID, n].abs)).arg));

            return (error - error2).abs;
            //return (value - Target[ID, n]).abs * (value - Target[ID, n]).abs / 2;
        }
        public void sendBackward(double LR)
        {

            // Calculate backprop Error
            backPropError *= derAct();

            // Calculate dW
            dB = backPropError;

            for (int i = 0; i < dW.GetLength(1); i++)
            {
                dW[0, i] = new Complex(backPropError.abs * backLayer[i].value.abs * Math.Cos(backPropError.arg - backLayer[i].value.arg), backPropError.abs * backLayer[i].value.abs * Math.Sin(backPropError.arg - backLayer[i].value.arg));
                dW[1, i] = new Complex(backPropError.abs * backLayer[i].value.abs * Math.Cos(backPropError.arg + backLayer[i].value.arg), backPropError.abs * backLayer[i].value.abs * Math.Sin(backPropError.arg + backLayer[i].value.arg));

            }
            // Pass back prop error
            for (int i = 0; i < backLayer.Length; i++)
            {
                backLayer[i].backPropError += new Complex(backPropError.abs * (weights[0, i] + weights[1, i]).abs * Math.Cos(backPropError.arg - (weights[0, i] + weights[1, i]).arg), backPropError.abs * (weights[0, i] - weights[1, i]).abs * Math.Sin(backPropError.arg - (weights[0, i] - weights[1, i]).arg));
            }

            // Change Weights
            bias -= LR * dB;
            for (int i = 0; i < weights.GetLength(1); i++)
            {
                weights[0, i] -= LR * dW[0, i];
                weights[1, i] -= LR * dW[1, i];
            }

        }
        public void OnBackPropStart(object source, EventArgs e)
        {
            backPropError = Complex.zero;
        }
        public void OnForPropStart(object source, EventArgs e)
        {
            net = Complex.zero;
        }
    }
}
