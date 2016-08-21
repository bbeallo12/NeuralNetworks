using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Complex_Neural_Network
{
    class Complex // complex number class
    {
        public double R { get; private set; }
        public double I { get; private set; }
        static private Random rand = new Random();

        public Complex(double Real, double Imaginary)
        {
            R = Real;
            I = Imaginary;
        }
        
        public static Complex one//return 1+0i 
        {
            get
            {
                return new Complex(1, 0);
            }
        } 
        public static Complex zero// return 0+0i 
        {
            get
            {
                return new Complex(0, 0);
            }
        } 
        public double arg // return angle
        {
            get
            {
                if (R == 0 && I == 0)
                {
                    return 0;
                }
                if (I >= 0)
                {
                    return Math.Acos(R / Math.Sqrt(R * R + I * I));
                }
                return Math.Acos(-R / Math.Sqrt(R * R + I * I)) + Math.PI;
            }
        }   
        public Complex conj // return conjugate
        {
            get
            {
                return new Complex(R, -I);
            }
        }
        public Complex oneWithAngle // return the unit vector of a complex number
        {
            get
            {
                return new Complex(R / Math.Sqrt(R * R + I * I), I / Math.Sqrt(R * R + I * I));
            }
        }
        public static Complex fromAngle(double theta) // return a complex number with magnitude of 1 and a given angle
        {
            return new Complex(Math.Cos(theta), Math.Sin(theta));
        }
        public double abs //return absolute value
        {
            get
            {
                return Math.Sqrt(R * R + I * I);
            }
        }
        public static Complex randC// return a random complex number
        {
            get
            {
                double randAng = rand.NextDouble() * 2 * Math.PI;
                return new Complex(Math.Cos(randAng), Math.Sin(randAng));
            }
        }
        // override operators
        public static Complex operator +(Complex A, Complex B)
        {
            return new Complex(A.R + B.R, A.I + B.I);
        }
        public static Complex operator -(Complex A, Complex B)
        {
            return new Complex(A.R - B.R, A.I - B.I);
        }
        public static Complex operator *(Complex A, Complex B)
        {
            return new Complex(A.R * B.R - A.I * B.I, A.R * B.I + A.I * B.R);
        }
        public static Complex operator *(double D, Complex C)
        {
            return new Complex(D * C.R, D * C.I);
        }
        public static Complex operator *(Complex C, double D)
        {
            return new Complex(D * C.R, D * C.I);
        }
        public static Complex operator /(Complex A, Complex B)
        {
            if (B.R * B.R + B.I * B.I == 0)
            {
                Console.WriteLine("******************* Divide by Zero *****************\n");
                return null;
            }
            return new Complex((A.R * B.R + A.I * B.I) / (B.R * B.R + B.I * B.I), (B.R * A.I - A.R * B.I) / (B.R * B.R + B.I * B.I));
        }
        public static Complex operator /(double D, Complex C)
        {
            if (C.R * C.R + C.I * C.I == 0)
            {
                Console.WriteLine("******************* Divide by Zero *****************\n");
                return null;
            }
            return new Complex(C.R * D / (C.R * C.R + C.I * C.I), -C.I * D / (C.R * C.R + C.I * C.I));
        }
        public static Complex operator /(Complex C, double D)
        {
            if (D == 0)
            {
                Console.WriteLine("******************* Divide by Zero *****************\n");
                return null;
            }
            return new Complex(C.R / D, C.I / D);
        }
        public override string ToString()
        {

            if (Math.Round(R, 7) == 0)
            {
                if (Math.Round(I, 7) == 0)
                {
                    return "0";
                }
                return string.Format("{0:f6}i", I);
            }
            else
            {
                if (Math.Round(I, 7) == 0)
                {
                    return string.Format("{0:f6}", R);
                }
                else if (Math.Round(I, 7) > 0)
                {
                    return string.Format("{0:f6}+{1:f6}i", R, I);
                }
                else
                {
                    return string.Format("{0:f6}{1:f6}i", R, I);
                }
            }
        }
        // parse complex inputs
        public static bool tryParse(string STR, out Complex COM)
        {
            int itA = 0, itB = 0, itC = 0, signs = 0;
            double[] numbs = new double[2];
            bool[] neg = new bool[2];

            while (itC < 2 && itA < STR.Length)
            {
                if (signs > 2)
                {
                    COM = null;
                    return false;
                }
                while (itB + 1 < STR.Length && ((STR[itB] >= '0' && STR[itB] <= '9') || STR[itB] == '.' || STR[itB] == 'E'))
                {
                    itB++;
                }
                if (STR[itB] == '+')
                {
                    if (itB - itA > 0)
                    {
                        if (!double.TryParse(STR.Substring(itA, itB - itA), out numbs[itC]))
                        {
                            COM = null;
                            return false;
                        }
                        numbs[itC] *= neg[itC] ? -1 : 1;
                        itC++;
                    }
                    signs++;
                    neg[itC] = false;
                    itB++;
                    itA = itB;
                }
                else if (STR[itB] == '-')
                {
                    if (itB - itA > 0)
                    {
                        if (!double.TryParse(STR.Substring(itA, itB - itA), out numbs[itC]))
                        {
                            COM = null;
                            return false;
                        }
                        numbs[itC] *= neg[itC] ? -1 : 1;
                        itC++;
                    }
                    signs++;
                    neg[itC] = true;
                    itB++;
                    itA = itB;
                }
                else if (STR[itB] == 'i')
                {
                    if (itB - itA > 0)
                    {
                        if (!double.TryParse(STR.Substring(itA, itB - itA), out numbs[itC]))
                        {
                            COM = null;
                            return false;
                        }
                        numbs[itC] *= neg[itC] ? -1 : 1;
                        itC++;
                    }
                    else
                    {
                        numbs[itC] = neg[itC] ? -1 : 1;
                        itC++;
                    }
                    itB++;
                    itA = itB;
                }
                else
                {
                    COM = null;
                    return false;
                }

            }

            COM = new Complex(numbs[0], numbs[1]);
            return true;
        }
        public static bool partComplexArray(string STR, Complex[,] matrix, int n)
        {
            int itA = 0, itB = 0, itC = 0;
            Complex temp;
            while (itA < STR.Length)
            {
                while (itB < STR.Length && STR[itB] != ' ')
                {
                    itB++;
                }
                if (!Complex.tryParse(STR.Substring(itA, itB - itA), out temp))
                {
                    return false;
                }
                matrix[n, itC++] = temp;
            }
            return true;
        }
    }
}
