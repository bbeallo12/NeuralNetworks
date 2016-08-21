% XOR training set
% Complex Valued Neural Network
% topography is
%   1 Input Layers
%       2 Inputs neurons
%   1 Hidden Layers
%       2 Hidden neurons
%   1 Output Layer
%       1 output neuron
% activation function used is:
%   act(N) =  N/(1+|N|)
clear variables
clc

RUNS = 10000; % training cycles
I1 = -0.9;  % initial inputs
I2 = -0.9;
T = -0.99;   % initial target
LR = 0.5; % learning rate

% set random weights
W1 = rand+rand*i;
V1 = rand+rand*i;
W2 = rand+rand*i;
V2 = rand+rand*i;
W3 = rand+rand*i;
V3 = rand+rand*i;
W4 = rand+rand*i;
V4 = rand+rand*i;
W5 = rand+rand*i;
V5 = rand+rand*i;
W6 = rand+rand*i;
V6 = rand+rand*i;
% set random bias
B1 = rand+rand*i;
B2 = rand+rand*i;
B3 = rand+rand*i;

for n = 0:RUNS
% feed forward
%   Hidden Layer
%       For a complex neural network
%       I modified the Net equation
%       Instead of
%           N = W*I+B
%       I changed it to
%           N = W*I + V*conj(I) + B
%       where W and V are two independent complex weights
%       And conj(I) is the conjugate of I
%       This modification helps the network learn reflections
    N1 = W1*I1+V1*conj(I1) + W2*I2+V2*conj(I2) + B1;
    N2 = W3*I1+V1*conj(I1) + W4*I2+V2*conj(I2) + B2;
    H1 = N1/(1+abs(N1));
    H2 = N2/(1+abs(N2));
%   Output Layer
    N3 = W5*H1+V5*conj(H1) + W6*H2+V6*conj(H2) + B3;
    O = N3/(1+abs(N3));

% start of back prop
%   The name dEdO means the derivative dE/dO and so on
%   So the general format is dYdX really means dY/dX
    dEdO = [real(O)-real(T) , imag(O)-imag(T)];
    dOdN3 = [(abs(N3) + imag(N3)^2)/(abs(N3)*(abs(N3)+1)^2),-real(N3)*imag(N3)/(abs(N3)*(abs(N3)+1)^2);-real(N3)*imag(N3)/(abs(N3)*(abs(N3)+1)^2),(abs(N3) + real(N3)^2)/(abs(N3)*(abs(N3)+1)^2)];

% complex derivatives
% dY/dX = [  dY_real/dX_real   , dY_real/dX_imaginary  ;  dY_imaginary/dX_real ,  dY_imaginary/dX_imaginary ]
    dN3dB3 = [1 0 ; 0 1];
    dN3dW5 = [real(H1) -imag(H1) ;  imag(H1) real(H1)];
    dN3dV5 = [real(H1)  imag(H1) ; -imag(H1) real(H1)];
    dN3dW6 = [real(H2) -imag(H2) ;  imag(H2) real(H2)];
    dN3dV6 = [real(H2)  imag(H2) ; -imag(H2) real(H2)];
    dN3dH1 = [real(W5)+real(V5),-imag(W5)+imag(V5);imag(W5)+imag(V5),real(W5)-real(V5)];
    dN3dH2 = [real(W6)+real(V6),-imag(W6)+imag(V6);imag(W6)+imag(V6),real(W6)-real(V6)];

    dH1dN1 = [(abs(N1) + imag(N1)^2)/(abs(N1)*(abs(N1)+1)^2),-real(N1)*imag(N1)/(abs(N1)*(abs(N1)+1)^2);-real(N1)*imag(N1)/(abs(N1)*(abs(N1)+1)^2),(abs(N1) + real(N1)^2)/(abs(N1)*(abs(N1)+1)^2)];
    dH2dN2 = [(abs(N2) + imag(N2)^2)/(abs(N2)*(abs(N2)+1)^2),-real(N2)*imag(N2)/(abs(N2)*(abs(N2)+1)^2);-real(N2)*imag(N2)/(abs(N2)*(abs(N2)+1)^2),(abs(N2) + real(N2)^2)/(abs(N2)*(abs(N2)+1)^2)];

    dN2dB2 = [1 0 ; 0 1];
    dN2dW3 = [real(I1) -imag(I1) ;  imag(I1) real(I1)];
    dN2dV3 = [real(I1)  imag(I1) ; -imag(I1) real(I1)];
    dN2dW4 = [real(I2) -imag(I2) ;  imag(I2) real(I2)];
    dN2dV4 = [real(I2)  imag(I2) ; -imag(I2) real(I2)];

    dN1dB1 = [1 0 ; 0 1];
    dN1dW1 = [real(I1) -imag(I1) ;  imag(I1) real(I1)];
    dN1dV1 = [real(I1)  imag(I1) ; -imag(I1) real(I1)];
    dN1dW2 = [real(I2) -imag(I2) ;  imag(I2) real(I2)];
    dN1dV2 = [real(I2)  imag(I2) ; -imag(I2) real(I2)];

% chain rules
    dEdB1 = dEdO*dOdN3*dN3dH1*dH1dN1*dN1dB1;
    dEdB2 = dEdO*dOdN3*dN3dH2*dH2dN2*dN2dB2;
    dEdB3 = dEdO*dOdN3*dN3dB3;

    dEdW1 = dEdO*dOdN3*dN3dH1*dH1dN1*dN1dW1;
    dEdW2 = dEdO*dOdN3*dN3dH1*dH1dN1*dN1dW2;
    dEdW3 = dEdO*dOdN3*dN3dH2*dH2dN2*dN2dW3;
    dEdW4 = dEdO*dOdN3*dN3dH2*dH2dN2*dN2dW4;
    dEdW5 = dEdO*dOdN3*dN3dW5;
    dEdW6 = dEdO*dOdN3*dN3dW6;

    dEdV1 =	dEdO*dOdN3*dN3dH1*dH1dN1*dN1dV1;
    dEdV2 =	dEdO*dOdN3*dN3dH1*dH1dN1*dN1dV2;
    dEdV3 =	dEdO*dOdN3*dN3dH2*dH2dN2*dN2dV3;
    dEdV4 =	dEdO*dOdN3*dN3dH2*dH2dN2*dN2dV4;
    dEdV5 = dEdO*dOdN3*dN3dV5;
    dEdV6 = dEdO*dOdN3*dN3dV6;

% add change to weights and bias
    B1 = B1 - LR*(dEdB1(1) + dEdB1(2)*i);
    B2 = B2 - LR*(dEdB2(1) + dEdB2(2)*i);
    B3 = B3 - LR*(dEdB3(1) + dEdB3(2)*i);

    W1 = W1 - LR*(dEdW1(1) + dEdW1(2)*i);
    W2 = W2 - LR*(dEdW2(1) + dEdW2(2)*i);
    W3 = W3 - LR*(dEdW3(1) + dEdW3(2)*i);
    W4 = W4 - LR*(dEdW4(1) + dEdW4(2)*i);
    W5 = W5 - LR*(dEdW5(1) + dEdW5(2)*i);
    W6 = W6 - LR*(dEdW6(1) + dEdW6(2)*i);

    V1 = V1 - LR*(dEdV1(1) + dEdV1(2)*i);
    V2 = V2 - LR*(dEdV2(1) + dEdV2(2)*i);
    V3 = V3 - LR*(dEdV3(1) + dEdV3(2)*i);
    V4 = V4 - LR*(dEdV4(1) + dEdV4(2)*i);
    V5 = V5 - LR*(dEdV5(1) + dEdV5(2)*i);
    V6 = V6 - LR*(dEdV6(1) + dEdV6(2)*i);

% calculate error
    Error = (T-O)*conj(T-O)/2;

% output first four iterations
    if (n==0)
        'Initial Values'
        Output = '  |    Input 1      |     Input 2     |     Output       |    Target        |     Error       |'
    end
    if (n<=4)
        Output = [I1, I2, O, T, Error]
    end
    
% output last four iterations
    if (n==RUNS-4)
        'Trained Values'
        Output = '  |    Input 1      |     Input 2     |     Output       |    Target        |     Error       |'
    end
    if (n>RUNS-4)
        Output = [I1, I2, O, T, Error]
    end

% switch Intputs and Outputs to next XOR values
    if(I1 < 0)
        T = -T;
    end

    I1=-I1;

    if(mod(n,2)==1)
        I2=-I2;
    end

end