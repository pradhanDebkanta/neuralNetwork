#include <vector>
#include <iostream>
#include <cstdlib>
#include <cassert>
#include <cmath>
#include <fstream>
#include <sstream>

using namespace std;

// Silly class to read training data from a text file -- Replace This.
// Replace class TrainingData with whatever you need to get input data into the
// program, e.g., connect to a database, or take a stream of data from stdin, or
// from a file specified by a command line argument, etc.

class TrainingData
{
public:
    TrainingData(const string filename);                  // this costructor accept the file
    bool isEof(void) { return m_trainingDataFile.eof(); } // this function return true if all data are read from file else false
    void getTopology(vector<unsigned> &topology);         // access the topology or neural network

    // Returns the number of input values read from the file:
    unsigned getNextInputs(vector<double> &inputVals);
    unsigned getTargetOutputs(vector<double> &targetOutputVals);

private:
    ifstream m_trainingDataFile; // this is our data file to give training to our topology or network
};

/********** this bellow function is check whether there is any data to build our network, if it is ok then further procced for next operation ******/

void TrainingData::getTopology(vector<unsigned> &topology)
{
    string line;
    string label;

    getline(m_trainingDataFile, line); // this function take input from ecah line
    stringstream ss(line);             // this is for breaking the line to word
    ss >> label;                       // insert the each word to label

    // this bellow function check whether our input file has any topology or data to build nueral network
    // if not then emmideatly terminate the program with out feeding the data

    if (this->isEof() || label.compare("topology:") != 0)
    {
        abort();
    }

    // this bellow loop feed the data to the neural network topology to build network
    while (!ss.eof())
    {
        unsigned n;
        ss >> n;
        topology.push_back(n);
    }

    return;
}

/**************** this bellow function is for open the file**************/

TrainingData::TrainingData(const string filename)
{
    //is a builtin function in C++ which returns
    // a pointer to an array that contains a null-terminated sequence of characters representing the current value of the basic_string object.
    m_trainingDataFile.open(filename.c_str());     
}

/**************** this bellow function is use for take the input value from data file*******/
unsigned TrainingData::getNextInputs(vector<double> &inputVals)
{
    inputVals.clear(); // first clear the input value that previously load in inputVals

    string line;
    getline(m_trainingDataFile, line); // take the line from the file that opened
    stringstream ss(line);             // similerly break the line into word

    string label;
    ss >> label; // store the word into label
    if (label.compare("in:") == 0)
    {
        double oneValue;
        while (ss >> oneValue)
        {
            inputVals.push_back(oneValue);
        }
    }

    return inputVals.size();
}

/**************** this bellow function is use for take the output value from data file*******/
unsigned TrainingData::getTargetOutputs(vector<double> &targetOutputVals)
{
    targetOutputVals.clear();

    string line;
    getline(m_trainingDataFile, line);
    stringstream ss(line);

    string label;
    ss >> label;
    if (label.compare("out:") == 0)
    {
        double oneValue;
        while (ss >> oneValue)
        {
            targetOutputVals.push_back(oneValue);
        }
    }

    return targetOutputVals.size();
}

///////////////////////
////// here start our main neural network //////////////
//////////////////////////////////////////////

// this Connection structure is made for well connection of each neuron
// weight is use for forward propagation
// deltaweight is use for backword propagation for correction of weight value

struct Connection
{
    double weight;
    double deltaWeight;
};

class Neuron; // Declaration of class Neuron // we have to declare tis because we use typedef for rename data type

typedef vector<Neuron> Layer;

// ****************** class Neuron ******************
class Neuron
{
public:
    Neuron(unsigned numOutputs, unsigned myIndex);
    void setOutputVal(double val) { m_outputVal = val; }
    double getOutputVal(void) const { return m_outputVal; }
    void feedForward(const Layer &prevLayer);
    void calcOutputGradients(double targetVal);
    void calcHiddenGradients(const Layer &nextLayer);
    void updateInputWeights(Layer &prevLayer);

private:
    static double eta;   // [0.0..1.0] overall net training rate
    static double alpha; // [0.0..n] multiplier of last weight change (momentum)
    static double transferFunction(double x);
    static double transferFunctionDerivative(double x);
    static double randomWeight(void) { return rand() / double(RAND_MAX); }
    double sumDOW(const Layer &nextLayer) const;
    double m_outputVal;
    vector<Connection> m_outputWeights;
    unsigned m_myIndex;
    double m_gradient;
};

double Neuron::eta = 0.15;  // overall net learning rate, [0.0..1.0]
double Neuron::alpha = 0.5; // momentum, multiplier of last deltaWeight, [0.0..1.0]

void Neuron::updateInputWeights(Layer &prevLayer)
{
    // The weights to be updated are in the Connection container
    // in the neurons in the preceding layer

    for (unsigned n = 0; n < prevLayer.size(); ++n)
    {
        Neuron &neuron = prevLayer[n];
        double oldDeltaWeight = neuron.m_outputWeights[m_myIndex].deltaWeight;

        double newDeltaWeight =
            // Individual input, magnified by the gradient and train rate:
            eta * neuron.getOutputVal() * m_gradient
            // Also add momentum = a fraction of the previous delta weight;
            + alpha * oldDeltaWeight;

        neuron.m_outputWeights[m_myIndex].deltaWeight = newDeltaWeight;
        neuron.m_outputWeights[m_myIndex].weight += newDeltaWeight;
    }
}

double Neuron::sumDOW(const Layer &nextLayer) const
{
    double sum = 0.0;

    // Sum our contributions of the errors at the nodes we feed.

    for (unsigned n = 0; n < nextLayer.size() - 1; ++n)
    {
        sum += m_outputWeights[n].weight * nextLayer[n].m_gradient;
    }

    return sum;
}

void Neuron::calcHiddenGradients(const Layer &nextLayer)
{
    double dow = sumDOW(nextLayer);
    m_gradient = dow * Neuron::transferFunctionDerivative(m_outputVal);
}

/**** this function use to calculate gradient only*****/
void Neuron::calcOutputGradients(double targetVal)
{
    double delta = targetVal - m_outputVal;
    m_gradient = delta * Neuron::transferFunctionDerivative(m_outputVal);
}

/***** this function take the sum value from feed forward function & give the o/p tanh(sum)    **/
double Neuron::transferFunction(double x)
{
    // tanh - output range [-1.0..1.0]

    return tanh(x);
}

/** this function is use to derivate the tanh(m_outputVals)**/
double Neuron::transferFunctionDerivative(double x)
{
    // tanh derivative
    return 1.0 - x * x;
}

/******** this member function is for forward propagration of neuron *****/
void Neuron::feedForward(const Layer &prevLayer)
{
    double sum = 0.0;

    // Sum the previous layer's outputs (which are our inputs)
    // Include the bias node from the previous layer.

    for (unsigned n = 0; n < prevLayer.size(); ++n)
    {
        sum += prevLayer[n].getOutputVal() *
               prevLayer[n].m_outputWeights[m_myIndex].weight;
    }

    m_outputVal = Neuron::transferFunction(sum);
}

/**** this function is for build neuron and its property ****/
Neuron::Neuron(unsigned numOutputs, unsigned myIndex)
{
    for (unsigned c = 0; c < numOutputs; ++c)
    {
        m_outputWeights.push_back(Connection());
        m_outputWeights.back().weight = randomWeight();
    }

    m_myIndex = myIndex;
}

// ****************** class Net ******************
class Net
{
public:
    Net(const vector<unsigned> &topology);
    void feedForward(const vector<double> &inputVals);
    void backProp(const vector<double> &targetVals);
    void getResults(vector<double> &resultVals) const;
    double getRecentAverageError(void) const { return m_recentAverageError; }

private:
    vector<Layer> m_layers; // m_layers[layerNum][neuronNum]
    double m_error;
    double m_recentAverageError;
    static double m_recentAverageSmoothingFactor;
};

double Net::m_recentAverageSmoothingFactor = 100.0; // Number of training samples to average over

/*** this function indirectly store the feed forward transfer data into each neuron***/
void Net::getResults(vector<double> &resultVals) const
{
    resultVals.clear();

    for (unsigned n = 0; n < m_layers.back().size() - 1; ++n)
    {
        resultVals.push_back(m_layers.back()[n].getOutputVal());
    }
}

//////////////////////////////////
// THIS IS FOR BACKWORD PROPAGRATION / FEEDBACK PROPAGATION
/////////////////////////////////

void Net::backProp(const vector<double> &targetVals)
{
    // Calculate overall net error (RMS of output neuron errors)

    Layer &outputLayer = m_layers.back();
    m_error = 0.0; // initially it is 0 because first we feed the new data on it

    for (unsigned n = 0; n < outputLayer.size() - 1; ++n)
    {
        double delta = targetVals[n] - outputLayer[n].getOutputVal();
        m_error += delta * delta;
    }
    m_error /= outputLayer.size() - 1; // get average error squared
    m_error = sqrt(m_error);           // RMS

    // Implement a recent average measurement

    m_recentAverageError =
        (m_recentAverageError * m_recentAverageSmoothingFactor + m_error) / (m_recentAverageSmoothingFactor + 1.0);

    // Calculate output layer gradients

    for (unsigned n = 0; n < outputLayer.size() - 1; ++n)
    {
        outputLayer[n].calcOutputGradients(targetVals[n]);
    }

    // Calculate hidden layer gradients

    for (unsigned layerNum = m_layers.size() - 2; layerNum > 0; --layerNum)
    {
        Layer &hiddenLayer = m_layers[layerNum];
        Layer &nextLayer = m_layers[layerNum + 1];

        for (unsigned n = 0; n < hiddenLayer.size(); ++n)
        {
            hiddenLayer[n].calcHiddenGradients(nextLayer);
        }
    }

    // For all layers from outputs to first hidden layer,
    // update connection weights

    for (unsigned layerNum = m_layers.size() - 1; layerNum > 0; --layerNum)
    {
        Layer &layer = m_layers[layerNum];
        Layer &prevLayer = m_layers[layerNum - 1];

        for (unsigned n = 0; n < layer.size() - 1; ++n)
        {
            layer[n].updateInputWeights(prevLayer);
        }
    }
}

//////////////////////////////////
// THIS IS FOR FORWARD PROPAGRATION / FEED PROPAGATION
/////////////////////////////////

void Net::feedForward(const vector<double> &inputVals)
{
    // this bellow line is used to check validaty of number if input neuron and number of input
    assert(inputVals.size() == m_layers[0].size() - 1); // here m_layers[0].size()-1 is used because we cann't provide the input to the bias neuron

    // Assign (latch) the input values into the input neurons
    for (unsigned i = 0; i < inputVals.size(); ++i)
    {
        m_layers[0][i].setOutputVal(inputVals[i]); // this line use for initilise the input neuron at layer[0] by using member function
    }

    // forward propagate
    for (unsigned layerNum = 1; layerNum < m_layers.size(); ++layerNum)
    {
        Layer &prevLayer = m_layers[layerNum - 1]; // this line define the previous layer
        for (unsigned n = 0; n < m_layers[layerNum].size() - 1; ++n)
        {
            m_layers[layerNum][n].feedForward(prevLayer);
        }
    }
}

/**************this constructor build the overall neuron network ****************/
Net::Net(const vector<unsigned> &topology)
{
    unsigned numLayers = topology.size(); // the work of this variable is to store the number of layer of network
    // now we make each layer & as well as also neuron of each layer

    for (unsigned layerNum = 0; layerNum < numLayers; ++layerNum)
    {
        m_layers.push_back(Layer()); //this line actually make a layer of network by using Layer() constructor

        // this bellow line is used for determine number of the next hidden layer
        // if the layer is final layer then next output layer is 0 ie. it is the output layer
        unsigned numOutputs = layerNum == topology.size() - 1 ? 0 : topology[layerNum + 1];

        // We have a new layer, now fill it with neurons, and
        // add a bias neuron in each layer.
        for (unsigned neuronNum = 0; neuronNum <= topology[layerNum]; ++neuronNum)
        {
            m_layers.back().push_back(Neuron(numOutputs, neuronNum)); // this line actually make the neurons of a specific layer by using neuron() constructor
            cout << "Made a Neuron! \t";
        }
        cout<< endl;

        // Force the bias node's output to 1.0 (it was the last neuron pushed in this layer):
        m_layers.back().back().setOutputVal(1.0);
    }
}

/********this member function just show the result *******/
void showVectorVals(string label, vector<double> &v)
{
    cout << label << " ";
    for (unsigned i = 0; i < v.size(); ++i)
    {
        cout << v[i] << " ";
    }

    cout << endl;
}

/***************** main function start from here ******************/
int main()
{
    TrainingData trainData("temp/inputFile.txt");

    // here if we initilise topology as {3,2,1}
    //then the network has 3 layer and 1st layer has 3 neurons, 2nd layer has 2 neurons and 3rd layer has 1 neuron
    // e.g., { 3, 2, 1 }
    vector<unsigned> topology;
    trainData.getTopology(topology);

    Net myNet(topology);

    vector<double> inputVals, targetVals, resultVals;
    int trainingPass = 0;

    // this while loop is execute untill the training data is ending
    while (!trainData.isEof())
    {
        ++trainingPass;
        cout << endl
             << "Pass " << trainingPass;

        // Get new input data and feed it forward:
        if (trainData.getNextInputs(inputVals) != topology[0]) // the condition is actually check the inputVals vector size with topology[0]
        {
            break;
        }
        showVectorVals(": Inputs:", inputVals);
        myNet.feedForward(inputVals);

        // Collect the net's actual output results:
        myNet.getResults(resultVals);
        showVectorVals("Outputs:", resultVals);

        // Train the net what the outputs should have been:
        trainData.getTargetOutputs(targetVals);
        showVectorVals("Targets:", targetVals);
        assert(targetVals.size() == topology.back());

        myNet.backProp(targetVals);

        // Report how well the training is working, average over recent samples:
        cout << "Net recent average error: "
             << myNet.getRecentAverageError() << endl;
    }

    cout << endl
         << "Done" << endl;
}
