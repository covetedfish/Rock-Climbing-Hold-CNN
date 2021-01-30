# Indoor Rock Climbing Hold Classifier
A convolutional neural network for the classification of indoor rock climbing holds. 

## Usage

### General
python3 run_nn_tf.py 

### Scraping the data

python3 scrape_holds.py 

### Processing the Images

python3 hold_identifier.py absolute/filepath/to/images

## Structure 

The CNN classifies images of holds into 6 categories: edges, jugs, pinches, pockets, slopers, crimps. The best accuracy  I was able to achieve on the validation set was ~35% with the followin hyperparameters:
 
 * 8x8 pooling with stride 2
 * Two convolutional layers with ReLU activation function 
     * 32 filters 5x5
     * 16 filters 3x3
 * One final dense layer with softmax activation function

 The plot of training vs. validation accuracy for this model is entitled 8x8 pooling.

 
