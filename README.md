# RNN Name Classification

A simple recurrent neural network (RNN) implemented in PyTorch to classify names by language.

## Repository Structure

- **rnn.ipynb**: Contains all the code for data processing, model definition, training, and evaluation.
- **data/**
  - **eng-fra.txt**: Contains English-French data.
  - **names/**: Directory with text files such as `Arabic.txt`, `Chinese.txt`, `Czech.txt`, etc., each containing names of that language.

## Requirements

- Python 3.10+
- [PyTorch](https://pytorch.org/)
- Additional packages: `unicodedata`, `string`, `glob`, `os`, `matplotlib`

## Usage

1. **Clone the Repository**

   ```sh
   git clone https://github.com/DakshGoyal1506/RNN_Implemented.git
   cd RNN_Implemented
   ```

2. **Install Dependencies**

   Set up a virtual environment (optional but recommended) and install PyTorch and matplotlib:

   ```sh
   python -m venv venv
   # On Linux/Mac:
   source venv/bin/activate
   # On Windows:
   venv\Scripts\activate
   pip install torch matplotlib
   ```

3. **Run the Notebook**

   Open the `rnn.ipynb` file in [Visual Studio Code](https://code.visualstudio.com/) and run the cells sequentially.

## Project Overview

- **Data Loading:** The `NameDataset` class reads name data from the `data/names` directory, associating each file with a language label.
- **Preprocessing:** Utility functions convert Unicode strings to ASCII and then transform names into one-hot encoded tensors.
- **Model:** A simple RNN is defined in the `SimpleRNN` class, consisting of linear layers for input-to-hidden, hidden-to-hidden, and hidden-to-output transformations.
- **Training & Evaluation:** The notebook provides functions to train the model (`train`) and evaluate it (`test`) along with training visualization over epochs.

## Results

Training logs and loss/accuracy plots are generated during training (e.g., using 50 epochs) to monitor model performance on the training and test sets.

## License

This project is licensed under the MIT License.

## Acknowledgments

This implementation is inspired by tutorials from [PyTorch](https://pytorch.org/tutorials/intermediate/char_rnn_classification_tutorial.html).
