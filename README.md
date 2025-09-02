# Character-Level Language Model

This notebook demonstrates the training of a simple character-level language model using PyTorch. The model learns to predict the next character in a sequence based on the preceding characters, trained on a dataset of names.

## Data Loading and Preprocessing

The dataset of names is loaded, and mappings between characters and integers are created. A special character '.' is used for the start and end of names. The dataset is then built by creating input-output pairs with a `block_size` of 3, meaning the model considers the previous 3 characters to predict the next one. The dataset is split into training, development, and test sets.

## Model Architecture

The model consists of:
- **Embedding Layer (C):** Maps characters to low-dimensional dense vectors (size 2).
- **Hidden Layer:** A linear layer with tanh activation, transforming concatenated embeddings into a hidden representation (size 300).
- **Output Layer:** Another linear layer producing logits for each possible next character (size 27, including '.').

The total number of parameters in the model is 10281.

## Training Loop

The model is trained using mini-batch gradient descent. Each iteration involves:
1. Constructing a mini-batch.
2. Performing a forward pass to compute logits.
3. Calculating the cross-entropy loss.
4. Performing a backward pass to compute gradients.
5. Updating parameters using a learning rate.
6. Tracking stats like learning rate and loss.

## Evaluation

The model's performance is evaluated by calculating the loss on the training and development datasets.

## Visualization

Visualizations include plotting the loss over training steps to observe learning progress and visualizing character embeddings to see spatial relationships between characters.

## Conclusion

This notebook provides a step-by-step guide to building and training a character-level language model, covering data preprocessing, model definition, training, and evaluation.
