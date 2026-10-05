SentimentScope: Custom Transformer for Sentiment Analysis
Executive Summary
SentimentScope is a custom PyTorch-based natural language processing (NLP) classifier designed to analyze user sentiment in movie reviews. Developed for CineScope's recommendation system, this project leverages a from-scratch Transformer architecture to classify IMDB movie reviews as either positive or negative, enabling more accurate and personalized content suggestions.

📊 Results & Impact
Final Test Accuracy: 76.65% (evaluated on 25,000 unseen reviews)

Validation Accuracy: 80.72% (achieved during 5-epoch training cycle)

The model successfully exceeded the 75% baseline requirement for a from-scratch architecture, demonstrating strong generalization to real-world, unstructured text data.

🧠 Methodology & Architecture
Instead of using a pre-trained classification model, this project involved building and customizing a Transformer from the ground up:

Transformer Adaptation: Converted a standard text-generation transformer into a binary classifier by implementing a mean pooling layer (torch.mean(x, dim=1)) across the time dimension, condensing 128 token embeddings into a single representation.

Classification Head: Added a linear output layer (nn.Linear) to map the pooled embeddings to 2 class probabilities (Positive/Negative).

Upgraded Configuration: Scaled the model capacity by utilizing an embedding dimension of 256, 6 stacked transformer blocks, and 8 attention heads, combined with a 0.2 dropout rate to prevent overfitting.

Tokenization: Utilized Hugging Face's bert-base-uncased subword tokenizer to efficiently handle padding, truncation, and out-of-vocabulary words without the bloat of character-level tokenization.

Data Pipelines: Built custom PyTorch Dataset and DataLoader classes to seamlessly batch, shuffle, and feed tokenized text tensors directly to the GPU.

💡 Key Learnings
Customizing Transformers: Gained practical experience modifying internal transformer mechanics (pooling and linear heads) to shift a model's objective from generation to classification.

PyTorch Ecosystem: Mastered end-to-end PyTorch workflows, including custom data pipelines, writing training loops from scratch, computing Cross-Entropy loss, and updating weights via the AdamW optimizer.

Model Optimization: Navigated the balance between model capacity and overfitting by tuning hyperparameters (epochs, dropout, embedding size) to maximize test accuracy on unseen data.

📂 Repository Structure
SentimentScope_starter.ipynb: The complete, fully executed Jupyter Notebook containing the data exploration, model architecture (DemoGPT), custom data loaders, and the training loop.

.gitignore: Configured to exclude massive dataset files and local Jupyter checkpoints from version control.

🚀 Reproduction Steps
Due to GitHub's file size limitations, the raw IMDB dataset is not included in this repository.