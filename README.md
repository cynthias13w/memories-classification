# memories-classification

Understanding the relationship between memories and storytelling holds profound significance as storytelling serves as a mechanism for humans to share experiences shaping our culture and communication. Imagination, recollection, and retelling rely on different cognitive and memory processes, thereby suggesting that this distinction might also be reflected in the language employed during the process of storytelling.

In this project, we explored the intricate interplay between memory processes and storytelling by contrasting narratives derived from imagined, recalled, and retold stories using "Hippocorpus", a comprehensive dataset of 6,854 English diary-like short stories (Sap et al., 2022), and by leveraging the capabilities of the most recent Natural Language Processing (NLP) techniques and Deep Learning algorithms.

Firstly, as a baseline in our project, we implemented a Machine Learning approach using several classifiers fed with a bag-of-words (bow) and term frequency- inverse document frequency (tf-idf) representation of our textual data, achieving 53% accuracy.

Secondly, we build a fully connected deep neural network (i.e., Multilayer Perceptron) featuring one embedding layer, a hidden layer, and an output layer. The vectorized inputs fed into our custom-built classification model were computed using three distinct word embedding algorithms such as Glove, Fasttext, and Word2vec, which allowed us to convert our textual data into vectors. As a result, our model achieved an average accuracy of approximately 50%.

Deep learning techniques still offer a promising avenue for investigating and classifying narratives originating from different types of memory. By continually refining our approach, e.g., using regularization methods, leveraging data augmentation and sampling techniques, and by exploring more advanced methodologies, such as Recurrent Neural Network (RNN), Long-Short Term Memory models (LSTM), Transformer models or fine-tuning pre-trained Large Language Models (LLM), we aim to enhance the accuracy and richness of our understanding.
