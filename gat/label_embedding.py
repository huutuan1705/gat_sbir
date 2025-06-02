import os
import numpy as np
import torch
import torch.nn as nn

class LabelEmbeddings(nn.Module):
    def __init__(self, num_classes: int, embedding_dim: int, 
                 glove_file_path: str = None, label_names: list = None):
        """
        Args:
            num_classes (int): Total number of unique labels in the dataset.
            embedding_dim (int): Dimension of the label embeddings.
            glove_file_path (str, optional): Path to the pre-trained GloVe file.
                                            If None, embeddings are randomly initialized.
            label_names (list of str, optional): List of actual string names for each class index.
                                                 Required if using GloVe and handling multi-word labels.
                                                 The order must correspond to class indices 0 to num_classes-1.
        """
        super(LabelEmbeddings, self).__init__()
        self.num_classes = num_classes
        self.embedding_dim = embedding_dim
        
        self.embedding_layer = nn.Embedding(num_classes, embedding_dim)
        
        if glove_file_path and label_names:
            if len(label_names) != num_classes:
                raise ValueError("Length of label_names must match num_classes.")
            print(f"Attempting to load GloVe embeddings from: {glove_file_path}")
            self._load_glove_embeddings(glove_file_path, label_names)
        else:
            print("Initializing label embeddings randomly (no GloVe path or label names provided).")
        
    def _load_glove_embeddings(self, glove_file_path: str, label_names: list):
        """
        Loads GloVe embeddings for the given label names and initializes the embedding layer.
        Handles multi-word labels by averaging their GloVe vectors[cite: 113].
        """
        # 1. Load GloVe file into a dictionary: word -> vector
        glove_vectors = {}
        try:
            with open(glove_file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    parts = line.strip().split()
                    word = parts[0]
                    vector = np.array(parts[1:], dtype=np.float32)
                    if len(vector) == self.embedding_dim: # Ensure consistent embedding dim
                        glove_vectors[word] = vector
                    
                            
        except Exception as e:
            print(f"Error loading GloVe file: {e}. Using random embeddings.")
            return
        
        # 2. Create embedding matrix for our labels
        initial_weights = np.random.rand(self.num_classes, self.embedding_dim).astype(np.float32) * 0.1 # Small random values
        found_count = 0

        for i, label_name_phrase in enumerate(label_names):
            # Process label name (e.g., "using_phone" -> ["using", "phone"])
            # The paper implies averaging word embeddings for multi-word category names[cite: 113].
            words_in_label = label_name_phrase.lower().replace('_', ' ').split()
            
            word_vectors_for_label = []
            for word in words_in_label:
                if word in glove_vectors:
                    word_vectors_for_label.append(glove_vectors[word])
            
            if word_vectors_for_label:
                # Average the embeddings for all words in the label phrase [cite: 113]
                avg_vector = np.mean(word_vectors_for_label, axis=0)
                initial_weights[i] = avg_vector
                found_count += 1
            else:
                print(f"Warning: No GloVe vectors found for any word in label '{label_name_phrase}'. " \
                      f"Using random initialization for this label.")

        self.embedding_layer.weight.data.copy_(torch.from_numpy(initial_weights))
        
    def forward(self, label_indices: torch.Tensor) -> torch.Tensor:
        """
        Retrieves embeddings for the given label indices.

        Args:
            label_indices (torch.Tensor): A tensor of label indices.
                                          Typically torch.arange(self.num_classes) to get all label embeddings.

        Returns:
            torch.Tensor: Embeddings for the specified labels, shape (len(label_indices), embedding_dim).
        """
        return self.embedding_layer(label_indices)