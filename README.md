# Vector-Embedding

📄 Description
This project implements a comprehensive semantic search and content-based recommendation system using Latent Semantic Analysis (LSA) on a custom text corpus. It demonstrates the full pipeline from data collection and preprocessing to embedding generation, vector storage, semantic querying, personalized recommendations, rigorous evaluation, and insightful visualizations.

The goal is to explore how text documents can be represented numerically (vector embeddings) to enable intelligent information retrieval and discovery, moving beyond simple keyword matching to understanding the semantic meaning of content.

✨ Features
Data Collection: Utilizes a built-in 30-document corpus spanning 5 diverse domains (Technology, Science, Health, Business, Education) for easy reproducibility.
Text Preprocessing: Robust pipeline for cleaning, tokenization, stopword removal, and minimum token length filtering.
Tokenization Comparison: Demonstrates both traditional NLP preprocessing and advanced Byte-Pair Encoding (BPE) using tiktoken (GPT-2 tokenizer).
PyTorch Dataset & DataLoader: Prepares text for deep learning models using sliding window tokenization.
Vector Embedding Generation:
PyTorch nn.Embedding: Illustration of token and positional embeddings (untrained).
TF-IDF Vectorization: Captures term importance within documents relative to the corpus.
Latent Semantic Analysis (LSA): Uses Truncated SVD to reduce high-dimensional TF-IDF vectors into dense, low-dimensional semantic embeddings.
Vector Storage & Indexing: Efficient VectorStore class with:
LSA embedding matrix.
Document ID to index mapping.
Category-based indexing.
Inverted index for keyword pre-filtering.
Semantic Search Engine:
Transforms natural language queries into LSA embedding space.
Performs cosine similarity search to find semantically relevant documents.
Supports category-filtered searches and batch querying.
Recommendation System:
Document-based recommendations: Finds documents most similar to a given document.
User-based recommendations: Aggregates preferences from multiple liked documents for personalized suggestions.
Evaluation Metrics: Implements standard Information Retrieval metrics:
Precision@K (P@K)
Recall@K (R@K)
Mean Average Precision (MAP)
Normalized Discounted Cumulative Gain (NDCG@K)
Visualization: Generates a suite of publication-quality plots to explain and understand the system:
PCA & t-SNE 2D projections of embeddings.
Full cosine similarity heatmap.
Top TF-IDF terms per document.
SVD Scree Plot for explained variance.
Latent SVD Topics (top terms per component).
Search result similarity scores.
Evaluation metrics bar chart and per-query heatmap.
Recommendation star graph.
Corpus category distribution.
🚀 Getting Started
To run this project, you will need a Google Colab environment or a local Python environment with Jupyter Notebook support.

1. Clone the Repository
git clone https://github.com/your-username/your-repo-name.git
cd your-repo-name
(Replace your-username and your-repo-name with your actual GitHub details).

2. Open in Google Colab (Recommended)
The easiest way to get started is to open the notebook directly in Google Colab:

[redacted link]

Once opened, simply run all cells in order.

3. Local Setup (Optional)
If you prefer to run it locally:

# Create a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: .\venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Or manually install:
pip install tiktoken scikit-learn matplotlib seaborn pandas numpy torch --quiet

# Launch Jupyter Notebook
jupyter notebook
Then navigate to and open your_notebook_name.ipynb.

💻 Usage
Simply execute the cells sequentially in the provided Jupyter Notebook (your_notebook_name.ipynb). Each module is clearly delineated with markdown headings and code cells, demonstrating data loading, preprocessing, embedding generation, search functionality, recommendation, evaluation, and visualization.

Module 1: Data Collection & Preprocessing loads and cleans the text corpus.
Module 2: Vector Embedding Generation creates numerical representations of documents.
Module 3: Vector Storage & Indexing organizes these embeddings for efficient access.
Module 4: Semantic Search Engine allows you to query the corpus semantically.
Module 5: Recommendation System provides content-based recommendations.
Module 6: Evaluation Metrics quantifies the performance of the search system.
Module 7: Visualization offers graphical insights into the embeddings and system performance.
📊 Key Results & Visualizations
The project successfully demonstrates that LSA embeddings can effectively capture the semantic meaning of documents. Key outcomes include:

Clear Categorical Clustering: PCA and t-SNE plots visually confirm that documents from the same category cluster together in the embedding space.
Accurate Semantic Search: The search engine consistently retrieves highly relevant documents for natural language queries, even across different domains.
Effective Recommendations: Both document-based and user-based recommendation systems provide sensible suggestions aligned with content similarity.
Strong Evaluation Metrics: Achieves high scores on P@K, R@K, MAP, and NDCG@K, indicating robust search performance.
All generated plots (PCA, t-SNE, Heatmap, TF-IDF terms, SVD Scree, Latent Topics, Search Scores, Evaluation Metrics, Recommendation Graph, Category Distribution) are saved as .png files in the notebook's output directory and are viewable within the notebook itself.

🔮 Future Enhancements
Advanced Embeddings: Integrate and compare with state-of-the-art embeddings (e.g., Word2Vec, GloVe, FastText, or transformer-based models like BERT, Sentence-BERT) for potentially higher semantic accuracy.
Scalability: Implement more advanced vector databases (e.g., Pinecone, FAISS, Annoy) for handling larger corpora.
User Interface: Develop a simple web application (e.g., using Streamlit or Flask) to interact with the search and recommendation systems.
Dynamic Corpus: Allow users to upload their own datasets.
Real-time Processing: Explore methods for updating embeddings and indexes in real-time.
🤝 Contributing
Contributions are welcome! If you have suggestions for improvements, new features, or bug fixes, please open an issue or submit a pull request.
