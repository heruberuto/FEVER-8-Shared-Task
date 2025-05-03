BASE_PATH="."
mkdir -p $BASE_PATH/vector_store/test_2025/
wget "https://huggingface.co/datasets/ctu-aic/averitec-embeddings/resolve/main/test_2025.zip" \
    -O "$BASE_PATH/vector_store/test_2025.zip" 
unzip "$BASE_PATH/vector_store/test_2025.zip" -d "$BASE_PATH/vector_store/test_2025" && \
rm "$BASE_PATH/vector_store/test_2025.zip"
