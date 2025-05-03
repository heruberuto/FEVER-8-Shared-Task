import json, random,os
import matplotlib.pyplot as plt
from langchain_core.example_selectors import (
    MaxMarginalRelevanceExampleSelector,
    SemanticSimilarityExampleSelector,
)
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_text_splitters import CharacterTextSplitter, RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_community.retrievers import KNNRetriever
from IPython.display import display, Markdown, Latex
from sentence_transformers import CrossEncoder
from langchain_community.vectorstores import FAISS
from tqdm import tqdm
from langchain_community.retrievers import BM25Retriever
#/mnt/data/factcheck/averitec-data/data_store/vecstore/dev
embeddings = HuggingFaceEmbeddings(model_name="mixedbread-ai/mxbai-embed-large-v1")
random.seed(111)

DATA_DIR = "/mnt/data/factcheck/averitec-data/data_store"
SPLIT = "dev"

#with open(f"{DATA_DIR}/{SPLIT}.json") as f:
#    datapoints = json.load(f)
# start=(int(os.environ['SLURM_JOB_ID'])%10)*100,
for CLAIM_ID in tqdm(range(500)):# Naive version with \n concatenated url2texts:   
    # skip if f"{DATA_DIR}/data_store/vecstore/{SPLIT}/full/{CLAIM_ID}" exists
    if os.path.exists(f"{DATA_DIR}/data_store/vecstore/{SPLIT}/full/{CLAIM_ID}"):
        continue
    
    os.makedirs(f"{DATA_DIR}/data_store/vecstore/{SPLIT}/full/{CLAIM_ID}")
    # make dummy
    
    # display(Markdown("### 🗯️ " + claim + " [" + datapoint["label"] + "]"))
    docstore = []
    #for line in open(f"{DATA_DIR}/{SPLIT}/{CLAIM_ID}.json"):
    for line in open(f"/mnt/data/factcheck/averitec-data/data_store/output_dev/{CLAIM_ID}.json"):
        docstore.append(json.loads(line))
        
    documents = [
        Document(
            page_content=" ".join(doc["url2text"]),
            metadata={
                "url": doc["url"],
                # "sentences": doc["url2text"]
            },
        )
        for doc in docstore
    ]

    TOKENS_PER_CHAR = 0.25
    EMBEDDING_INPUT_SIZE = 512

    chunks = []
    for doc in docstore:
        if not doc["url2text"]:
            continue
        buffer = ""
        for i, sentence in enumerate(doc["url2text"]):
            if (
                i == len(doc["url2text"]) - 1
                or len(buffer) + len(sentence) >= EMBEDDING_INPUT_SIZE / TOKENS_PER_CHAR
            ):
                context_before = ""
                if chunks and chunks[-1].metadata["url"] == doc["url"]:
                    chunks[-1].metadata["context_after"] = buffer
                    context_before = chunks[-1].page_content
                chunks.append(
                    Document(
                        page_content=buffer,
                        metadata={"url": doc["url"], "context_before": context_before, "context_after": ""},
                    )
                )

                buffer = ""
            buffer += sentence + " "
    # chunk the documents into smaller pieces
    chid = random.randint(0, len(chunks))

    # display(Markdown(chunks[chid].metadata["context_before"]))
    # display(Markdown(chunks[chid].page_content))
    # display(Markdown(chunks[chid].metadata["context_after"]))
    
    #print(chunks[chid].page_content)
    #retriever = BM25Retriever.from_documents(
    #    chunks, k=4000
    #)
    chunks_pruned = chunks #retriever.invoke(claim)
    db = FAISS.from_documents(chunks_pruned, embeddings)
    db.save_local(f"{DATA_DIR}/data_store/vecstore/{SPLIT}/full/{CLAIM_ID}")
                          