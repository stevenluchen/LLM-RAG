import os
from collections import defaultdict
from typing import Any, Dict, List

import numpy as np
import ray
import torch
import vllm
from blingfire import text_to_sentences_and_offsets
from bs4 import BeautifulSoup
from sentence_transformers import SentenceTransformer

from openai import OpenAI

from tqdm import tqdm
from rank_bm25 import BM25Okapi
import faiss
from sklearn.preprocessing import MinMaxScaler

#### CONFIG PARAMETERS ---

# Define the number of context sentences to consider for generating an answer.
NUM_CONTEXT_SENTENCES = 30
# Set the maximum length for each context sentence (in characters).
MAX_CONTEXT_SENTENCE_LENGTH = 1500
# Set the maximum context references length (in characters).
MAX_CONTEXT_REFERENCES_LENGTH = 4500

# Batch size you wish the evaluators will use to call the `batch_generate_answer` function
AICROWD_SUBMISSION_BATCH_SIZE = 1 # TUNE THIS VARIABLE depending on the number of GPUs you are requesting and the size of your model.

# VLLM Parameters 
VLLM_TENSOR_PARALLEL_SIZE = 1 # TUNE THIS VARIABLE depending on the number of GPUs you are requesting and the size of your model.
VLLM_GPU_MEMORY_UTILIZATION = 0.85 # TUNE THIS VARIABLE depending on the number of GPUs you are requesting and the size of your model.

# Sentence Transformer Parameters
SENTENTENCE_TRANSFORMER_BATCH_SIZE = 32 # TUNE THIS VARIABLE depending on the size of your embedding model and GPU mem available

class ChunkExtractor:
    
    def paragraph_chunking(self, text, window_size=5, overlap=2):
        """Splits text into paragraphs, then further breaks long paragraphs into sentence windows."""
        paragraphs = [p.strip() for p in text.split("\n\n") if p.strip() and len(p.strip()) > 40]

        if not paragraphs:  # If no valid paragraphs, fall back to sentence chunking
            return self.sentence_chunking(text)
        
        chunks = []
        for paragraph in paragraphs:
            sentences = paragraph.split(". ")  # Split into sentences
            for i in range(0, len(sentences), window_size - overlap):
                chunk = ". ".join(sentences[i:i+window_size])
                chunks.append(chunk[:MAX_CONTEXT_SENTENCE_LENGTH])  # Truncate to max length
        return chunks
    
    def sentence_chunking(self, text):
        """Falls back to sentence-based chunking if paragraph chunking is not viable."""
        _, offsets = text_to_sentences_and_offsets(text)
        return [text[start:end][:MAX_CONTEXT_SENTENCE_LENGTH] for start, end in offsets]
    
    def clean_html(self, html_source):
        """Cleans HTML content by removing unnecessary elements and extracting the title."""
        soup = BeautifulSoup(html_source, "lxml")

        # Remove non-relevant elements
        for element in soup.find_all(['script', 'style', 'nav', 'footer', 'header']):
            element.decompose()

        # Extract main text
        text = soup.get_text(" ", strip=True)

        # Extract title if available
        title = ""
        if soup.title:
            title = soup.title.get_text().strip()

        return title, text

    @ray.remote
    def _extract_chunks(self, interaction_id, html_source):
        """Extracts chunks from HTML content while preserving context using paragraph-based chunking."""
        title, text = self.clean_html(html_source)

        if not text:
            return interaction_id, [""]

        # Apply paragraph chunking first, then fall back to sentence chunking
        chunks = self.paragraph_chunking(text)

        # Add title as an additional chunk if it exists
        if title:
            chunks.insert(0, title)

        # Remove empty and duplicate chunks
        chunks = [chunk for chunk in chunks if chunk.strip()]
        chunks = list(dict.fromkeys(chunks))

        return interaction_id, chunks

    def extract_chunks(self, batch_interaction_ids, batch_search_results):
        """Extracts chunks from batch search results using parallel processing with Ray."""
        ray_response_refs = [
            self._extract_chunks.remote(self, batch_interaction_ids[idx], html_text["page_result"])
            for idx, search_results in enumerate(batch_search_results)
            for html_text in search_results
        ]

        chunk_dictionary = defaultdict(list)
        for response_ref in ray_response_refs:
            interaction_id, _chunks = ray.get(response_ref)
            chunk_dictionary[interaction_id].extend(_chunks)

        return self._flatten_chunks(chunk_dictionary)

    def _flatten_chunks(self, chunk_dictionary):
        """Flattens chunk dictionary into lists of chunks and their corresponding interaction IDs."""
        chunks, chunk_interaction_ids = [], []
        for interaction_id, _chunks in chunk_dictionary.items():
            unique_chunks = list(set(_chunks))  # Remove duplicates
            chunks.extend(unique_chunks)
            chunk_interaction_ids.extend([interaction_id] * len(unique_chunks))
        return np.array(chunks), np.array(chunk_interaction_ids)


class RAGModel:
    """
    An example RAGModel for the KDDCup 2024 Meta CRAG Challenge
    which includes all the key components of a RAG lifecycle.
    """
    def __init__(self, llm_name="meta-llama/Llama-3.2-3B-Instruct", is_server=False, vllm_server=None):
        self.initialize_models(llm_name, is_server, vllm_server)
        self.chunk_extractor = ChunkExtractor()

    def initialize_models(self, llm_name, is_server, vllm_server):
        self.llm_name = llm_name
        self.is_server = is_server
        self.vllm_server = vllm_server

        if self.is_server:
            # initialize the model with vllm server
            openai_api_key = "EMPTY"
            openai_api_base = self.vllm_server
            self.llm_client = OpenAI(
                api_key=openai_api_key,
                base_url=openai_api_base,
            )
        else:
            # initialize the model with vllm offline inference
            self.llm = vllm.LLM(
                model=self.llm_name,
                worker_use_ray=True,
                tensor_parallel_size=VLLM_TENSOR_PARALLEL_SIZE,
                gpu_memory_utilization=VLLM_GPU_MEMORY_UTILIZATION,
                trust_remote_code=True,
                dtype="half",  # note: bfloat16 is not supported on nvidia-T4 GPUs
                enforce_eager=True
            )
            self.tokenizer = self.llm.get_tokenizer()

        # Load a sentence transformer model optimized for sentence embeddings, using CUDA if available.
        self.sentence_model = SentenceTransformer(
            "all-MiniLM-L6-v2",
            device=torch.device(
                "cuda" if torch.cuda.is_available() else "cpu"
            ),
        )
        
        
        # Initialize FAISS index for dense retrieval
        self.faiss_index = None
        self.dense_docs = []  # Store original chunks
        self.bm25 = None  # Initialize BM25 later

    def calculate_embeddings(self, sentences):
        """
        Compute normalized embeddings for a list of sentences using a sentence encoding model.

        This function leverages multiprocessing to encode the sentences, which can enhance the
        processing speed on multi-core machines.

        Args:
            sentences (List[str]): A list of sentences for which embeddings are to be computed.

        Returns:
            np.ndarray: An array of normalized embeddings for the given sentences.

        """
        embeddings = self.sentence_model.encode(
            sentences=sentences,
            normalize_embeddings=True,
            batch_size=SENTENTENCE_TRANSFORMER_BATCH_SIZE,
        )
        # Note: There is an opportunity to parallelize the embedding generation across 4 GPUs
        #       but sentence_model.encode_multi_process seems to interefere with Ray
        #       on the evaluation servers. 
        #       todo: this can also be done in a Ray native approach.
        #       
        return embeddings
    
    
    def build_retrieval_index(self, chunks, chunk_interaction_ids):
        """
        Builds separate BM25 and FAISS retrieval indices per transaction ID.
        """
        self.transaction_faiss = {}  # Store FAISS index per transaction
        self.transaction_bm25 = {}   # Store BM25 model per transaction
        self.transaction_docs = {}   # Store original chunks per transaction

        unique_interaction_ids = set(chunk_interaction_ids)

        for interaction_id in unique_interaction_ids:
            # Get all chunks belonging to this transaction
            transaction_chunks = [chunks[i] for i in range(len(chunks)) if chunk_interaction_ids[i] == interaction_id]

            if not transaction_chunks:
                continue  # Skip if no chunks for this transaction
            
            # Store chunks for later use
            self.transaction_docs[interaction_id] = transaction_chunks

            # BM25 Index for this transaction
            tokenized_chunks = [chunk.split() for chunk in transaction_chunks]
            self.transaction_bm25[interaction_id] = BM25Okapi(tokenized_chunks)

            # FAISS Index for this transaction
            dense_embeddings = self.calculate_embeddings(transaction_chunks)
            faiss.normalize_L2(dense_embeddings)
            dimension = dense_embeddings.shape[1]
            faiss_index = faiss.IndexFlatIP(dimension)
            faiss_index.add(dense_embeddings)

            self.transaction_faiss[interaction_id] = faiss_index
        
    def adjust_weights(self,query):
        if "who" in query.lower() or "when" in query.lower() or "which" in query.lower():
            return 0.7, 0.3  # More weight to BM25
        else:
            return 0.3, 0.7  # More weight to FAISS

    def hybrid_retrieve(self, query, interaction_id, top_k=10):
        if interaction_id not in self.transaction_faiss or interaction_id not in self.transaction_bm25:
            return []  # No retrieval index for this transaction

        # Retrieve documents using FAISS
        query_embedding = self.calculate_embeddings([query])
        faiss.normalize_L2(query_embedding)
        faiss_index = self.transaction_faiss[interaction_id]
        faiss_scores, faiss_indices = faiss_index.search(query_embedding, len(self.transaction_docs[interaction_id]))

        # Retrieve documents using BM25
        bm25_model = self.transaction_bm25[interaction_id]
        bm25_scores = bm25_model.get_scores(query.split())


        # Normalize scores
        faiss_scores = faiss_scores.flatten()
        bm25_scores = 1/(1+np.exp(-bm25_scores/10))
        print("max vals",max(faiss_scores),max(bm25_scores))
        # Dynamic weight assignment

        bm25_weight, faiss_weight = self.adjust_weights(query)

        # Combine scores
        hybrid_scores = {}
        for idx, score in zip(faiss_indices[0], faiss_scores):

            hybrid_scores[idx] = hybrid_scores.get(idx, 0) + faiss_weight * score
        for idx, score in enumerate(bm25_scores):
            hybrid_scores[idx] = hybrid_scores.get(idx, 0) + bm25_weight * score

        # Sort by highest combined score
        top_indices = sorted(hybrid_scores.keys(), key=lambda i: hybrid_scores[i], reverse=True)[:top_k]
        # Confidence score is the highest hybrid score (normalized between 0-1)
        confidence_score = max(hybrid_scores.values(), default=0.0)
        print("confidence (max) score is ",confidence_score)
        return [self.transaction_docs[interaction_id][i] for i in top_indices],confidence_score
    
    def generate_dynamic_prompt(self, query, query_time,retrieved_docs, confidence_score):
        """
        Generates an LLM prompt dynamically based on retrieval confidence.
        """
        
        if confidence_score < 0.5:
            uncertainty_statement = "I'm not completely certain about this answer, but based on available references:"
        elif confidence_score < 0.7:
            uncertainty_statement = "Here is what I found based on references, but some details might be missing:"
        else:
            uncertainty_statement = "Based on the retrieved references, here is a direct answer:"

        return f"""
        {uncertainty_statement} 
        ##Current Time:
        {query_time}   
        ##Question:
        {query}
        {retrieved_docs}
        --------
        
        Please generate a response strictly based on the references above.\n
        """
    def get_batch_size(self) -> int:
        """
        Determines the batch size that is used by the evaluator when calling the `batch_generate_answer` function.
        
        The evaluation timeouts linearly scale with the batch size. 
            i.e.: time out for the `batch_generate_answer` call = batch_size * per_sample_timeout 
        

        Returns:
            int: The batch size, an integer between 1 and 16. It can be dynamic
                 across different batch_generate_answer calls, or stay a static value.
        """
        self.batch_size = AICROWD_SUBMISSION_BATCH_SIZE  
        return self.batch_size

    def batch_generate_answer(self, batch: Dict[str, Any]) -> List[str]:
        """
        Generates answers for a batch of queries using associated (pre-cached) search results and query times.

        Parameters:
            batch (Dict[str, Any]): A dictionary containing a batch of input queries with the following keys:
                - 'interaction_id;  (List[str]): List of interaction_ids for the associated queries
                - 'query' (List[str]): List of user queries.
                - 'search_results' (List[List[Dict]]): List of search result lists, each corresponding
                                                      to a query. Please refer to the following link for
                                                      more details about the individual search objects:
                                                      https://gitlab.aicrowd.com/aicrowd/challenges/meta-comprehensive-rag-benchmark-kdd-cup-2024/meta-comphrehensive-rag-benchmark-starter-kit/-/blob/master/docs/dataset.md#search-results-detail
                - 'query_time' (List[str]): List of timestamps (represented as a string), each corresponding to when a query was made.

        Returns:
            List[str]: A list of plain text responses for each query in the batch. Each response is limited to 75 tokens.
            If the generated response exceeds 75 tokens, it will be truncated to fit within this limit.

        Notes:
        - If the correct answer is uncertain, it's preferable to respond with "I don't know" to avoid
          the penalty for hallucination.
        - Response Time: Ensure that your model processes and responds to each query within 30 seconds.
          Failing to adhere to this time constraint **will** result in a timeout during evaluation.
        """
        batch_interaction_ids = batch["interaction_id"]
        queries = batch["query"]
        batch_search_results = batch["search_results"]
        query_times = batch["query_time"]

        # Chunk all search results using ChunkExtractor
        chunks, chunk_interaction_ids = self.chunk_extractor.extract_chunks(
            batch_interaction_ids, batch_search_results
        )
         
        #hybrid version
        
        # Build separate retrieval index per transaction
        self.build_retrieval_index(chunks, chunk_interaction_ids)
        confidence_scores = []
        # Retrieve top matches per query, per transaction
        batch_retrieval_results = []
        for _idx, interaction_id in enumerate(batch_interaction_ids):
            query = queries[_idx]
            query_time = query_times[_idx]
            retrieval_results,confidence_score = self.hybrid_retrieve(query, interaction_id, top_k=NUM_CONTEXT_SENTENCES)
            batch_retrieval_results.append(retrieval_results)
            confidence_scores.append(confidence_score)
        # Prepare formatted prompts from the LLM        
        formatted_prompts = self.format_prompts(queries, query_times, batch_retrieval_results,confidence_scores)

        # Generate responses via vllm
        # note that here self.batch_size = 1
        if self.is_server:
            response = self.llm_client.chat.completions.create(
                model=self.llm_name,
                messages=formatted_prompts[0],
                n=1,  # Number of output sequences to return for each prompt.
                top_p=0.9,  # Float that controls the cumulative probability of the top tokens to consider.
                temperature=0.1,  # randomness of the sampling
                #skip_special_tokens=True,  # Whether to skip special tokens in the output.
                max_tokens=50,  # Maximum number of tokens to generate per output sequence.
            )
            answers = [response.choices[0].message.content]
        else:
            responses = self.llm.generate(
                formatted_prompts,
                vllm.SamplingParams(
                    n=1,  # Number of output sequences to return for each prompt.
                    top_p=0.9,  # Float that controls the cumulative probability of the top tokens to consider.
                    temperature=0.1,  # randomness of the sampling
                    skip_special_tokens=True,  # Whether to skip special tokens in the output.
                    max_tokens=50,  # Maximum number of tokens to generate per output sequence.
                ),
                use_tqdm=False
            )
            answers = []
            for response in responses:
                answers.append(response.outputs[0].text)

        return answers

    def format_prompts(self, queries, query_times, batch_retrieval_results=[],confidence_scores=[]):
        """
        Formats queries, corresponding query_times and retrieval results using the chat_template of the model.
            
        Parameters:
        - queries (List[str]): A list of queries to be formatted into prompts.
        - query_times (List[str]): A list of query_time strings corresponding to each query.
        - batch_retrieval_results (List[str])
        """        
        system_prompt = "You are an accurate, helpful assistant. When answering questions based on references, cite specific facts from the provided references. Your answers should be clear, precise, and directly address the query. If the references don't contain relevant information, respond with 'I don't know' rather than attempting to guess. Focus on brevity, factual accuracy, and avoid any unnecessary elaboration."
        formatted_prompts = []

        for _idx, query in enumerate(queries):
            query_time = query_times[_idx]
            retrieval_results = batch_retrieval_results[_idx]
            confidence_score=confidence_scores[_idx]
            user_message = ""
            references = ""

            if len(retrieval_results) > 0:
                references += "# References \n"
                # Format the top sentences as references in the model's prompt template.
                for _snippet_idx, snippet in enumerate(retrieval_results):
                    references += f"- {snippet.strip()}\n"
            
            references = references[:MAX_CONTEXT_REFERENCES_LENGTH]
   
            user_message=self.generate_dynamic_prompt(query,query_time,references,confidence_score)



            if self.is_server:
                # there is no need to wrap the messages into chat when using the server
                # because we use the chat API: chat.completions.create
                formatted_prompts.append(
                    [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_message},
                    ]
                )
            else:
                formatted_prompts.append(
                    self.tokenizer.apply_chat_template(
                        [
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": user_message},
                        ],
                        tokenize=False,
                        add_generation_prompt=True,
                    )
                )

        return formatted_prompts
