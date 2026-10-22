# Your-Own-AI-using-Java

A Java 21 / Spring Boot implementation of the core features in [Your-OWN-AI](https://github.com/narayaniduttpanday/Your-OWN-AI), a C++ project for vector search, an HNSW graph index, KD-Tree search, Ollama embeddings, and Retrieval-Augmented Generation (RAG).

The Java version is maintained separately. It does not modify the original C++ repository.

## What is implemented?

- **Brute-force nearest-neighbour search:** exact ranking for cosine, Euclidean, and Manhattan distance.
- **KD-Tree:** a balanced cyclic-axis KD-Tree with exact nearest-neighbour search for Euclidean and Manhattan distance. Cosine queries deliberately fall back to brute force because a standard KD split-plane bound cannot safely prune cosine distance in general.
- **HNSW:** a multilayer navigable small-world graph with randomized levels, greedy upper-layer descent, best-first layer search, neighbour selection, bidirectional link creation, and bounded connection lists. The implementation supports configurable construction parameters in code and uses separate graph instances for the supported distance metrics.
- **Vector management:** insert, list, and delete vectors. HNSW indexes receive incremental inserts; the balanced KD-Tree is rebuilt after insertion, and indexes are rebuilt after deletion.
- **Benchmarking:** compare in-process search latency and recall@k against the exact brute-force baseline.
- **Ollama integration:** dynamically sized embeddings from Ollama and non-streaming text generation.
- **Document retrieval and RAG:** overlapping text chunks, embedding-dimension validation, cosine retrieval, brute-force search for fewer than 10 chunks, HNSW search for 10 or more chunks, and context-aware answer generation.
- **REST API and browser UI:** served from the same Spring Boot application.

This is a compact educational implementation, not a line-by-line translation of the C++ source or a replacement for a heavily optimized vector database. Search quality and performance should be tested on your own data.

## Architecture

```text
Browser / curl
      |
      v
Spring Boot REST API (port 8080)
      |---------------------------|
      v                           v
Demo Vector Store             Document Service
  |       |       |                |
  v       v       v                v
Brute    KD-Tree HNSW          Ollama embeddings
force                           |
                                v
                       Brute force (< 10 chunks)
                       HNSW (>= 10 chunks)
                                |
                                v
                       Retrieved text context
                                |
                                v
                       Ollama text generation
```

## Requirements

- JDK 21 (recommended; the Maven build targets Java 21)
- Apache Maven 3.9+
- Git (only required for cloning)
- Ollama (optional for demo-vector search; required for document embeddings and AI-generated answers)

Verify Java and Maven:

```bash
java -version
mvn -version
```

### Install Java and Maven

**macOS with Homebrew** (after installing a JDK 21 distribution if needed):

```bash
brew install maven
```

**Windows:** Install JDK 21 from [Eclipse Temurin](https://adoptium.net/) or another trusted provider. Install Maven from the [official Apache Maven download page](https://maven.apache.org/download.cgi), configure `JAVA_HOME` and `PATH`, and reopen the terminal.

**Linux:** Install JDK 21 and Maven through your distribution's package manager or follow the [official Maven installation guide](https://maven.apache.org/install.html).

## Get the code

Clone the Java repository after it has been created and pushed to GitHub:

```bash
git clone https://github.com/narayaniduttpanday/Your-Own-AI-using-Java.git
cd Your-Own-AI-using-Java
```

If you are starting from the supplied ZIP instead, extract it and open a terminal in the extracted `Your-Own-AI-using-Java` directory. The root directory should contain `pom.xml`, `README.md`, and `src/`.

## Build and run

From the project root:

```bash
mvn clean package
```

This downloads Maven dependencies, compiles the Java source, and runs the automated tests. To start the application in development mode:

```bash
mvn spring-boot:run
```

Open **http://localhost:8080** in your browser. To stop the application, press `Ctrl+C` in the terminal.

To run the packaged JAR after a successful build:

```bash
java -jar target/your-own-ai-using-java-1.0.0.jar
```

If the exact JAR filename differs, inspect the `target/` directory and use the generated filename.

## Configure Ollama (for embeddings and RAG)

1. Install Ollama from [ollama.com](https://ollama.com/).
2. Start the Ollama desktop app/service, or run `ollama serve` if your installation requires it.
3. Pull the default embedding and generation models:

```bash
ollama pull nomic-embed-text
ollama pull llama3.2
```

4. Verify Ollama is reachable:

```bash
curl http://localhost:11434/api/tags
```

The application uses the following defaults:

| Setting | Default | Purpose |
|---|---|---|
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Ollama service address |
| `OLLAMA_EMBED_MODEL` | `nomic-embed-text` | Creates document/query embeddings |
| `OLLAMA_GENERATE_MODEL` | `llama3.2` | Generates RAG answers |

macOS/Linux example:

```bash
export OLLAMA_BASE_URL=http://localhost:11434
export OLLAMA_EMBED_MODEL=nomic-embed-text
export OLLAMA_GENERATE_MODEL=llama3.2
mvn spring-boot:run
```

Windows PowerShell example:

```powershell
$env:OLLAMA_BASE_URL = "http://localhost:11434"
$env:OLLAMA_EMBED_MODEL = "nomic-embed-text"
$env:OLLAMA_GENERATE_MODEL = "llama3.2"
mvn spring-boot:run
```

Set environment variables in the same terminal session that starts Spring Boot. Demo-vector search works without Ollama; `/doc/insert`, `/doc/search`, and RAG answer generation require the appropriate Ollama models.

## API reference

All endpoints are served from `http://localhost:8080`.

| Method | Endpoint | Purpose |
|---|---|---|
| `GET` | `/` | Browser interface |
| `GET` | `/status` | Ollama status, configured models, document dimensions/count, and demo-vector count |
| `GET` | `/stats` | Vector count, dimensions, algorithms, and metrics |
| `GET` | `/items` | List demo vectors |
| `GET` | `/search` | Search demo vectors |
| `POST` | `/insert` | Insert a demo vector |
| `DELETE` | `/delete/{id}` | Delete a demo vector |
| `GET` | `/benchmark` | Compare latency and recall@k across algorithms |
| `GET` | `/hnsw-info` | Inspect the active cosine HNSW graph |
| `POST` | `/doc/insert` | Chunk a document, embed it, and index its chunks |
| `GET` | `/doc/list` | List document chunks and previews |
| `DELETE` | `/doc/delete/{id}` | Delete a document chunk |
| `POST` | `/doc/search` | Retrieve relevant document chunks |
| `POST` | `/doc/ask` | Retrieve context and generate an answer with Ollama |

### Vector search

`GET /search` accepts:

- `v`: comma-separated vector containing exactly 16 finite numbers
- `k`: number of neighbours to return (default `5`)
- `metric`: `cosine`, `euclidean`, or `manhattan` (default `cosine`)
- `algo`: `bruteforce`, `kdtree`, or `hnsw` (default `hnsw`)

Example:

```bash
curl 'http://localhost:8080/search?v=0.9,0.8,0.7,0.6,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1&k=3&metric=cosine&algo=hnsw'
```

Each result includes the vector ID, metadata, category, distance, and embedding. The response also reports latency, the requested algorithm, and the actual algorithm. For a cosine query with `algo=kdtree`, `actualAlgo` is `bruteforce-fallback`; this is intentional and preserves correctness.

### Insert a vector

`POST /insert` JSON body:

```json
{
  "metadata": "Example document",
  "category": "demo",
  "embedding": [0.9, 0.8, 0.7, 0.6, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]
}
```

The embedding must contain exactly 16 finite numbers. The response contains the new vector ID.

### Delete a vector

```bash
curl -X DELETE http://localhost:8080/delete/1
```

Replace `1` with the actual ID. The response contains `{"ok":true}` if the item existed.

### Benchmark algorithms

```bash
curl 'http://localhost:8080/benchmark?v=0.9,0.8,0.7,0.6,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1&k=5&metric=euclidean'
```

The response includes `bruteforceUs`, `kdtreeUs`, and `hnswUs` timing values, plus recall@k values. Brute force is the exact reference. KD-Tree should have recall `1.0` for Euclidean and Manhattan distance. For cosine, KD-Tree uses the exact brute-force fallback, and its KD recall field is `null` because no KD traversal was performed. HNSW is approximate in principle; recall depends on dataset size and its graph/search parameters.

Timing values are microseconds measured in-process and can be noisy, especially with the small built-in dataset. Do not treat them as a reliable performance benchmark until you test a larger representative dataset over multiple runs.

### Document insertion

`POST /doc/insert` JSON body:

```json
{
  "title": "Introduction to AI",
  "text": "Artificial intelligence is a field of computer science..."
}
```

The service splits text into chunks of up to 250 words with a 30-word overlap, asks Ollama to embed each chunk, validates embedding dimensions, and stores the chunks in memory. The response contains generated IDs, chunk count, embedding dimensions, and the selected retrieval index strategy.

### Search document chunks

`POST /doc/search` JSON body:

```json
{
  "question": "What is artificial intelligence?",
  "k": 3
}
```

The question is embedded using the same configured embedding model. Retrieval uses cosine distance and returns sufficiently relevant chunks (cosine distance at most `0.7`). The document index uses brute force while there are fewer than 10 chunks and HNSW at 10 or more chunks.

### Ask a question using RAG

`POST /doc/ask` JSON body:

```json
{
  "question": "Explain artificial intelligence using the stored document.",
  "k": 3
}
```

The application embeds the question, retrieves the nearest chunks, builds a prompt containing their text, and sends it to the configured Ollama generation model. The response includes the generated `answer`, retrieved `contexts`, model name, and current document-chunk count.

### Inspect the HNSW graph

```bash
curl http://localhost:8080/hnsw-info
```

The response includes the graph's top layer, entry point, node count, nodes and edges per layer, graph nodes, connection lists, and HNSW parameters (`M`, `M0`, `efConstruction`, and `efSearch`). The endpoint reports the cosine-metric graph used for the default search configuration.

## Algorithms in this implementation

### Brute-force search

Computes the selected distance from the query to every stored vector, sorts all candidates by distance, and returns the first `k`. It is exact and is used as the baseline for correctness and recall tests. Its work scales linearly with the number of stored vectors per query, excluding sorting.

### KD-Tree

The tree is built by recursively choosing a cyclic dimension, sorting by that coordinate, and splitting at the median. Nearest-neighbour search explores the side of the split containing the query first, then explores the other side only when the distance to the split plane could improve the current result set.

This pruning bound is valid for Euclidean and Manhattan distance. It is **not generally valid for cosine distance**, so the API intentionally uses brute force for cosine queries instead of silently returning potentially incorrect results. The balanced tree is rebuilt after insertions and deletions to keep its height controlled.

### HNSW

The HNSW implementation creates nodes with randomized maximum layers, maintains neighbour lists per layer, searches greedily from the upper layers toward layer zero, and uses an `ef`-bounded best-first search for candidate expansion. It selects neighbours and prunes lists to configured connection limits (`M0 = 2M` at layer zero and `M` above it). Separate indexes are maintained for cosine, Euclidean, and Manhattan distance, so graph construction and query scoring use the same metric.

Vector insertion updates the HNSW graph incrementally. The in-memory store rebuilds the indexes after deletion to ensure deleted IDs cannot remain in the graph. The document service similarly maintains a cosine HNSW index and rebuilds it after deletion.

## Project structure

```text
Your-Own-AI-using-Java/
├── pom.xml
├── README.md
├── .gitignore
└── src/
    ├── main/
    │   ├── java/com/yourai/
    │   │   ├── YourOwnAiApplication.java
    │   │   ├── api/
    │   │   │   ├── ApiController.java
    │   │   │   └── ApiExceptionHandler.java
    │   │   ├── docs/DocumentService.java
    │   │   ├── ollama/OllamaService.java
    │   │   └── search/
    │   │       ├── BruteForceIndex.java
    │   │       ├── DistanceMetric.java
    │   │       ├── HnswIndex.java
    │   │       ├── KdTreeIndex.java
    │   │       ├── VectorItem.java
    │   │       └── VectorStore.java
    │   └── resources/
    │       ├── application.properties
    │       └── static/index.html
    └── test/java/com/yourai/
        ├── docs/DocumentServiceTest.java
        └── search/
            ├── DistanceMetricTest.java
            ├── SearchIndexTest.java
            └── VectorStoreTest.java
```

## Tests

Run the complete test suite:

```bash
mvn test
```

The tests cover distance calculations, KD-Tree exactness against brute force for Euclidean and Manhattan distance, cosine fallback behavior, HNSW search and graph construction, vector-store mutation, benchmark recall, dynamic Ollama embedding dimensions, and document RAG context construction.

Run a clean package build:

```bash
mvn clean package
```

## Troubleshooting

### `mvn: command not found`

Install Maven and reopen the terminal. Verify with `mvn -version`.

### Java version mismatch

Run `java -version` and `mvn -version`. The project targets Java 21; make sure Maven uses a compatible JDK.

### Port 8080 is already in use

Run on another port:

```bash
mvn spring-boot:run -Dspring-boot.run.arguments=--server.port=8081
```

Then open `http://localhost:8081`.

### Ollama is offline or a model is missing

Check `curl http://localhost:11434/api/tags`, then run:

```bash
ollama pull nomic-embed-text
ollama pull llama3.2
```

Ensure the environment variables point to the service and model names you installed.

### Embedding dimension mismatch

The application determines document embedding dimensions from the first stored document chunk. Use the same embedding model for all document insertions and question embeddings. If you change embedding models, clear/restart the in-memory document store before indexing documents with the new model.

### Search returns no document contexts

The document retrieval endpoint filters out chunks with cosine distance above `0.7`. Try a more specific question or insert more relevant source text.

## Limitations and differences from the C++ project

- This is a Java reimplementation of the core algorithms and API, not a line-for-line source translation.
- The demo vector store starts with 20 synthetic 16-dimensional vectors. These are example vectors, not embeddings generated from the demo metadata text.
- Demo vectors and document chunks are stored in memory and are lost when the application restarts. There is no persistent database configured.
- KD-Tree cosine queries intentionally use brute-force fallback for correctness; the original C++ KD-Tree attempts split-plane pruning for the selected distance function, which is not a valid general pruning rule for cosine distance.
- HNSW uses a compact implementation and default parameters (`M=16`, `efConstruction=200`, `efSearch=50`). It is not as optimized or extensively battle-tested as mature vector-search libraries.
- Indexes and search quality should be evaluated with real embeddings, realistic data sizes, and repeated benchmark runs before production use.
- Authentication, rate limiting, durable storage, and production deployment hardening are not included.

## Related project

Original C++ repository: [Your-OWN-AI](https://github.com/narayaniduttpanday/Your-OWN-AI)

The original repository remains unchanged by this Java project.

## License

Add an appropriate license before distributing the project. Review the original repository's license and any applicable terms before reusing its code or assets.
