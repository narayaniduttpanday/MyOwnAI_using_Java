package com.yourai.docs;

import com.yourai.ollama.OllamaService;
import com.yourai.search.BruteForceIndex;
import com.yourai.search.DistanceMetric;
import com.yourai.search.HnswIndex;
import com.yourai.search.VectorItem;
import org.springframework.stereotype.Service;

import java.util.*;
import java.util.concurrent.atomic.AtomicInteger;

/**
 * In-memory document store with Ollama embeddings and cosine-based retrieval.
 */
@Service
public class DocumentService {
    public record DocChunk(int id, String title, String text, List<Double> embedding) {
        public DocChunk {
            embedding = List.copyOf(embedding);
        }
    }

    private static final int CHUNK_WORDS = 250;
    private static final int CHUNK_OVERLAP = 30;
    private static final int HNSW_THRESHOLD = 10;
    private static final double MAX_COSINE_DISTANCE = 0.7;

    private final Map<Integer, DocChunk> docs = new LinkedHashMap<>();
    private final AtomicInteger ids = new AtomicInteger(1);
    private final OllamaService ollama;
    private HnswIndex hnswIndex = new HnswIndex(List.of(), DistanceMetric.COSINE);
    private int embeddingDimensions;

    public DocumentService(OllamaService ollama) {
        this.ollama = ollama;
    }

    public synchronized Map<String, Object> insert(String title, String text) {
        if (title == null || title.isBlank() || text == null || text.isBlank()) {
            throw new IllegalArgumentException("title and text are required");
        }
        List<String> chunks = chunk(text, CHUNK_WORDS, CHUNK_OVERLAP);
        List<List<Double>> embeddings = new ArrayList<>(chunks.size());
        int expectedDimensions = embeddingDimensions;

        // Prepare and validate all embeddings first so a failed Ollama request cannot leave a partial document.
        for (String chunk : chunks) {
            List<Double> embedding;
            try {
                embedding = ollama.embed(chunk);
            } catch (Exception ex) {
                throw new IllegalStateException("Could not generate an embedding. Check that Ollama is running and the embedding model is installed.", ex);
            }
            if (embedding == null || embedding.isEmpty()) {
                throw new IllegalStateException("Ollama returned no embedding. Start Ollama and pull the configured embedding model.");
            }
            validateEmbedding(embedding);
            if (expectedDimensions == 0) expectedDimensions = embedding.size();
            if (embedding.size() != expectedDimensions) {
                throw new IllegalStateException("Embedding dimension mismatch: expected " + expectedDimensions + " but received " + embedding.size());
            }
            embeddings.add(List.copyOf(embedding));
        }

        List<Integer> created = new ArrayList<>();
        for (int i = 0; i < chunks.size(); i++) {
            String chunkTitle = chunks.size() == 1 ? title : title + " [" + (i + 1) + "/" + chunks.size() + "]";
            int id = ids.getAndIncrement();
            DocChunk docChunk = new DocChunk(id, chunkTitle, chunks.get(i), embeddings.get(i));
            docs.put(id, docChunk);
            hnswIndex.insert(asVectorItem(docChunk));
            created.add(id);
        }
        embeddingDimensions = expectedDimensions;
        return Map.of("ids", created, "chunks", chunks.size(), "dims", embeddingDimensions,
                "index", docs.size() >= HNSW_THRESHOLD ? "hnsw" : "bruteforce");
    }

    public synchronized List<Map<String, Object>> list() {
        return docs.values().stream().map(d -> Map.<String, Object>of("id", d.id(), "title", d.title(),
                "preview", d.text().substring(0, Math.min(120, d.text().length())),
                "words", d.text().trim().isEmpty() ? 0 : d.text().trim().split("\\s+").length,
                "dimensions", d.embedding().size())).toList();
    }

    public synchronized boolean delete(int id) {
        boolean removed = docs.remove(id) != null;
        if (removed) {
            if (docs.isEmpty()) embeddingDimensions = 0;
            rebuildHnswIndex();
        }
        return removed;
    }

    public synchronized int size() {
        return docs.size();
    }

    public synchronized int dimensions() {
        return embeddingDimensions;
    }

    public synchronized String indexAlgorithm() {
        return docs.size() >= HNSW_THRESHOLD ? "hnsw" : "bruteforce";
    }

    public synchronized List<Map<String, Object>> retrieve(String question, int k) {
        if (question == null || question.isBlank()) throw new IllegalArgumentException("question is required");
        if (k < 1) throw new IllegalArgumentException("k must be >= 1");
        if (docs.isEmpty()) return List.of();
        List<Double> query;
        try {
            query = ollama.embed(question);
        } catch (Exception ex) {
            throw new IllegalStateException("Could not embed the question. Check Ollama and the embedding model.", ex);
        }
        if (query == null || query.isEmpty()) throw new IllegalStateException("Ollama returned no query embedding");
        validateEmbedding(query);
        if (query.size() != embeddingDimensions) {
            throw new IllegalStateException("Question embedding has " + query.size() + " dimensions, but stored documents use " + embeddingDimensions + ". Ensure the same embedding model is used.");
        }

        List<BruteForceIndex.Neighbor> hits;
        if (docs.size() >= HNSW_THRESHOLD) {
            hits = hnswIndex.search(query, Math.min(k, docs.size()), DistanceMetric.COSINE);
        } else {
            List<VectorItem> vectors = docs.values().stream().map(this::asVectorItem).toList();
            hits = BruteForceIndex.search(vectors, query, Math.min(k, docs.size()), DistanceMetric.COSINE);
        }
        return hits.stream().filter(hit -> hit.distance() <= MAX_COSINE_DISTANCE).map(hit -> {
            DocChunk d = docs.get(hit.item().id());
            return Map.<String, Object>of("id", d.id(), "title", d.title(), "text", d.text(), "distance", hit.distance());
        }).toList();
    }

    public Map<String, Object> askWithContext(String question, int k) {
        List<Map<String, Object>> hits = retrieve(question, k);
        if (size() == 0) {
            return Map.of("answer", "No documents have been indexed yet. Insert a document before asking a question.",
                    "contexts", List.of());
        }
        StringBuilder context = new StringBuilder();
        for (int i = 0; i < hits.size(); i++) {
            context.append("[").append(i + 1).append("] ").append(hits.get(i).get("title"))
                    .append(":\n").append(hits.get(i).get("text")).append("\n\n");
        }

        String prompt = """
                You are a document question-answering assistant.
                Answer the question using ONLY the retrieved context below.
                Do not use outside knowledge or invent facts.
                If the context does not contain enough information to answer,
                respond exactly: "The provided documents do not contain enough information to answer this question."
                Be concise and directly answer the question.
                
                Retrieved context:
                %s
                
                Question: %s
                
                Answer:
                """.formatted(context, question);

        try {
            String answer = ollama.generate(prompt);
            if (answer == null || answer.isBlank()) throw new IllegalStateException("Ollama returned an empty answer");
            return Map.of("answer", answer, "contexts", hits);
        } catch (Exception ex) {
            throw new IllegalStateException("Could not generate an answer. Check Ollama and the configured generation model.", ex);
        }
    }

    private VectorItem asVectorItem(DocChunk chunk) {
        return new VectorItem(chunk.id(), chunk.title(), "document", chunk.embedding());
    }

    private void rebuildHnswIndex() {
        List<VectorItem> vectors = docs.values().stream().map(this::asVectorItem).toList();
        hnswIndex = new HnswIndex(vectors, DistanceMetric.COSINE);
    }

    private static void validateEmbedding(List<Double> embedding) {
        for (Double value : embedding) {
            if (value == null || !Double.isFinite(value)) throw new IllegalStateException("Embedding contains a non-finite value");
        }
    }

    private static List<String> chunk(String text, int size, int overlap) {
        String trimmed = text.trim();
        if (trimmed.isEmpty()) return List.of();
        String[] words = trimmed.split("\\s+");
        List<String> output = new ArrayList<>();
        for (int start = 0; start < words.length; start += Math.max(1, size - overlap)) {
            output.add(String.join(" ", Arrays.copyOfRange(words, start, Math.min(words.length, start + size))));
            if (start + size >= words.length) break;
        }
        return output;
    }
}
